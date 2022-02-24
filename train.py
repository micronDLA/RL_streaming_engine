import time
import torch
import random
import argparse
from collections import deque

import os
import dgl
from coolname import generate_slug
import numpy as np
import networkx as nx
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from net import PolicyNet
from env import StreamingEngineEnv
from ppo_discrete import PPO
from graph_def import PREDEF_GRAPHS
from util import calc_score, initial_fill, get_graph_json, create_graph
from preproc import PreInput
#torch.autograd.set_detect_anomaly(True)
# random.seed(10)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TF messages

def get_args():
    parser = argparse.ArgumentParser(description='Streaming Engine RL Mapper')
    arg = parser.add_argument

    arg('--device_topology', nargs='+', type=int, default=(4, 1, 3), help='Device topology of Streaming Engine')
    arg('--epochs', type=int, default=50000, help='number of epochs')
    arg('--nodes', type=int, default=20,  help='number of nodes')
    arg('--debug', dest='debug', action='store_true', default=False, help='enable debug mode')
    arg('--input', type=str, default='input_graphs/vector_add_bashartest.json', help='load input json from file')

    # Constraints
    arg('--pass-timing', action='store_true', help='enable pass through timing')
    arg('--no-tm-constr', action='store_true', help='disable tile memory constraint')
    arg('--no-sf-constr', action='store_true', help='disable sync flow constraint')
    arg('--no-device-cross-connections', action='store_true', help='disable sync flow constraint')

    # PPO
    arg('--ppo-epoch', type=int, default=4)
    arg('--max-grad-norm', type=float, default=1)
    arg('--graph_size', type=int, default=128, help='graph embedding size')
    arg('--emb_size', type=int, default=128, help='embedding size')
    arg('--update_timestep', type=int, default=500, help='update policy every n timesteps')
    arg('--K_epochs', type=int, default=100, help='update policy for K epochs')
    arg('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
    arg('--gamma', type=float, default=0.99, help='discount factor')
    arg('--lr', type=float, default=1e-3, help='parameters for Adam optimizer')
    arg('--betas', type=float, default=(0.9, 0.999), help='')
    arg('--loss_entropy_c', type=float, default=0.01, help='coefficient for entropy term in loss')
    arg('--loss_value_c', type=float, default=0.5, help='coefficient for value term in loss')
    arg('--model', type=str, default='', help='load saved model from file')
    arg('--log_interval', type=int, default=100, help='interval for logging data')
    arg('--graph_feat_size', type=int, default=32, help='graph_feat_size')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()  # Holds all the input arguments
    args.device_topology = tuple(args.device_topology)
    print('[ARGS]')
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    writer = SummaryWriter(comment=f'_{generate_slug(2)}')
    print(f'[INFO] Saving log data to {writer.log_dir}')
    writer.add_text('experiment config', str(args))
    writer.flush()

    if args.input:
        graphdef = get_graph_json(args.input)
    else:
        graphdef = PREDEF_GRAPHS["FFT"]
    graphdef = create_graph(graphdef)
    args.nodes = nodes = graphdef['graph'].number_of_nodes()

    #graphdef:
    # graphdef['graph']: dgl graph, ndata['tm_req'], ndata['feat']
    # graphdef['graphdef']: (edge_src, edge_dst, extra_node)
    # graphdef['tile_memory_req']: dict{ tile_mem variable str : int index }
    # graphdef['tile_memory_map']: dict{ node id : list of tile mem var indexes }
    #tensor_in:
    # tensor_in['state'] = torch.FloatTensor(state).view(-1).unsqueeze(1)
    # tensor_in['node_sel'] = node_1hot.unsqueeze(1)
    #device:
    # device['topology'] = args.device_topology
    # device['action_dim'] = np.prod(args.device_topology)
    #constr:
    # constr['grp_nodes'] = tile mem based grp_nodes

    if args.debug:
        graph_in = graphdef['graph'].adjacency_matrix_scipy().toarray()
        print('graph adjacency matrix: ', graph_in)
        nx_g = graphdef['graph'].to_networkx()
        nx.draw(nx_g, nx.nx_agraph.graphviz_layout(nx_g, prog='dot'), with_labels=True)
        plt.show()

    device = {}
    device['topology'] = args.device_topology
    device['action_dim'] = np.prod(args.device_topology)
    tensor_in = {}

    preproc = PreInput(args)
    graphdef = preproc.pre_graph(graphdef, device)

    env = StreamingEngineEnv(args=args, graphdef=graphdef, device=device)

    ppo = PPO(args,
              graphdef = graphdef,
              device = device,
              state_dim = args.nodes*2)

    # logging variables
    reward = best_reward = 0
    reward_buf = deque(maxlen=100)
    reward_buf.append(0)
    time_step = 0
    start = time.time()

    # RL place each node
    # training loop:
    print('Starting PPO training...')
    for i_episode in range(1, args.epochs + 1):
        env.reset()
        state = -torch.ones(args.nodes) * 2 #ready time: -2 not placed
        action = -torch.ones(args.nodes, 3)
        time_step += 1 #number of epoch to train model
        # pre place constraint
        constr, action = preproc.pre_constr(action, graphdef, device)

        for node_id in range(0, args.nodes):
            if (action[node_id] > -1).all() : #skip pre placed nodes
                continue
            node_1hot = torch.zeros(args.nodes)
            node_1hot[node_id] = 1.0
            tensor_in['state'] = torch.FloatTensor(state).view(-1).unsqueeze(1)
            tensor_in['node_sel'] = node_1hot.unsqueeze(1)
            rl_state = preproc.pre_input(tensor_in)

            assigment, tobuff = ppo.select_action(tensor_in = rl_state, graphdef = graphdef, node_id = node_id, action=action, pre_constr=constr) # node assigment index in streaming eng slice
            action = ppo.get_coord(assigment, action, node_id) # put node assigment to vector of node assigments, 2D tensor
            reward, state, isvalid = env.step(action)

            # Saving reward and is_terminals:
            done = node_id == (args.nodes - 1)
            ppo.add_buffer(tobuff, reward, done)
            best_reward = max(best_reward, state.max().item())
            reward_buf.append(reward.mean())

        # learning:
        if time_step % args.update_timestep == 0:
            ppo.update()
            time_step = 0


        # logging
        if i_episode % args.log_interval == 0:
            writer.add_scalar('mean reward/episode', np.mean(reward_buf), i_episode)
            writer.add_scalar('total time/episode', best_reward, i_episode)
            writer.flush()
            end = time.time()
            print(f'\rEpisode: {i_episode} | Ready time: {best_reward} | Mean Reward: {np.mean(reward_buf):.2f} | Time elpased: {end - start:.2f}s', end='')
            # writer.add_scalar('avg improvement/episode', avg_improve, i_episode)
            # print('Episode {} \t Avg improvement: {}'.format(i_episode, avg_improve))
            torch.save(ppo.policy.state_dict(), 'model_epoch.pth')
            running_reward = 0

