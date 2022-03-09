import time
import argparse
import logging
from collections import deque

import gym
import torch
from util import get_graph_json, create_graph
from preproc import PreInput
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from coolname import generate_slug
from torch.utils.tensorboard import SummaryWriter

from envs.streaming_engine_env import StreamingEngineEnv
from ppo_discrete import PPO

def get_args():
    parser = argparse.ArgumentParser(description='Streaming Engine RL Mapper')
    arg = parser.add_argument

    arg('--device-topology', nargs='+', type=int, default=(4, 3), help='Device topology of Streaming Engine')
    arg('--pipeline-depth', type=int, default=3, help='processing pipeline depth')
    arg('--epochs', type=int, default=50000, help='number of epochs')
    arg('--nodes', type=int, default=20,  help='number of nodes')
    arg('--debug', dest='debug', action='store_true', default=False, help='enable debug mode')
    arg('--input', type=str, default='input_graphs/vectorAdd_ir.json', help='load input json from file')

    # Constraints
    arg('--pass-timing', action='store_true', help='enable pass through timing')
    arg('--no-tm-constr', action='store_true', help='disable tile memory constraint')
    arg('--no-sf-constr', action='store_true', help='disable sync flow constraint')
    arg('--no-device-cross-connections', action='store_true', help='disable sync flow constraint')

    # PPO
    arg('--graph_feat_size', type=int, default=128, help='graph_feat_size')
    arg('--emb_size', type=int, default=64, help='embedding size')
    arg('--update_timestep', type=int, default=500, help='update policy every n episodes')
    arg('--K_epochs', type=int, default=4, help='update policy for K epochs')
    arg('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
    arg('--gamma', type=float, default=0.99, help='discount factor')
    arg('--lr', type=float, default=1e-3, help='parameters for Adam optimizer')
    arg('--betas', type=float, default=(0.9, 0.999), help='')
    arg('--loss_entropy_c', type=float, default=0.01, help='coefficient for entropy term in loss')
    arg('--loss_value_c', type=float, default=0.5, help='coefficient for value term in loss')
    arg('--model', type=str, default='', help='load saved model from file')
    arg('--log_interval', type=int, default=100, help='interval for logging data')
    

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse arguments
    args = get_args()  # Holds all the input arguments
    args.device_topology = tuple(args.device_topology)
    logging.info('[ARGS]')
    logging.info('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    # Tensorboard logging
    writer = SummaryWriter(comment=f'_{generate_slug(2)}')
    print(f'[INFO] Saving log data to {writer.log_dir}')
    writer.add_text('experiment config', str(args))
    writer.flush()

    # Get computation graph definition
    graph_json = get_graph_json(args.input)
    graphdef = create_graph(graph_json)

    # graphdef['graph'] is a dgl graph
    args.nodes = nodes = graphdef['graph'].number_of_nodes()

    if args.debug:
        graph_in = graphdef['graph'].adjacency_matrix_scipy().toarray()
        print('graph adjacency matrix: ', graph_in)
        nx_g = graphdef['graph'].to_networkx()
        nx.draw(nx_g, nx.nx_agraph.graphviz_layout(nx_g, prog='dot'), with_labels=True)
        plt.show()

    # SE Device attributes
    device = {}
    device['topology'] = args.device_topology
    device['action_dim'] = np.prod(args.device_topology)
    tensor_in = {}

    preproc = PreInput(args)
    graphdef = preproc.pre_graph(graphdef, device)

    # Init gym env
    env = StreamingEngineEnv(args,
                             graphdef = graphdef,
                             tile_count = args.device_topology[0], 
                             spoke_count = args.device_topology[1], 
                             pipeline_depth = args.pipeline_depth)

    # Init ppo
    ppo = PPO(args,
             graphdef = graphdef,
             device = device,
             state_dim = env.observation_space.n,  # Will change later to include node to be placed
             mode='simple_ff')

    # Setup logging variables
    reward_buf = deque(maxlen=100)
    reward_buf.append(0)
    start = time.time()
    time_step = 0

    # Start training loop
    for i_episode in range(1, args.epochs + 1):
        state = env.reset()
        time_step += 1
        total_reward = 0
        done = False
        
        # Iterate over nodes to place
        for node_id in range(0, args.nodes):
            mask = env.get_mask(node_id)
            tile_slice_idx, tobuff = ppo.select_action(state, graphdef, node_id, mask)
            tile, spoke = np.unravel_index(tile_slice_idx, args.device_topology)
            action = [node_id, tile, spoke]
            state, reward, done, _ = env.step(action)

            total_reward += reward
            if node_id == args.nodes - 1:
                done = True

            # Save things to buffer
            ppo.add_buffer(tobuff, reward, done)
            reward_buf.append(reward)

        if len(env.placed_nodes) == args.nodes:
            print(f'Episode {i_episode}: {env.placed_nodes}')
            
        # learning:
        if i_episode % args.update_timestep == 0:
            ppo.update()

        # logging
        if i_episode % args.log_interval == 0:
            writer.add_scalar('Mean reward/episode', np.mean(reward_buf), i_episode)
            writer.add_scalar('No. of nodes placed', len(env.placed_nodes), i_episode)
            writer.flush()
            end = time.time()
            print(f'\rEpisode: {i_episode} | Total reward: {total_reward} | Mean Reward: {np.mean(reward_buf):.2f} | Nodes placed: {len(env.placed_nodes)} | Time elpased: {end - start:.2f}s', end='')
            # writer.add_scalar('avg improvement/episode', avg_improve, i_episode)
            # print('Episode {} \t Avg improvement: {}'.format(i_episode, avg_improve))
            torch.save(ppo.policy.state_dict(), 'model_epoch.pth')
            running_reward = 0
        
