import os
import time
import argparse
import logging
from collections import deque
import dgl
import torch
from util import get_graph_json, create_graph, output_json, print_graph
from preproc import PreInput
import numpy as np
from coolname import generate_slug
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random

from envs.streaming_engine_env import StreamingEngineEnv
from ppo_discrete import PPO

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def get_args():
    parser = argparse.ArgumentParser(description='Streaming Engine RL Mapper')
    arg = parser.add_argument

    arg('--device-topology', nargs='+', type=int, default=(16, 6), help='Device topology of Streaming Engine')
    arg('--pipeline-depth', type=int, default=3, help='processing pipeline depth')
    arg('--epochs', type=int, default=200000, help='number of epochs')
    arg('--nodes', type=int, default=20,  help='number of nodes')
    arg('--debug', dest='debug', action='store_true', default=False, help='enable debug mode')
    arg('--input', type=str, default='input_graphs/vectorAdd_ir.json', help='load input json from file')
    arg('--nnmode', type=str, default='ff_gnn_attention', help='select nn to use as actor/critic model: simple_ff, ff_gnn, ff_gnn_attention, ff_transf_attention')

    # Constraints
    arg('--pass-timing', action='store_true', help='enable pass through timing')
    arg('--no-sibling-constr', action='store_true', help='disable sibling nodes constraint')
    arg('--no-tm-constr', action='store_true', help='disable tile memory constraint')
    arg('--no-sf-constr', action='store_true', help='disable sync flow constraint')
    arg('--no-device-cross-connections', action='store_true', help='disable sync flow constraint')

    # PPO
    arg('--graph_feat_size', type=int, default=128, help='graph_feat_size')
    arg('--emb_size', type=int, default=64, help='embedding size')
    arg('--update_timestep', type=int, default=100, help='update policy every n episodes')
    arg('--K_epochs', type=int, default=5, help='update policy for K epochs')
    arg('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
    arg('--gamma', type=float, default=0.99, help='discount factor')
    arg('--lr', type=float, default=1e-3, help='parameters for Adam optimizer')
    arg('--betas', type=float, default=(0.9, 0.999), help='')
    arg('--loss_entropy_c', type=float, default=0.01, help='coefficient for entropy term in loss')
    arg('--loss_value_c', type=float, default=0.5, help='coefficient for value term in loss')
    arg('--model', type=str, default='', help='load saved model from file')
    arg('--log_interval', type=int, default=100, help='interval for logging data')
    arg('--quiet', action='store_true', help='dont save model and tensorboard')
    args = parser.parse_args()
    return args

def run_mapper(args, graphs, writer=None):
    # Parse arguments
    args.device_topology = tuple(args.device_topology)
    logging.info('[ARGS]')
    logging.info('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    print(args)

    # Tensorboard logging
    if writer is None:
        writer = SummaryWriter(comment=f'_{generate_slug(2)}')
        print(f'[INFO] Saving log data to {writer.log_dir}')
        writer.add_text('experiment config', str(args))
        writer.flush()

    if isinstance(graphs, list):
        graphdef = graphs[0]
    else:
        graphdef = graphs

    args.nodes = graphdef['graph'].number_of_nodes()

    if args.debug:
        print_graph(graphdef)

    # SE Device attributes
    device = {}
    device['topology'] = args.device_topology
    device['action_dim'] = np.prod(args.device_topology)

    preproc = PreInput(args)
    if isinstance(graphs, list):
        for i, gg in enumerate(graphs):
            graphs[i] = preproc.pre_graph(gg, device)
        graphdef = graphs[0]
    else:
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
    )

    # Setup logging variables
    reward_buf = deque(maxlen=100)
    reward_buf.append(0)
    start = time.time()
    time_step = 0
    best_ready_time = float('inf')
    best_reward = 0

    # Start training loop
    for i_episode in range(1, args.epochs + 1):
        if isinstance(graphs, list):
            graphdef = random.choice(graphs)
            env.set_graph(graphdef)

        state = env.reset()
        time_step += 1
        total_reward = 0
        done = False
        # Iterate over nodes to place in topological order
        asc = dgl.topological_nodes_generator(graphdef['graph'])
        lnodes = [i.item() for t in asc for i in t]
        for node_id in lnodes:
        # for node_id in range(args.nodes):
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

        if not args.quiet:
            writer.add_scalar('No. of nodes placed', len(env.placed_nodes), i_episode)

        if env.all_nodes_placed and env.graph_ready_time < best_ready_time:
            best_ready_time = env.graph_ready_time
            best_reward = np.mean(reward_buf)
            if not args.quiet:
                print(f'\nEpisode {i_episode}: {env.placed_nodes}')
                print(f'Best graph ready time yet: {best_ready_time}')
                # Save mapping json
                suffix = os.path.basename(args.input)
                output_json(env.placed_nodes,
                            no_of_tiles=args.device_topology[0],
                            spoke_count=args.device_topology[1],
                            out_file_name=f'mappings/mapping_{suffix}')
            
        # learning:
        if i_episode % args.update_timestep == 0:
            ppo.update()

        # logging
        if i_episode % args.log_interval == 0:
            end = time.time()
            print(f'\rEpisode: {i_episode} | best time {best_ready_time} | Total reward: {total_reward} | Mean Reward: {np.mean(reward_buf):.2f} | Nodes placed: {len(env.placed_nodes)} | Time elpased: {end - start:.2f}s', end='')
            if not args.quiet:
                writer.add_scalar('Mean reward/episode', np.mean(reward_buf), i_episode)
                writer.flush()
                torch.save(ppo.policy.state_dict(), 'model_epoch.pth')

    return best_ready_time, best_reward

if __name__ == "__main__":
    args = get_args()  # Holds all the input arguments
    graph_json = get_graph_json(args.input)# Get computation graph definition
    graphdef = create_graph(graph_json)
    run_mapper(args, graphdef)
