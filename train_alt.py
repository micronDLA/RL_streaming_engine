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
import sa
import random

from envs.streaming_engine_env import StreamingEngineEnv
from ppo_discrete import PPO

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

# get a node placement given mask
def get_masked_rand(mask, device):
    l = []
    for i in range(device['action_dim']):
        if mask[i] == 1:
            l.append(i)
    tile_idx = random.choice(l)
    return tile_idx

# generate node placement random in sequence
def get_nodes_rand(init_nodes, args, env, graphdef, device, reward_buf):

    init_nodeid, place_nodes = [], []
    readytime = 100
    #init_nodes: nodes already placed (node_id, tile_slice_idx)
    for s in init_nodes:
        tile, spoke = np.unravel_index(s[1], args.device_topology)
        action = [s[0], tile, spoke]
        state, reward, done, mdata = env.step(action)
        readytime = mdata['ready_time']
        place_nodes.append((s[0], s[1]))
        init_nodeid.append(s[0])

    asc = dgl.topological_nodes_generator(graphdef['graph'])
    lnodes = [i.item() for t in asc for i in t]
    for node_id in lnodes:
        if node_id in init_nodeid:
            continue
        mask = env.get_mask(node_id)
        if np.all(mask == 0):
            return 100, []
        tile_slice_idx = get_masked_rand(mask, device)
        tile, spoke = np.unravel_index(tile_slice_idx, args.device_topology)
        action = [node_id, tile, spoke]
        state, reward, done, mdata = env.step(action)
        readytime = mdata['ready_time']
        reward_buf.append(reward)
        place_nodes.append((node_id, tile_slice_idx))

    return readytime, place_nodes

def run_sa_mapper(args, graphs, writer=None):
    # Parse arguments
    args.device_topology = tuple(args.device_topology)
    logging.info('[ARGS]')
    logging.info('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    print('Simulate Annealing SE Mapper ', args)
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

    best_ready_time = float('inf')
    best_reward = 0
    opt = sa.minimize(args, env, graphdef, device, cooling_schedule='linear', step_max=1000000000000000000, t_max=1, t_min=0)
    opt.results()
    return best_ready_time, best_reward


def run_mapper_es(args, graphdef):

    # Parse arguments
    args.device_topology = tuple(args.device_topology)
    logging.info('[ARGS]')
    logging.info('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    # Tensorboard logging
    if not args.quiet:
        writer = SummaryWriter(comment=f'_{generate_slug(2)}')
        print(f'[INFO] Saving log data to {writer.log_dir}')
        writer.add_text('experiment config', str(args))
        writer.flush()

    args.nodes = graphdef['graph'].number_of_nodes()

    if args.debug:
        print_graph(graphdef)

    # SE Device attributes
    device = {}
    device['topology'] = args.device_topology
    device['action_dim'] = np.prod(args.device_topology)

    preproc = PreInput(args)
    graphdef = preproc.pre_graph(graphdef, device)

    # Init gym env
    env = StreamingEngineEnv(args,
                             graphdef=graphdef,
                             tile_count=args.device_topology[0],
                             spoke_count=args.device_topology[1],
                             pipeline_depth=args.pipeline_depth)

    # randomly occupy with nodes (not occupied=0 value):
    device_topology = args.device_topology
    # Setup logging variables
    best_ready_time = 100
    best_reward = 0
    final_value = None

    import nevergrad as ng

    budget = args.epochs  # How many steps of training we will do before concluding.
    workers = 16
    # param = ng.p.Array(shape=(int(nodes), 1)).set_integer_casting().set_bounds(lower=0, upper=ROW*COL*nodes)
    param = ng.p.Array(shape=(int(args.nodes), 1)).set_integer_casting().set_bounds(lower=0, upper=np.prod(device_topology)-1)
    # ES optim
    names = "CMA"
    optim = ng.optimizers.registry[names](parametrization=param, budget=budget, num_workers=workers)
    # optim = ng.optimizers.RandomSearch(parametrization=param, budget=budget, num_workers=workers)
    # optim = ng.optimizers.NGOpt(parametrization=param, budget=budget, num_workers=workers)
    def es_calculate_reward(actions):
        ready_buf = []
        reward_buf = []
        xv = [i[0] for i in actions]
        for node_id, act in enumerate(xv):
            tile, spoke = np.unravel_index(act, args.device_topology)
            action = [node_id, tile, spoke]
            state, reward, done, mdata = env.step(action)
            ready_buf.append(mdata['ready_time'])
            reward_buf.append(reward)
            if done and reward < 0:
                return 100, -10
        return np.max(ready_buf), np.mean(reward_buf)

    def isvalid(x):
        xv = [i[0] for i in x]
        return len(set(xv)) == len(xv)

    print('Running ES optimization ...')
    for _ in tqdm(range(budget)):
        x = optim.ask()
        while isvalid(x.value):
            x = optim.ask()
        loss, reward = es_calculate_reward(x.value)
        optim.tell(x, loss)
        if best_ready_time > loss:
            final_value = x.value
            best_ready_time = loss
            best_reward = reward

    rec = optim.recommend()
    es_calculate_reward(rec.value)
    print('best score found:', best_ready_time)
    if args.debug:
        print('optim placement:\n', final_value)

    return best_ready_time, best_reward

if __name__ == "__main__":
    args = get_args()  # Holds all the input arguments
    graph_json = get_graph_json(args.input)# Get computation graph definition
    graphdef = create_graph(graph_json)
    run_sa_mapper(args, graphdef)
