import argparse
import logging

from util import get_graph_json, create_graph
from preproc import PreInput
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from envs.streaming_engine_env import StreamingEngineEnv

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
    # Parse arguments
    args = get_args()  # Holds all the input arguments
    args.device_topology = tuple(args.device_topology)
    logging.info('[ARGS]')
    logging.info('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

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

    device = {}
    device['topology'] = args.device_topology
    device['action_dim'] = np.prod(args.device_topology)
    tensor_in = {}

    preproc = PreInput(args)
    graphdef = preproc.pre_graph(graphdef, device)

    env = StreamingEngineEnv(graphdef=graphdef,
                             tile_count=args.device_topology[0], 
                             spoke_count=args.device_topology[1], 
                             pipeline_depth=args.pipeline_depth)

    """
    # Start training loop
    time_step = 0
    for i_episode in range(1, args.epochs + 1):
        state = env.reset
        time_step += 1
        
        # Iterate over nodes to place
        for node_id in range(0, args.nodes):
            mask = env.get_mask(node_id)
            action = ppo.get_action(node_id, state, mask)
            state, reward, done, _ = env.step(action)

            # Save things to buffer

            # learning:
            if time_step % args.update_timestep == 0:
                ppo.update()
                time_step = 0
    """
