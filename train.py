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

#torch.autograd.set_detect_anomaly(True)
# random.seed(10)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TF messages

def get_args():
    parser = argparse.ArgumentParser(description='Streaming Engine RL Mapper')
    arg = parser.add_argument
    arg('--mode', type=int, default=2, help='0 - random search, 1 - CMA-ES search, 2 - RL PPO, 3 - sinkhorn, 4 - multigraph, 5 - transformer')

    arg('--device_topology', nargs='+', type=int, default=(4, 1, 3), help='Device topology of Streaming Engine')
    arg('--epochs', type=int, default=5000, help='number of epochs')
    arg('--nodes', type=int, default=20,  help='number of nodes')
    arg('--debug', dest='debug', action='store_true', default=False, help='enable debug mode')
    arg('--input', type=str, default='input_graphs/vectorAdd_ir.json', help='load input json from file')

    # Constraints
    arg('--pass-timing', action='store_true', help='enable pass through timing')
    arg('--no-tm-constr', action='store_true', help='disable tile memory constraint')
    arg('--no-sf-constr', action='store_true', help='disable sync flow constraint')

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
    graph = create_graph(graphdef)
    args.nodes = nodes = graph.number_of_nodes()

    if args.debug:
        graph_in = graph.adjacency_matrix_scipy().toarray()
        print('graph adjacency matrix: ', graph_in)
        nx_g = graph.to_networkx()
        nx.draw(nx_g, nx.nx_agraph.graphviz_layout(nx_g, prog='dot'), with_labels=True)
        plt.show()

    # random search
    if args.mode == 0:
        # randomly occupy with nodes (not occupied=0 value):
        device_topology = args.device_topology
        # device_topology = (args.grid_size, args.grid_size, args.spokes)
        grid, grid_in, place = initial_fill(nodes, device_topology)

        env = StreamingEngineEnv(args=args,
                                 graphs=[graph],
                                 graphdef=graphdef,
                                 device_topology=device_topology,
                                 device_cross_connections=True,
                                 device_feat_size=48,
                                 graph_feat_size=32)

        # testing grid placement scoring:
        _, ready_time, valid = env._calculate_reward(torch.tensor(grid_in))

        # random search
        before_rs = ready_time.max().item() if valid else float('inf')
        best_grid = grid_in.copy()

        print('Running Random search optimization ...')
        for i in tqdm(range(args.epochs)):
            grid, grid_in, _ = initial_fill(nodes, grid.shape)
            _, ready_time, valid = env._calculate_reward(torch.tensor(grid_in))
            after_rs = ready_time.max().item()
            if before_rs > after_rs and valid:
                before_rs = after_rs
                best_grid = grid_in.copy()
                if args.debug:
                    print('best_grid score so far: ', after_rs)

        print('best score found: ', before_rs)
        if args.debug:
            print('optim placement: ', best_grid)

    # ES search
    elif args.mode == 1:
        # randomly occupy with nodes (not occupied=0 value):
        device_topology = args.device_topology
        # device_topology = (args.grid_size, args.grid_size, args.spokes)
        grid, grid_in, place = initial_fill(nodes, device_topology)

        env = StreamingEngineEnv(args=args,
                                 graphs=[graph],
                                 graphdef=graphdef,
                                 device_topology=device_topology,
                                 device_cross_connections=True,
                                 device_feat_size=48,
                                 graph_feat_size=32)

        # testing grid placement scoring:
        _, ready_time, valid = env._calculate_reward(torch.tensor(grid_in))
        final_es = ready_time.max().item() if valid else float('inf')
        final_value = grid_in if valid else None

        import nevergrad as ng

        budget = args.epochs  # How many steps of training we will do before concluding.
        workers = 16
        # param = ng.p.Array(shape=(int(nodes), 1)).set_integer_casting().set_bounds(lower=0, upper=ROW*COL*nodes)
        param = ng.p.Array(init=place).set_integer_casting().set_bounds(lower=0, upper= np.prod(device_topology))
        # ES optim
        names = "CMA"
        optim = ng.optimizers.registry[names](parametrization=param, budget=budget, num_workers=workers)
        # optim = ng.optimizers.RandomSearch(parametrization=param, budget=budget, num_workers=workers)
        # optim = ng.optimizers.NGOpt(parametrization=param, budget=budget, num_workers=workers)

        print('Running ES optimization ...')
        for _ in tqdm(range(budget)):
            x = optim.ask()
            grid, grid_in, _ = initial_fill(nodes, device_topology, manual=x.value)
            _, ready_time, valid = env._calculate_reward(torch.tensor(grid_in))
            loss = ready_time.max().item() if valid else float('inf')
            optim.tell(x, loss)
            if final_es > loss:
                final_value = grid_in
                final_es = loss

        rec = optim.recommend()
        grid, grid_in, _ = initial_fill(nodes, device_topology, manual=rec.value)
        _, ready_time, valid = env._calculate_reward(torch.tensor(grid_in))

        print('best score found:', final_es)
        if args.debug:
            print('optim placement:\n', final_value)

    # PPO Feedforward FF
    elif args.mode == 2:
        device_topology = args.device_topology
        action_dim = np.prod(args.device_topology)
        # RL place each node
        env = StreamingEngineEnv(args=args,
                                 graphs=[graph],
                                 graphdef=graphdef,
                                 device_topology=device_topology,
                                 device_cross_connections=True,
                                 device_feat_size=action_dim,
                                 graph_feat_size=32,
                                 placement_mode='one_node',
                                 )
        ppo = PPO(args, state_dim=args.nodes*2, action_dim=action_dim, gnn_in=env.compute_graph.ndata['feat'].shape[1])

        # logging variables
        reward = best_reward = 0
        reward_buf = deque(maxlen=100)
        reward_buf.append(0)
        time_step = 0
        start = time.time()
        gprod = np.prod(device_topology[:2])
        # training loop:
        print('Starting PPO training...')
        for i_episode in range(1, args.epochs + 1):
            env.reset()
            gr_edges = torch.stack(env.compute_graph.edges()).unsqueeze(0).float()  # [[src_nodes], [dst_nodes]] 
            state = -torch.ones(args.nodes)*2 #ready time: -2 not placed
            action = -torch.ones(args.nodes, 3)
            time_step += 1 #number of epoch to train model

            if not args.no_sf_constr:
                not_used = [ii for ii in range(gprod)]
                for node_id in range(0, args.nodes):
                    if len(env.compute_graph.predecessors(node_id)) == 0:
                        place = random.choice(not_used)
                        not_used.remove(place)
                        x, y = np.unravel_index(place, device_topology[:2])
                        action[node_id] = torch.Tensor([x, y, 0])

            for node_id in range(0, args.nodes):
                if not args.no_sf_constr and len(env.compute_graph.predecessors(node_id)) == 0:
                    continue
                node_1hot = torch.zeros(args.nodes)
                node_1hot[node_id] = 1.0
                rl_state = torch.cat((torch.FloatTensor(state).view(-1), node_1hot))  # grid, node to place
                assigment, tobuff = ppo.select_action(rl_state, graph, node_id) # node assigment index in streaming eng slice
                action = ppo.get_coord(assigment, action, node_id, device_topology) # put node assigment to vector of node assigments, 2D tensor
                reward, state, _ = env.step(action)

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

    # Sinkhorn
    elif args.mode == 3:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_topology = args.device_topology
        action_dim = np.prod(args.device_topology)
        grid, grid_in, place = initial_fill(nodes, device_topology)

        # initialize Environment, Network and Optimizer
        env = StreamingEngineEnv(args=args,
                                 graphs=[graph],
                                 graphdef=graphdef,
                                 device_topology=device_topology,
                                 device_cross_connections=True,
                                 device_feat_size=action_dim,
                                 graph_feat_size=32,
                                 init_place=None, # torch.tensor(grid_in),
                                 emb_mode='topological',
                                 placement_mode='all_node')
        policy = PolicyNet(cg_in_feats=action_dim,
                           cg_hidden_dim=64,
                           cg_conv_k=1,
                           transformer_dim=action_dim,
                           transformer_nhead=4,
                           transformer_ffdim=128,
                           transformer_dropout=0.1,
                           transformer_num_layers=4,
                           sinkhorn_iters=100)
        optim = Adam(policy.parameters(), lr=args.lr)
        if args.model != '':
            policy.load_state_dict(torch.load(args.pre_train)['model_state_dict'])
            optim.load_state_dict(torch.load(args.pre_train)['optimizer_state_dict'])

        # to keep track of average reward
        reward_buf = deque(maxlen=100)
        reward_buf.append(0)

        # train
        for episode in range(args.epochs):

            # reset env
            state = env.reset()

            # collect 'trajectory'
            # only one trajectory step
            with torch.no_grad():
                old_action, old_logp, old_entropy, old_scores = policy(*state)
                old_reward, _, _ = env.step(old_action)

            # use 'trajectory' to train network
            for epoch in range(args.ppo_epoch):

                action, logp, entropy, scores = policy(*state)
                reward, ready_time, _ = env.step(action)

                ratio = torch.exp(logp) / (torch.exp(old_logp) + 1e-8)
                surr1 = ratio * reward
                surr2 = torch.clamp(ratio, 1-args.eps_clip, 1+args.eps_clip) * reward
                action_loss = -torch.fmin(surr1, surr2)
                entropy_loss = entropy * args.loss_entropy_c

                loss = (action_loss + entropy)
                loss = loss.mean()

                optim.zero_grad()
                loss.backward()
                clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optim.step()

                reward_buf.append(reward.mean())
            if episode % args.log_interval == 0:
                # TODO: Add number of nodes places to this prompt and also log it
                print(f'Episode: {episode} | Epoch: {epoch} | Ready time: {ready_time.max().item()} | Loss: {loss.item()} | Mean Reward: {np.mean(reward_buf)}')
                # TODO: Is mean of reward buffer the total mean up until now?
                writer.add_scalar('mean reward/episode', np.mean(reward_buf), episode)
                writer.add_scalar('total time/episode', ready_time.max().item(), episode)
                writer.flush()

            if episode % args.log_interval == 0 and args.debug:
                plt.imshow(old_scores[0].exp().detach(), vmin=0, vmax=1)
                plt.pause(1e-6)
                #plt.show()
                #env.render()

            if episode % 1000 == 0:
                torch.save({
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'mean_reward': np.mean(reward_buf)
                }, 'models/model'+ str(episode) +'.pth')
    
    # PPO multiple graphs
    elif args.mode == 4:
        device_topology = args.device_topology
        action_dim = np.prod(args.device_topology)
        # different graphs
        g_defs = [([1, 1, 2, 2, 3, 4, 6, 6, 8, 10, 11, 11], [2, 3, 4, 6, 6, 5, 7, 8, 9, 11, 12, 13], 0),
                  ([1, 1, 2, 3, 4, 6, 6, 8, 10, 11, 11], [2, 3, 4, 6, 5, 7, 8, 9, 11, 12, 13], 0),
                  ([1, 1, 2, 2, 3, 4, 6, 6, 8, 10, 11, 12], [2, 3, 4, 6, 6, 5, 7, 8, 9, 11, 12, 13], 0),
                  ([1, 1, 2, 2, 3, 4, 4, 6, 6, 8, 10, 11, 11], [2, 3, 4, 6, 6, 5, 7, 7, 8, 9, 11, 12, 13], 0)]
        graphs = []
        for gdef in g_defs:
            g = create_graph(gdef)
            graphs.append(g)

        # RL place each node
        env = StreamingEngineEnv(args=args,
                                 graphs=graphs,
                                 graphdef=graphdef,
                                 device_topology=device_topology,
                                 device_cross_connections=True,
                                 device_feat_size=action_dim,
                                 graph_feat_size=32,
                                 placement_mode='one_node')
        ppo = PPO(args, state_dim=args.nodes*2, action_dim=action_dim,)
                 # mode='super', ntasks = len(graphs))

        # logging variables
        reward = best_reward = 0
        reward_buf = deque(maxlen=100)
        reward_buf.append(0)
        time_step = 0
        start = time.time()
        # training loop:

        for taskid in range(len(graphs)):
            print(f'Starting PPO Different Graphs training {taskid} ...')
            print('[WARNING] Multiple graphs with feed forward PPO is not currently supported. \n \
                    Check `evaluate()` functio \n \
                    Remove this warning if issue has been patched')
            for i_episode in range(1, args.epochs + 1):
                env.reset()
                env.get_graph(taskid) #set compute_graph to a graph from collection of graphs
                gr_edges = torch.stack(env.compute_graph.edges()).unsqueeze(0).float()
                state = -torch.ones(args.nodes)*2 #ready time: -2 not placed
                action = -torch.ones(args.nodes, 3)
                time_step += 1 #number of epoch to train model

                node_id = 0
                while node_id < args.nodes:
                    node_1hot = torch.zeros(args.nodes)
                    node_1hot[node_id] = 1.0
                    rl_state = torch.cat((torch.FloatTensor(state).view(-1), node_1hot))  # grid, node to place
                    assigment, tobuff = ppo.select_action(rl_state, gr_edges)#, taskid=taskid) # node assigment index
                    action = ppo.get_coord(assigment, action, node_id, device_topology) # put node assigment to vector of node assigments
                    reward, state, _ = env._calculate_reward(action)

                    if not torch.any(state < 0):
                        # Saving reward and is_terminals:
                        done = node_id == (args.nodes - 1)
                        ppo.add_buffer(tobuff, reward, done)
                        best_reward = max(best_reward, state.max().item())
                        reward_buf.append(reward.mean())
                        node_id += 1

                # learning:
                if time_step % args.update_timestep == 0:
                    ppo.update(taskid=taskid)
                    time_step = 0

                # logging
                if i_episode % args.log_interval == 0:
                    i_ep = i_episode + taskid * args.epochs
                    print(f'Episode: {i_ep} | Ready time: {best_reward} | Mean Reward: {np.mean(reward_buf)}')
                    writer.add_scalar('mean reward/episode', np.mean(reward_buf), i_ep)
                    writer.add_scalar('total time/episode', best_reward, i_ep)
                    writer.flush()
                    end = time.time()
                    print('Execution time {} s'.format(end - start))
                    # torch.save(ppo.policy.state_dict(), 'model_epoch.pth')

    # PPO Sequence Transformer
    elif args.mode == 5:
        device_topology = args.device_topology
        action_dim = np.prod(args.device_topology)
        # RL place each node
        env = StreamingEngineEnv(args=args,
                                 graphs=[graph],
                                 graphdef=graphdef,
                                 device_topology=device_topology,
                                 device_cross_connections=True,
                                 device_feat_size=action_dim,
                                 graph_feat_size=32,
                                 placement_mode='one_node')
        ppo = PPO(args, state_dim=args.nodes, action_dim=action_dim, mode='transformer')

        # logging variables
        reward = best_reward = 0
        reward_buf = deque(maxlen=100)
        reward_buf.append(0)
        time_step = 0
        start = time.time()
        # training loop:
        print('Starting PPO Sequence training...')
        for i_episode in range(1, args.epochs + 1):
            env.reset()
            gr_edges = torch.stack(env.compute_graph.edges()).unsqueeze(0).float()
            time_step += 1 #number of epoch to train model
            state = -torch.ones(args.nodes) * 2  # ready time: -2 not placed
            action = -torch.ones(args.nodes, 3)
            state_in = torch.zeros(args.nodes, 1, action_dim)
            for node in range(0, args.nodes):
                rl_state = (state_in, torch.tensor(node)) # grid, node to place
                assigment = ppo.select_action(rl_state, gr_edges) # node assigment index
                action = ppo.get_coord(assigment, action, node, device_topology) # put node assigment to vector of node assigments
                reward, state, _ = env._calculate_reward(action)
                state_in[node, :, assigment] = state[node]
                # Saving reward and is_terminals:
                ppo.buffer.rewards.append(reward.mean())
                if node == (args.nodes - 1):
                    done = True
                else:
                    done = False
                ppo.buffer.is_terminals.append(done)
                best_reward = max(best_reward, state.max().item())
                reward_buf.append(reward.mean())
            # learning:
            if time_step % args.update_timestep == 0:
                ppo.update()
                time_step = 0


            # logging
            if i_episode % args.log_interval == 0:
                print(f'Episode: {i_episode} | Ready time: {best_reward} | Mean Reward: {np.mean(reward_buf)}')
                writer.add_scalar('mean reward/episode', np.mean(reward_buf), i_episode)
                writer.add_scalar('total time/episode', best_reward, i_episode)
                writer.flush()
                end = time.time()
                print('Execution time {} s'.format(end - start))
                # writer.add_scalar('avg improvement/episode', avg_improve, i_episode)
                # print('Episode {} \t Avg improvement: {}'.format(i_episode, avg_improve))
                torch.save(ppo.policy.state_dict(), 'model_epoch.pth')
                running_reward = 0

