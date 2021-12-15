import torch
import argparse
import numpy as np
import networkx as nx
from collections import deque
from matplotlib import pyplot as plt
import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import dgl
from net import PolicyNet
from env import StreamingEngineEnv
from tqdm import tqdm
import random
from matplotlib import pyplot as plt
from util import calc_score, initial_fill, ROW, COL, fix_grid_bins
from torch.utils.tensorboard import SummaryWriter
import time
from ppo_discrete import PPO

#torch.autograd.set_detect_anomaly(True)
# random.seed(10)

def get_args():
    parser = argparse.ArgumentParser(description='grid placement')
    arg = parser.add_argument
    arg('--mode', type=int, default=2, help='0 random search, 1 CMA-ES search, 2- RL PPO, 3- sinkhorn')

    arg('--device_topology',   type=tuple, default=(16, 1, 3), help='number of PE')
    arg('--spokes',   type=int, default=3, help='Number of spokes')
    arg('--epochs',   type=int, default=5000, help='number of iterations')
    arg('--nodes', type=int, default=20,  help='number of nodes')
    arg('--debug', dest='debug', action='store_true', default=False, help='debug mode')

    # PPO
    arg('--num-episode', type=int, default=100000)
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
    arg('--model', type=str, default='', help='load saved model')
    arg('--log_interval', type=int, default=100, help='interval for log')

    args = parser.parse_args()
    return args

# Specify graph as ([src_ids], [dst_ids], extra isolated nodes) edges, node ordering
# starts from 0 regardless of specification
PREDEF_GRAPHS = {
    "DISTANCE": ([0, 1, 2, 2, 2, 3, 4, 4, 5, 5, 6, 7, 8, 8, 11, 12, 12, 13, 13, 14, 14, 14, 15, 16, 17, 18, 19, 19, 20, 20, 21, 22],
                 [1, 2, 3, 4, 5, 4, 5, 6, 6, 7, 7, 8, 9, 10, 12, 13, 19, 15, 14, 16, 17, 18, 19, 17, 18, 19, 20, 21, 21, 22, 22, 23]),
    "FFT_OLD":
             ([0,
               2, 3,
               5, 6, 6, 7, 8, 9, 10, 11, 12,
               14, 15, 16, 17, 18, 19, 20, 21, 17, 17,
               23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
               35, 35, 36, 37, 38, 38, 39, 39, 40, 42, 41, 43, 43, 48, 49, 50, 51,
               53,
               55,
              ],
              [1,
               3, 4,
               6, 7, 8, 9, 9, 10, 11, 12, 13,
               15, 16, 17, 18, 19, 20, 21, 22, 19, 21,
               24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
               36, 37, 38, 39, 40, 42, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
               54,
               56,
              ]),
    "FFT_SIMPLE": ([0, 0, 1, 1, 2, 3, 4, 4, 5], [1, 2, 3, 4, 4, 6, 5, 8, 7]),
    "FFT_SYNC2": ([1, 1, 2, 2, 3, 4, 6, 6, 8], [2, 3, 4, 6, 6, 5, 7, 8, 9]),
    # Complete FFT graph
            # ([src_nodes],[dst_nodes], extra nodes to add)
    "FFT": {'graphdef': ([1, 1, 2, 2, 3, 4, 6, 6, 8, 10, 11, 11], [2, 3, 4, 6, 6, 5, 7, 8, 9, 11, 12, 13], 3),
            # instr_ID: [required TM index]
            'tile_memory_req': {0: [4], 1: [0], 2: [5], 3: [9], 4: [7], 5: [3], 6: [8], 7: [1], 8: [6], 9: [1], 10: [12,14], 11: [12,14], 12: [13], 13: [13], 14: [2,10], 15: [2,11], 16: [0]}
           }
}

TILE_MEMORY_MAP = {
    0: None,
    1: 'TM_inDataA',
    2: 'TM_inDataB',
    3: 'TM_table',
    4: 'TM_half_pnts',
    5: 'TM_mask',
    6: 'TM_halfsize',
    7: 'TM_tablestep',
    8: 'TM_size',
    9: 'TM_I',
    10: 'TM_right',
    11: 'TM_left',
    12: 'TM_righti',
    13: 'TM_lefti',
    14: 'TM_tablei'
}

TM_IDX_TOTAL = max(TILE_MEMORY_MAP.keys())

def create_graph(graphdef, numnodes = 10):
    # random generate a directed acyclic graph
    if graphdef is None:
        a = nx.generators.directed.gn_graph(numnodes)
        graph = dgl.from_networkx(a)
    else:
        tile_memory_req = graphdef['tile_memory_req']
        edges = graphdef['graphdef']
        graph = dgl.graph((torch.Tensor(edges[0]).int(), torch.Tensor(edges[1]).int()))
        if len(edges) == 3:
            graph.add_nodes(edges[2])

        # Add tile memory constraints as features to graph
        tm_req_feat = torch.zeros(graph.num_nodes(), TM_IDX_TOTAL + 1)
        for instr_idx, tm_idxs in tile_memory_req.items():
            for tm_idx in tm_idxs:
                tm_req_feat[instr_idx][tm_idx] = 1
        
        graph.ndata['tm_req'] = tm_req_feat
    return graph

if __name__ == "__main__":
    args = get_args()  # Holds all the input arguments
    print('Arguments:', args)
    writer = SummaryWriter()

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
        device_topology = (16, 1, args.spokes)
        # device_topology = (args.grid_size, args.grid_size, args.spokes)
        grid, grid_in, place = initial_fill(nodes, device_topology)

        env = StreamingEngineEnv(graphs=[graph],
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
        for i in tqdm(range(args.num_episode)):
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
        device_topology = (16, 1, args.spokes)
        # device_topology = (args.grid_size, args.grid_size, args.spokes)
        grid, grid_in, place = initial_fill(nodes, device_topology)

        env = StreamingEngineEnv(graphs=[graph],
                                 device_topology=device_topology,
                                 device_cross_connections=True,
                                 device_feat_size=48,
                                 graph_feat_size=32)

        # testing grid placement scoring:
        _, ready_time, valid = env._calculate_reward(torch.tensor(grid_in))
        final_es = ready_time.max().item() if valid else float('inf')
        final_value = grid_in if valid else None

        import nevergrad as ng

        budget = args.num_episode  # How many steps of training we will do before concluding.
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
        device_topology = (16, 1, args.spokes)
        # RL place each node
        env = StreamingEngineEnv(graphs=[graph],
                                 graphdef=graphdef,
                                 tm_idx_total=TM_IDX_TOTAL,
                                 device_topology=device_topology,
                                 device_cross_connections=True,
                                 device_feat_size=48,
                                 graph_feat_size=32,
                                 placement_mode='one_node',
                                 )
        ppo = PPO(args, state_dim=args.nodes*2, action_dim=48)

        # logging variables
        reward = best_reward = 0
        reward_buf = deque(maxlen=100)
        reward_buf.append(0)
        time_step = 0
        start = time.time()
        # training loop:
        print('Starting PPO training...')
        for i_episode in range(1, args.epochs + 1):
            env.reset()
            gr_edges = torch.stack(env.compute_graph.edges()).unsqueeze(0).float()  # [[src_nodes], [dst_nodes]] 
            state = -torch.ones(args.nodes)*2 #ready time: -2 not placed
            action = -torch.ones(args.nodes, 3)
            time_step += 1 #number of epoch to train model
            for node in range(0, args.nodes):
                node_1hot = torch.zeros(args.nodes)
                node_1hot[node] = 1.0
                rl_state = torch.cat((torch.FloatTensor(state).view(-1), node_1hot))  # grid, node to place
                assigment = ppo.select_action(rl_state, graph) # node assigment index in streaming eng slice
                action = ppo.get_coord(assigment, action, node, device_topology) # put node assigment to vector of node assigments, 2D tensor
                reward, state, _ = env.step(action)
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
                print('Training time elpased: {:.2f} s'.format(end - start))
                # writer.add_scalar('avg improvement/episode', avg_improve, i_episode)
                # print('Episode {} \t Avg improvement: {}'.format(i_episode, avg_improve))
                torch.save(ppo.policy.state_dict(), 'model_epoch.pth')
                running_reward = 0

    # PPO multiple graphs
    elif args.mode == 4:
        device_topology = (16, 1, args.spokes)

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
        env = StreamingEngineEnv(graphs=graphs,
                                 device_topology=device_topology,
                                 device_cross_connections=True,
                                 device_feat_size=48,
                                 graph_feat_size=32,
                                 placement_mode='one_node')
        ppo = PPO(args, state_dim=args.nodes*2, action_dim=48,)
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
            for i_episode in range(1, args.epochs + 1):
                env.reset()
                env.get_graph(taskid) #set compute_graph to a graph from collection of graphs
                gr_edges = torch.stack(env.compute_graph.edges()).unsqueeze(0).float()
                state = -torch.ones(args.nodes)*2 #ready time: -2 not placed
                action = -torch.ones(args.nodes, 3)
                time_step += 1 #number of epoch to train model
                for node in range(0, args.nodes):
                    node_1hot = torch.zeros(args.nodes)
                    node_1hot[node] = 1.0
                    rl_state = torch.cat((torch.FloatTensor(state).view(-1), node_1hot))  # grid, node to place
                    assigment = ppo.select_action(rl_state, gr_edges)#, taskid=taskid) # node assigment index
                    action = ppo.get_coord(assigment, action, node, device_topology) # put node assigment to vector of node assigments
                    reward, state, _ = env._calculate_reward(action)
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
        device_topology = (16, 1, args.spokes)
        action_dim = 48
        # RL place each node
        env = StreamingEngineEnv(graphs=[graph],
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


    # sinkhorn
    elif args.mode == 3:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_topology = (16, 1, args.spokes)
        # device_topology = (args.grid_size, args.grid_size, args.spokes)
        grid, grid_in, place = initial_fill(nodes, device_topology)

        # initialize Environment, Network and Optimizer
        env = StreamingEngineEnv(graphs=[graph],
                                 graphdef=graphdef,
                                 tm_idx_total=TM_IDX_TOTAL,
                                 device_topology=device_topology, 
                                 device_cross_connections=True,
                                 device_feat_size=48,
                                 graph_feat_size=32,
                                 init_place=None, # torch.tensor(grid_in),
                                 emb_mode='topological',
                                 placement_mode='all_node')
        policy = PolicyNet(cg_in_feats=48,
                           cg_hidden_dim=64,
                           cg_conv_k=1,
                           transformer_dim=48,
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
        for episode in range(args.num_episode):

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

