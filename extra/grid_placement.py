import dgl
import networkx as nx
from scipy.spatial import distance
import math
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import random
from matplotlib import pyplot as plt
from util import calc_score, initial_fill, GridEnv, Q_learn
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def get_args():
    parser = ArgumentParser(description='grid placement')
    arg = parser.add_argument
    arg('--mode', type=int, default=5, help='0 random search, 1 CMA-ES search, 2- run both to compare, 3- RL PPO, 4- DQN ')
    arg('--grid_size',   type=int, default=4, help='number of sqrt PE')
    arg('--grid_depth',   type=int, default=3, help='PE pipeline depth')
    arg('--epochs',   type=int, default=5000, help='number of iterations')
    arg('--nodes', type=int, default=20,  help='number of nodes')
    arg('--debug', dest='debug', action='store_true', default=False, help='debug mode')

    # PPO
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

    #Q-learn
    arg('--decay_gamma', type=float, default=0.9, help='q-learn decay gamma')
    arg('--q_lr', type=float, default=0.2, help='q-learn learn rate')
    arg('--exp_rate', type=float, default=0.3, help='q-learn exploration rate')

    #GraphNet - supervised
    arg('-d', '--dataset', type=str, default='data_graph_grid.pth', help='data to use')
    arg('--batch_size', type=int, default=64, help='batch size')

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_args()  # Holds all the input arguments
    print('Arguments:', args)

    # random generate a directed acyclic graph
    nodes = args.nodes
    a = nx.generators.directed.gn_graph(nodes)
    g = dgl.from_networkx(a)

    if args.debug:
        graph_in = g.adjacency_matrix_scipy().toarray()
        print('graph adjacency matrix: ', graph_in)
        nx_g = g.to_networkx()
        nx.draw(nx_g, nx.nx_agraph.graphviz_layout(nx_g, prog='dot'), with_labels = True)
        plt.show()

    # randomly occupy with nodes (not occupied=0 value):
    grid, grid_in = initial_fill(nodes, (args.grid_depth, args.grid_size, args.grid_size))

    # testing grid placement scoring:
    score_test = calc_score(grid_in, g, args)
    print('Initial score: ', score_test)
    if args.debug:
        print('Inital placement: ', grid_in)

    # random search
    if args.mode == 0 or args.mode == 2:
        # random search
        budget = args.epochs  # How many steps of training we will do before concluding.
        before_rs = score_test
        best_grid = grid_in.copy()

        print('Running Random search optimization ...')
        for i in tqdm(range(budget)):
            grid, grid_in = initial_fill(nodes, grid.shape)
            after_rs = calc_score(grid_in, g, args)
            if before_rs > after_rs:
                before_rs = after_rs
                best_grid = grid_in.copy()
                if args.debug:
                    print('best_grid score so far: ', after_rs)

        print('best score found: ', before_rs)
        if args.debug:
            print('optim placement: ', best_grid)

    # ES search
    if args.mode == 1 or args.mode == 2:
        import nevergrad as ng
        budget = args.epochs  # How many steps of training we will do before concluding.
        workers = 16
        # param = ng.p.Array(shape=(nodes, 3)).set_integer_casting() #3 coord: x, y, pipe
        param = ng.p.Array(init=grid_in).set_integer_casting()\
            .set_bounds(lower=0, upper=max(args.grid_size, args.grid_depth))
        # ES optim
        names = "CMA"
        optim = ng.optimizers.registry[names](parametrization=param, budget=budget, num_workers=workers)
        # optim = ng.optimizers.RandomSearch(parametrization=param, budget=budget, num_workers=workers)
        # optim = ng.optimizers.NGOpt(parametrization=param, budget=budget, num_workers=workers)

        final_es = score_test
        final_value = grid_in
        print('Running ES optimization ...')
        for _ in tqdm(range(budget)):
            x = optim.ask()
            loss = calc_score(x.value, g, args)
            optim.tell(x, loss)
        rec = optim.recommend()
        after_es = calc_score(rec.value, g, args)
        if final_es > after_es:
            final_value = rec.value
            final_es = after_es
        print('best score found:', final_es)
        if args.debug:
            print('optim placement:\n', final_value)

    # PPO
    if args.mode == 3:
        from ppo_discrete import PPO
        #RL RNN place each node
        env = GridEnv(args, grid_in, g) #change util.py line 61 state_dim
        # env = GridEnv(args)
        ppo = PPO(args, env)

        # logging variables
        running_reward = 0
        time_step = 0
        avg_improve = 0

        writer = SummaryWriter(comment='train')
        #writer.add_hparams(vars(args), {'hparam': 0})

        # training loop:
        print('Starting PPO training...')
        for i_episode in range(1, args.epochs + 1):
            state, initial_rl = env.reset()
            gr_edges = torch.stack(env.graph.edges()).unsqueeze(0).float()
            best_reward = initial_rl
            for node in range(args.nodes):
                time_step += 1
                node_1hot = torch.zeros(args.nodes)
                node_1hot[node] = 1.0
                rl_state = torch.cat((torch.FloatTensor(state).view(-1), node_1hot)) #grid, node to place
                action = ppo.select_action(rl_state, gr_edges)
                state, reward = env.step(action, node)
                # print('node: ', node); print('action: ', action); print('reward: ', reward); print('state: ', state)
                # input()
                # Saving reward and is_terminals:
                ppo.buffer.rewards.append(reward)
                if node == (args.nodes - 1):
                    done = True
                else:
                    done = False
                ppo.buffer.is_terminals.append(done)

                # learning:
                if time_step % args.update_timestep == 0:
                    ppo.update()
                    time_step = 0
                running_reward += reward
                best_reward = max(best_reward, reward)

            # print(final_rl, best_reward, (abs(final_rl) - abs(best_reward)))
            avg_improve += (abs(initial_rl) - abs(best_reward))

            if initial_rl > best_reward:
                print('got worse: ', initial_rl, best_reward)

            # logging
            if i_episode % args.log_interval == 0:
                running_reward = int((running_reward / args.log_interval))
                avg_improve = int(avg_improve / args.log_interval)
                writer.add_scalar('running_reward/episode', running_reward, i_episode)
                print('Episode {} \t Avg reward: {}'.format(i_episode, running_reward))
                writer.add_scalar('final_reward/episode', best_reward, i_episode)
                print('Episode {} \t Final reward: {}'.format(i_episode, best_reward))
                # writer.add_scalar('avg improvement/episode', avg_improve, i_episode)
                # print('Episode {} \t Avg improvement: {}'.format(i_episode, avg_improve))

                torch.save(ppo.policy.state_dict(), 'model_epoch_' + str(avg_improve) + '.pth')

                running_reward = avg_improve = 0

    # DQN
    if args.mode == 4:
        from qlearn import DQNAgent

        # RL RNN place each node
        env = GridEnv(args, grid_in, g)
        dqn = DQNAgent(args, env)

        # logging variables
        running_reward = 0
        time_step = 0
        final_rl = -score_test  # invert score. RL maximize reward
        writer = SummaryWriter(comment='train')

        # training loop:
        print('Starting DQN training...')
        for i_episode in range(1, args.epochs + 1):
            state, _ = env.reset()
            for node in range(args.nodes):
                time_step += 1
                state = torch.FloatTensor(state).view(1, -1)
                action = dqn.select_action(state, i_episode)
                next_state, reward = env.step(action.cpu().numpy(), node)
                next_state = torch.FloatTensor(next_state).view(1, -1)
                reward = torch.IntTensor([int(reward)])

                if node == (args.nodes - 1):
                    done = True
                    next_state = None
                else:
                    done = False

                # update memory and states:
                dqn.memory.push(state, action, next_state, reward)
                state = next_state

                # learning:
                dqn.optimize_model()
                running_reward += reward.item()

            # final score
            if final_rl < reward.item():
                final_rl = reward.item()  # last reward is score with all nodes placed
                # save if final score is better
                torch.save(dqn.policy_net.state_dict(), 'model_epoch_' + str(int(final_rl)) + '.pth')

            # logging
            if i_episode % args.log_interval == 0:
                running_reward = int((running_reward / args.log_interval))
                writer.add_scalar('running_reward/episode', running_reward, i_episode)
                print('Episode {} \t Avg reward: {}'.format(i_episode, running_reward))
                writer.add_scalar('final_reward/episode', final_rl, i_episode)
                print('Episode {} \t Final reward: {}'.format(i_episode, final_rl))
                running_reward = 0

        # print('Q-Learn')
        # ag = Q_learn(args, g)
        # ag.play(5000)
        # ag.test()

    # GraphNet - supervised
    if args.mode == 5:
        from torch.utils.data import DataLoader
        from torch.utils.data.dataset import Dataset
        from torch import optim
        from model import GraphNet

        class GraphDataset(Dataset):
            def __init__(self, data):
                #data 0: graph, 1: grid, 2: score
                #pre-process graph and grid to model input
                # model CNN: input list of edges, output node idx vector
                gg = []
                for g in data:
                    gg.append(torch.stack(g[0].edges()).unsqueeze(0).float())
                self.graph = torch.cat(gg)
                gg = []
                for g in data:
                    gg.append(torch.tensor(g[1]).unsqueeze(0).float())
                self.grid = torch.cat(gg)

            def __getitem__(self, index):
                return self.graph[index], self.grid[index]

            def __len__(self):
                return self.graph.shape[0]

        data = torch.load(args.dataset)
        writer = SummaryWriter(comment='train')

        train_set = 0.8  # 80% for train
        t_portion = int(len(data) * train_set)

        dataset = GraphDataset(data)
        train_set, val_set = torch.utils.data.random_split(dataset, [t_portion, int(len(data) - t_portion)])

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=8, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=8, drop_last=True, pin_memory=True)

        # in_size (bs, 2, edges)
        # out_size (bs, nodes idx)
        graph_shape = (args.grid_depth, args.grid_size, args.grid_size)
        model = GraphNet(args.nodes)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        best_test_loss = float("inf")

        for epoch in range(args.epochs):
            # train with examples first:
            train_loss = model.train_step(model, train_loader, criterion, optimizer)
            test_loss = model.train_step(model, val_loader, criterion, optimizer, False)
            # save model when test error is lower:
            writer.add_scalar('Train Loss/epoch', train_loss, epoch)
            writer.add_scalar('Val Loss/epoch', test_loss, epoch)
            print('Train Loss', train_loss, 'Val Loss', test_loss, ' epoch ', epoch)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss
                }
                filename_net = 'model_epoch.pth'
                torch.save(checkpoint, filename_net)

        #test
        




