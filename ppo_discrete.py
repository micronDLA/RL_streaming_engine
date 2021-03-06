
import torch
import torch.nn as nn

torch.manual_seed(0)
_engine = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from modules import RolloutBuffer, ActorCritic

class PPO:
    def __init__(self,
                 args,
                 graphdef,
                 device,
                 state_dim,
                 ntasks = 1):

        #ntasks: number of different graphs
        self.args = args
        self.device_topology = device['topology']
        self.ntasks = ntasks
        self.state_dim = state_dim  # Input ready time (Number of tiles slices, 1)
        self.action_dim = device['action_dim'] #output (nodes, 48)
        self.gnn_in = graphdef['graph'].ndata['feat'].shape[1]
        self.buffer = RolloutBuffer()
        self.ntokens = args.device_topology

        self.policy = ActorCritic(args=args,
                                  device=device,
                                  state_dim=self.state_dim,
                                  emb_size=self.args.emb_size,
                                  action_dim=self.action_dim,
                                  graph_feat_size=self.args.graph_feat_size,
                                  gnn_in=self.gnn_in,
                                  ntasks=ntasks).to(_engine)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.lr, betas=self.args.betas)

        self.policy_old = ActorCritic(args=args,
                                      device=device,
                                      state_dim=self.state_dim,
                                      emb_size=self.args.emb_size,
                                      action_dim=self.action_dim,
                                      graph_feat_size=self.args.graph_feat_size,
                                      gnn_in=self.gnn_in,
                                      ntasks=ntasks).to(_engine)
        self.policy_old.load_state_dict(self.policy.state_dict())
        if args.model != '':
            self.load(args.model)

        self.MseLoss = nn.MSELoss()

    def select_action(self, tensor_in, graphdef, node_id, mask):
        with torch.no_grad():
            graph_info = graphdef['graph'].to(_engine)
            state = torch.FloatTensor(tensor_in).to(_engine)
            mask = torch.tensor(mask, dtype=torch.bool).to(_engine)
            node_id = torch.atleast_2d(torch.tensor(node_id)).to(_engine)
            action, action_logprob = self.policy_old.act(state, graph_info, node_id, mask)

        return action.item(), (state, action, graph_info, action_logprob, mask, node_id)

    def add_buffer(self, inbuff, reward, done):
        state, action, graph_info, action_logprob, mask, node_id = inbuff
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.graphs.append(graph_info)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)
        self.buffer.masks.append(mask)
        self.buffer.node_ids.append(node_id)


    def update(self):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.args.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(_engine)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # rewards = rewards.float().squeeze()

        # convert list to tensor
        old_masks = 0  # Used in transformer mode
        old_masks = torch.squeeze(torch.stack(self.buffer.masks, dim=0)).detach().to(_engine)
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(_engine)
        old_graph = [graph.to(_engine) for graph in self.buffer.graphs]
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(_engine)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(_engine)
        old_node_ids = torch.vstack(self.buffer.node_ids).detach().to(_engine)

        # Optimize policy for K epochs
        for _ in range(self.args.K_epochs):

            # Evaluating old actions and values
            if self.args.nnmode == 'transformer':
                logprobs, state_values, dist_entropy = self.policy.evaluate_seq((old_states, old_masks), old_actions, old_graph)
            else:
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_graph, old_masks, old_node_ids)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + \
                   + self.args.loss_value_c*self.MseLoss(state_values, rewards) + \
                   - self.args.loss_entropy_c*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        print('Loaded model \n')
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
