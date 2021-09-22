import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import MultivariateNormal # continuous
from torch.distributions import Categorical # discrete
import numpy as np
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GraphNet(nn.Module):
    def __init__(self, out_size, dropout=0.2):
        super(GraphNet, self).__init__()
        graph_emb = 64
        self.cg_conv = nn.Conv1d(2, graph_emb, 1) #2: g.edges src node, dst node
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(graph_emb)

        #class
        self.l1 = nn.Linear(graph_emb, graph_emb)
        self.l2 = nn.Linear(graph_emb, graph_emb)
        self.lo = nn.Linear(graph_emb, out_size)

    def forward(self, x):
        x = self.cg_conv(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(self.norm(x))

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.lo(x)

    def train_step(self, model, train_loader, criterion, optimizer, train=True):
        self.to(device)
        if train:
            model.train()
        else:
            model.eval()
        train_loss = 0
        for _, (sample, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            input_tensor = sample.to(device) # (bs, 2, edges)
            target = target.to(device) # (bs, nodes)
            optimizer.zero_grad()
            predictions = self.forward(input_tensor)
            print('predictions: ', predictions, 'target: ', target)
            loss = criterion(predictions, target)  # compare to next values
            if train:
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader)

