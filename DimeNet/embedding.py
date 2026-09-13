import torch
import torch.nn as nn
from torch.nn import Embedding
from torch_geometric.nn import Linear
'''对原子距离和原子种类映射'''
class EmbeddingBlock(nn.Module):
    def __init__(self,num_radial,hidden_channels,act):
        super().__init__()
        self.act = act
        self.emb = Embedding(95,hidden_channels)
        self.lin_rbf = Linear(num_radial,hidden_channels)
        self.lin = Linear(3 * hidden_channels,hidden_channels)
        self.reset_parameters()
    def reset_parameters(self):
        self.emb.reset_parameters()
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()
    def forward(self,x,rbf,i,j):
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        x = self.lin(torch.cat([x[i],x[j],rbf],dim=-1))
        x = self.act(x)
        return x
