import torch
import torch.nn as nn

from torch_geometric.nn import Linear

from dimenet_utils import glorot_orthogonal


class ResidualLayer(nn.Module):

    def __init__(
        self,
        hidden_channels,
        act
    ):
        super().__init__()

        self.lin1 = Linear(
            hidden_channels,
            hidden_channels
        )

        self.lin2 = Linear(
            hidden_channels,
            hidden_channels
        )

        self.act = act

        self.reset_parameters()

    def reset_parameters(self):

        glorot_orthogonal(
            self.lin1.weight,
            scale=2.0
        )

        nn.init.zeros_(
            self.lin1.bias
        )

        glorot_orthogonal(
            self.lin2.weight,
            scale=2.0
        )

        nn.init.zeros_(
            self.lin2.bias
        )

    def forward(self, x):

        x = self.lin1(x)

        x = self.act(x)

        x = self.lin2(x)

        x = self.act(x)

        return x