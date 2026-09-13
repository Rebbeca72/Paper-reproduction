import torch
import torch.nn as nn

from torch_geometric.nn import Linear
from torch_geometric.utils import scatter

from dimenet_utils import glorot_orthogonal


class OutputBlock(nn.Module):

    def __init__(
        self,
        num_radial,
        hidden_channels,
        num_layers,
        out_channels,
        act
    ):
        super().__init__()

        self.act = act

        # RBF → hidden dimension
        self.lin_rbf = Linear(
            num_radial,
            hidden_channels,
            bias=False
        )

        # 输出 MLP
        self.lins = nn.ModuleList()

        for _ in range(num_layers):
            self.lins.append(
                Linear(
                    hidden_channels,
                    hidden_channels
                )
            )

        # 最终输出
        self.lin = Linear(
            hidden_channels,
            out_channels,
            bias=False
        )

        self.reset_parameters()

    def reset_parameters(self):

        glorot_orthogonal(
            self.lin_rbf.weight,
            scale=2.0
        )

        for layer in self.lins:

            glorot_orthogonal(
                layer.weight,
                scale=2.0
            )

            nn.init.zeros_(
                layer.bias
            )

        nn.init.zeros_(
            self.lin.weight
        )

    def forward(
        self,
        x,
        rbf,
        i,
        num_nodes
    ):

        # -----------------------------------------
        # 1. RBF → hidden
        # -----------------------------------------

        rbf = self.lin_rbf(
            rbf
        )

        # -----------------------------------------
        # 2. edge feature × RBF
        # -----------------------------------------

        x = rbf * x

        # -----------------------------------------
        # 3. edge → atom
        # -----------------------------------------

        x = scatter(
            x,
            i,
            dim=0,
            dim_size=num_nodes,
            reduce="sum"
        )

        # -----------------------------------------
        # 4. atom-level MLP
        # -----------------------------------------

        for layer in self.lins:

            x = self.act(
                layer(x)
            )

        # -----------------------------------------
        # 5. atom output
        # -----------------------------------------

        x = self.lin(x)

        return x