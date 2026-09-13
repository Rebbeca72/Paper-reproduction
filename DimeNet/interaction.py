import torch
import torch.nn as nn

from torch_geometric.nn import Linear
from torch_geometric.utils import scatter

from residual import ResidualLayer
from dimenet_utils import glorot_orthogonal


class InteractionBlock(nn.Module):

    def __init__(
        self,
        hidden_channels,
        num_radial,
        num_spherical,
        num_bilinear,
        num_before_skip,
        num_after_skip,
        act
    ):
        super().__init__()

        self.act = act
        self.lin_rbf = Linear(
            num_radial,
            hidden_channels,
            bias=False
        )
        self.lin_sbf = Linear(
            num_spherical * num_radial,
            num_bilinear,
            bias=False
        )
        self.lin_kj = Linear(
            hidden_channels,
            hidden_channels
        )
        self.lin_ji = Linear(
            hidden_channels,
            hidden_channels
        )
        self.W = nn.Parameter(
            torch.empty(
                hidden_channels,
                num_bilinear,
                hidden_channels
            )
        )
        self.layers_before_skip = nn.ModuleList()

        for _ in range(num_before_skip):

            self.layers_before_skip.append(
                ResidualLayer(
                    hidden_channels,
                    act
                )
            )
        self.lin = Linear(
            hidden_channels,
            hidden_channels
        )
        self.layers_after_skip = nn.ModuleList()

        for _ in range(num_after_skip):

            self.layers_after_skip.append(
                ResidualLayer(
                    hidden_channels,
                    act
                )
            )

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(
            self.lin_rbf.weight,
            scale=2.0
        )
        glorot_orthogonal(
            self.lin_sbf.weight,
            scale=2.0
        )
        glorot_orthogonal(
            self.lin_kj.weight,
            scale=2.0
        )

        nn.init.zeros_(
            self.lin_kj.bias
        )
        glorot_orthogonal(
            self.lin_ji.weight,
            scale=2.0
        )

        nn.init.zeros_(
            self.lin_ji.bias
        )
        self.W.data.normal_(
            mean=0.0,
            std=2.0 / self.W.size(0)
        )
        glorot_orthogonal(
            self.lin.weight,
            scale=2.0
        )

        nn.init.zeros_(
            self.lin.bias
        )
        for layer in self.layers_before_skip:
            layer.reset_parameters()

        for layer in self.layers_after_skip:
            layer.reset_parameters()

    def forward(
        self,
        x,
        rbf,
        sbf,
        idx_kj,
        idx_ji
    ):
        rbf = self.lin_rbf(
            rbf
        )
        sbf = self.lin_sbf(
            sbf
        )
        x_ji = self.lin_ji(
            x
        )

        x_ji = self.act(
            x_ji
        )
        x_kj = self.lin_kj(
            x
        )

        x_kj = self.act(
            x_kj
        )
        x_kj = x_kj * rbf
        x_kj = x_kj[
            idx_kj
        ]
        x_kj = torch.einsum(
            "wj,wl,ijl->wi",
            sbf,
            x_kj,
            self.W
        )
        x_kj = scatter(
            x_kj,
            idx_ji,
            dim=0,
            dim_size=x.size(0),
            reduce="sum"
        )
        h = x_ji + x_kj
        for layer in self.layers_before_skip:

            h = layer(h)
        h = self.act(
            self.lin(h)
        )

        h = h + x
        for layer in self.layers_after_skip:

            h = layer(h)

        return h
