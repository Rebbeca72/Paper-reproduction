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

        # ==========================================
        # 1. RBF → hidden_channels
        # ==========================================

        self.lin_rbf = Linear(
            num_radial,
            hidden_channels,
            bias=False
        )

        # ==========================================
        # 2. SBF → num_bilinear
        # ==========================================

        self.lin_sbf = Linear(
            num_spherical * num_radial,
            num_bilinear,
            bias=False
        )

        # ==========================================
        # 3. k → j
        # ==========================================

        self.lin_kj = Linear(
            hidden_channels,
            hidden_channels
        )

        # ==========================================
        # 4. j → i
        # ==========================================

        self.lin_ji = Linear(
            hidden_channels,
            hidden_channels
        )

        # ==========================================
        # 5. Bilinear interaction parameter
        # ==========================================

        self.W = nn.Parameter(
            torch.empty(
                hidden_channels,
                num_bilinear,
                hidden_channels
            )
        )

        # ==========================================
        # 6. skip connection 前 residual
        # ==========================================

        self.layers_before_skip = nn.ModuleList()

        for _ in range(num_before_skip):

            self.layers_before_skip.append(
                ResidualLayer(
                    hidden_channels,
                    act
                )
            )

        # ==========================================
        # 7. skip connection
        # ==========================================

        self.lin = Linear(
            hidden_channels,
            hidden_channels
        )

        # ==========================================
        # 8. skip connection 后 residual
        # ==========================================

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

        # ==========================================
        # RBF linear
        # ==========================================

        glorot_orthogonal(
            self.lin_rbf.weight,
            scale=2.0
        )

        # ==========================================
        # SBF linear
        # ==========================================

        glorot_orthogonal(
            self.lin_sbf.weight,
            scale=2.0
        )

        # ==========================================
        # k → j
        # ==========================================

        glorot_orthogonal(
            self.lin_kj.weight,
            scale=2.0
        )

        nn.init.zeros_(
            self.lin_kj.bias
        )

        # ==========================================
        # j → i
        # ==========================================

        glorot_orthogonal(
            self.lin_ji.weight,
            scale=2.0
        )

        nn.init.zeros_(
            self.lin_ji.bias
        )

        # ==========================================
        # Bilinear W
        # ==========================================

        self.W.data.normal_(
            mean=0.0,
            std=2.0 / self.W.size(0)
        )

        # ==========================================
        # skip connection
        # ==========================================

        glorot_orthogonal(
            self.lin.weight,
            scale=2.0
        )

        nn.init.zeros_(
            self.lin.bias
        )

        # ==========================================
        # residual layers
        # ==========================================

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

        # ==========================================
        # 1. RBF
        # ==========================================

        rbf = self.lin_rbf(
            rbf
        )

        # [E, num_radial]
        #       ↓
        # [E, hidden_channels]


        # ==========================================
        # 2. SBF
        # ==========================================

        sbf = self.lin_sbf(
            sbf
        )

        # [T, num_spherical * num_radial]
        #       ↓
        # [T, num_bilinear]


        # ==========================================
        # 3. j → i
        # ==========================================

        x_ji = self.lin_ji(
            x
        )

        x_ji = self.act(
            x_ji
        )

        # [E, hidden_channels]


        # ==========================================
        # 4. k → j
        # ==========================================

        x_kj = self.lin_kj(
            x
        )

        x_kj = self.act(
            x_kj
        )

        # [E, hidden_channels]


        # ==========================================
        # 5. RBF 调制 k → j
        # ==========================================

        x_kj = x_kj * rbf

        # [E, hidden_channels]


        # ==========================================
        # 6. 取出每个 triplet 对应的 k → j
        # ==========================================

        x_kj = x_kj[
            idx_kj
        ]

        # [T, hidden_channels]


        # ==========================================
        # 7. Bilinear interaction
        # ==========================================

        x_kj = torch.einsum(
            "wj,wl,ijl->wi",
            sbf,
            x_kj,
            self.W
        )

        # [T, hidden_channels]


        # ==========================================
        # 8. triplet message 聚合到 j → i edge
        # ==========================================

        x_kj = scatter(
            x_kj,
            idx_ji,
            dim=0,
            dim_size=x.size(0),
            reduce="sum"
        )

        # [E, hidden_channels]


        # ==========================================
        # 9. 加上 j → i 本身的信息
        # ==========================================

        h = x_ji + x_kj

        # [E, hidden_channels]


        # ==========================================
        # 10. skip connection 前 residual
        # ==========================================

        for layer in self.layers_before_skip:

            h = layer(h)

        # [E, hidden_channels]


        # ==========================================
        # 11. skip connection
        # ==========================================

        h = self.act(
            self.lin(h)
        )

        h = h + x

        # [E, hidden_channels]


        # ==========================================
        # 12. skip connection 后 residual
        # ==========================================

        for layer in self.layers_after_skip:

            h = layer(h)

        # [E, hidden_channels]

        return h