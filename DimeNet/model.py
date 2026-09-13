import torch
import torch.nn as nn

from torch_geometric.nn import radius_graph
from torch_geometric.nn import global_add_pool

from basis import (
    BesselBasisLayer,
    SphericalBasisLayer
)

from embedding import EmbeddingBlock
from interaction import InteractionBlock
from output import OutputBlock

from triplet import triplets


class DimeNet(nn.Module):

    def __init__(
        self,
        hidden_channels=128,
        out_channels=1,
        num_blocks=6,
        num_bilinear=8,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act=torch.nn.functional.silu
    ):
        super().__init__()

        self.cutoff = cutoff
        self.num_blocks = num_blocks
        self.embedding = EmbeddingBlock(
            num_radial=num_radial,
            hidden_channels=hidden_channels,
            act=act
        )
        self.bessel = BesselBasisLayer(
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent
        )
        self.spherical_basis = SphericalBasisLayer(
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent
        )
        self.interaction_blocks = nn.ModuleList()

        for _ in range(num_blocks):

            self.interaction_blocks.append(
                InteractionBlock(
                    hidden_channels=hidden_channels,
                    num_radial=num_radial,
                    num_spherical=num_spherical,
                    num_bilinear=num_bilinear,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    act=act
                )
            )
        self.output_blocks = nn.ModuleList()

        for _ in range(num_blocks + 1):

            self.output_blocks.append(
                OutputBlock(
                    num_radial=num_radial,
                    hidden_channels=hidden_channels,
                    num_layers=num_output_layers,
                    out_channels=out_channels,
                    act=act
                )
            )
    def forward(
        self,
        z,
        pos,
        batch
    ):

        num_nodes = z.size(0)
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            loop=False
        )
        i, j = edge_index
        dist = (
            pos[i] - pos[j]
        ).pow(2).sum(
            dim=-1
        ).sqrt()
        (
            i,
            j,
            idx_i,
            idx_j,
            idx_k,
            idx_kj,
            idx_ji
        ) = triplets(
            edge_index,
            num_nodes
        )
        pos_ji = (
            pos[idx_j]
            - pos[idx_i]
        )

        pos_kj = (
            pos[idx_k]
            - pos[idx_i]
        )

        a = (
            pos_ji * pos_kj
        ).sum(
            dim=-1
        )

        b = torch.cross(
            pos_ji,
            pos_kj,
            dim=1
        ).norm(
            dim=-1
        )

        angle = torch.atan2(
            b,
            a
        )
        rbf = self.bessel(
            dist
        )
        sbf = self.spherical_basis(
            dist,
            angle,
            idx_kj
        )
        x = self.embedding(
            z,
            rbf,
            i,
            j
        )
        output = self.output_blocks[0](
            x,
            rbf,
            i,
            num_nodes
        )
        for interaction_block, output_block in zip(
            self.interaction_blocks,
            self.output_blocks[1:]
        ):

            x = interaction_block(
                x,
                rbf,
                sbf,
                idx_kj,
                idx_ji
            )

            output = (
                output
                + output_block(
                    x,
                    rbf,
                    i,
                    num_nodes
                )
            )
        output = global_add_pool(
            output,
            batch
        )

        return output
