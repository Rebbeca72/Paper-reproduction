import torch
from torch_sparse import SparseTensor


def triplets(edge_index, num_nodes):

    row, col = edge_index
    value = torch.arange(
        row.size(0),
        device=row.device
    )
    adj_t = SparseTensor(
        row=col,
        col=row,
        value=value,
        sparse_sizes=(num_nodes, num_nodes)
    )
    adj_t_row = adj_t[row]
    num_triplets = (
        adj_t_row
        .set_value(None)
        .sum(dim=1)
        .to(torch.long)
    )
    idx_i = col.repeat_interleave(
        num_triplets
    )

    idx_j = row.repeat_interleave(
        num_triplets
    )

    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k

    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    idx_k = idx_k[mask]
    idx_kj = adj_t_row.storage.value()[mask]

    idx_ji = adj_t_row.storage.row()[mask]

    return (
        col,
        row,
        idx_i,
        idx_j,
        idx_k,
        idx_kj,
        idx_ji
    )
