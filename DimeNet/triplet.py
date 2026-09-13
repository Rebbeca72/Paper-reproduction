import torch
from torch_sparse import SparseTensor


def triplets(edge_index, num_nodes):

    row, col = edge_index

    # 给每一条 edge 一个唯一编号
    value = torch.arange(
        row.size(0),
        device=row.device
    )

    # 构造转置邻接矩阵
    adj_t = SparseTensor(
        row=col,
        col=row,
        value=value,
        sparse_sizes=(num_nodes, num_nodes)
    )

    # 对每一条 j -> i 的边，
    # 找到所有 k -> j 的边
    adj_t_row = adj_t[row]

    # 每条 j -> i 对应多少个 k -> j
    num_triplets = (
        adj_t_row
        .set_value(None)
        .sum(dim=1)
        .to(torch.long)
    )

    # --------------------------------------------------------
    # 三个原子索引
    # --------------------------------------------------------

    idx_i = col.repeat_interleave(
        num_triplets
    )

    idx_j = row.repeat_interleave(
        num_triplets
    )

    idx_k = adj_t_row.storage.col()

    # --------------------------------------------------------
    # 删除 k == i
    #
    # 即删除：
    #
    # i <- j <- i
    #
    # 这种来回走的 triplet
    # --------------------------------------------------------

    mask = idx_i != idx_k

    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    idx_k = idx_k[mask]

    # --------------------------------------------------------
    # 两条边对应的 edge ID
    # --------------------------------------------------------

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