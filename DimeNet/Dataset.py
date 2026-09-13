import torch
from torch_geometric.datasets import QM9
dataset = QM9(root="./data/QM9")
print("Dataset size:", len(dataset))
data = dataset[0]
print(data)
print("z:", data.z)
print("pos shape:", data.pos.shape)
print("y shape:", data.y.shape)