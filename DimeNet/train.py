import torch
import torch.nn as nn
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from model import DimeNet

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

dataset = QM9(root="./data/QM9")

print("Dataset size:", len(dataset))
num_data = len(dataset)

train_size = int(0.8 * num_data)
val_size = int(0.1 * num_data)

train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:train_size + val_size]
test_dataset = dataset[train_size + val_size:]

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Test size:", len(test_dataset))
target_index = 7

train_targets = torch.stack([
    data.y[:, target_index]
    for data in train_dataset
])

target_mean = train_targets.mean().to(device)
target_std = train_targets.std().to(device)

print("Target mean:", target_mean)
print("Target std:", target_std)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
model = DimeNet(
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
    num_output_layers=3
).to(device)

print(model)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

loss_fn = nn.MSELoss()

num_epochs = 3

train_losses = []
val_losses = []
for epoch in range(1, num_epochs + 1):
    model.train()

    total_train_loss = 0.0
    total_train_samples = 0

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()
        target = batch.y[:, target_index]

        # Normalize target
        target = (
            target - target_mean
        ) / target_std
        pred = model(
            z=batch.z,
            pos=batch.pos,
            batch=batch.batch
        ).view(-1)
        loss = loss_fn(
            pred,
            target
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()
        batch_size = batch.num_graphs

        total_train_loss += (
            loss.item() * batch_size
        )

        total_train_samples += batch_size

    train_loss = (
        total_train_loss /
        total_train_samples
    )

    train_losses.append(train_loss)
    model.eval()

    total_val_loss = 0.0
    total_val_mae = 0.0
    total_val_samples = 0

    with torch.no_grad():

        for batch in val_loader:

            batch = batch.to(device)
            target = batch.y[:, target_index]

            normalized_target = (
                target - target_mean
            ) / target_std
            pred = model(
                z=batch.z,
                pos=batch.pos,
                batch=batch.batch
            ).view(-1)
            loss = loss_fn(
                pred,
                normalized_target
            )
            pred_original = (
                pred * target_std
                + target_mean
            )
            mae = torch.abs(
                pred_original - target
            ).mean()

            batch_size = batch.num_graphs

            total_val_loss += (
                loss.item() * batch_size
            )

            total_val_mae += (
                mae.item() * batch_size
            )

            total_val_samples += batch_size

    val_loss = (
        total_val_loss /
        total_val_samples
    )

    val_mae = (
        total_val_mae /
        total_val_samples
    )

    val_losses.append(val_loss)
    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Val MAE: {val_mae:.6f}"
    )
model.eval()

total_test_mae = 0.0
total_test_rmse = 0.0
total_test_samples = 0

with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(device)
        target = batch.y[:, target_index]

        normalized_target = (
            target - target_mean
        ) / target_std

        pred = model(
            z=batch.z,
            pos=batch.pos,
            batch=batch.batch
        ).view(-1)

        pred_original = (
            pred * target_std
            + target_mean
        )

        mae = torch.abs(
            pred_original - target
        ).sum()

        squared_error = (
            (pred_original - target) ** 2
        ).sum()

        batch_size = batch.num_graphs

        total_test_mae += mae.item()
        total_test_rmse += squared_error.item()

        total_test_samples += batch_size


test_mae = (
    total_test_mae /
    total_test_samples
)

test_rmse = (
    total_test_rmse /
    total_test_samples
) ** 0.5


print()
print("==============================")
print("Test MAE :", test_mae)
print("Test RMSE:", test_rmse)
print("==============================")

import matplotlib.pyplot as plt

plt.figure()

plt.plot(
    range(1, num_epochs + 1),
    train_losses,
    label="Train Loss"
)

plt.plot(
    range(1, num_epochs + 1),
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("DimeNet Training Curve")

plt.show()
