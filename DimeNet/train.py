import torch
import torch.nn as nn
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from model import DimeNet


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. Load QM9
# ============================================================

dataset = QM9(root="./data/QM9")

print("Dataset size:", len(dataset))


# ============================================================
# 3. Train / Validation / Test split
# ============================================================

num_data = len(dataset)

train_size = int(0.8 * num_data)
val_size = int(0.1 * num_data)

train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:train_size + val_size]
test_dataset = dataset[train_size + val_size:]

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Test size:", len(test_dataset))


# ============================================================
# 4. Target
# ============================================================

# QM9 target index
# 7 = U0
target_index = 7

train_targets = torch.stack([
    data.y[:, target_index]
    for data in train_dataset
])

target_mean = train_targets.mean().to(device)
target_std = train_targets.std().to(device)

print("Target mean:", target_mean)
print("Target std:", target_std)


# ============================================================
# 5. DataLoader
# ============================================================

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


# ============================================================
# 6. Model
# ============================================================

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


# ============================================================
# 7. Optimizer and Loss
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

loss_fn = nn.MSELoss()


# ============================================================
# 8. Training settings
# ============================================================

num_epochs = 3

train_losses = []
val_losses = []


# ============================================================
# 9. Training
# ============================================================

for epoch in range(1, num_epochs + 1):

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.train()

    total_train_loss = 0.0
    total_train_samples = 0

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        # ----------------------------------------------------
        # QM9 target
        # ----------------------------------------------------

        target = batch.y[:, target_index]

        # Normalize target
        target = (
            target - target_mean
        ) / target_std

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        pred = model(
            z=batch.z,
            pos=batch.pos,
            batch=batch.batch
        ).view(-1)

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = loss_fn(
            pred,
            target
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

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


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    total_val_loss = 0.0
    total_val_mae = 0.0
    total_val_samples = 0

    with torch.no_grad():

        for batch in val_loader:

            batch = batch.to(device)

            # ------------------------------------------------
            # Target
            # ------------------------------------------------

            target = batch.y[:, target_index]

            normalized_target = (
                target - target_mean
            ) / target_std

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            pred = model(
                z=batch.z,
                pos=batch.pos,
                batch=batch.batch
            ).view(-1)

            # ------------------------------------------------
            # Validation loss
            # ------------------------------------------------

            loss = loss_fn(
                pred,
                normalized_target
            )

            # ------------------------------------------------
            # Convert prediction back to original unit
            # ------------------------------------------------

            pred_original = (
                pred * target_std
                + target_mean
            )

            # ------------------------------------------------
            # MAE
            # ------------------------------------------------

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


    # ========================================================
    # Print
    # ========================================================

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Val MAE: {val_mae:.6f}"
    )


# ============================================================
# 10. Test
# ============================================================

model.eval()

total_test_mae = 0.0
total_test_rmse = 0.0
total_test_samples = 0

with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(device)

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        target = batch.y[:, target_index]

        normalized_target = (
            target - target_mean
        ) / target_std

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        pred = model(
            z=batch.z,
            pos=batch.pos,
            batch=batch.batch
        ).view(-1)

        # ----------------------------------------------------
        # Convert back to original unit
        # ----------------------------------------------------

        pred_original = (
            pred * target_std
            + target_mean
        )

        # ----------------------------------------------------
        # MAE
        # ----------------------------------------------------

        mae = torch.abs(
            pred_original - target
        ).sum()

        # ----------------------------------------------------
        # Squared error
        # ----------------------------------------------------

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


# ============================================================
# 11. Loss curve
# ============================================================

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