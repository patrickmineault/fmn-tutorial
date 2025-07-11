# %% [markdown]
# # Tutorial 1: Masked Autoencoding for Neural Data
# 
# This tutorial walks through the core pieces of the repository and shows how to
# train a small masked autoencoder on neural spike data. We work with the Lorenz
# dataset provided with this repository. The same code can be adapted to larger
# datasets such as `mc_maze`.
#
# We will:
#
# 1. Load spike data using the `SpikesDataset` class.
# 2. Visualize example spike trains.
# 3. Build a small UNet-based autoencoder.
# 4. Train the model using masked autoencoding.
# 5. Inspect the learned representation.
#
# Running this notebook requires the dependencies listed in `requirements.txt`.
# Make sure the toy H5 files have been downloaded into `data/h5` as explained in
# the repository `README.md`.

# %% [markdown]
# ## Imports
# First we import PyTorch and the helper utilities from the `src` package.
# %%

import os
from pathlib import Path

from h5py import h5
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# %%
if not os.path.exists("../data/h5/lfads_lorenz.h5"):
    !wget -P data/h5 https://github.com/snel-repo/neural-data-transformers/raw/refs/heads/master/data/lfads_lorenz.h5

if not os.path.exists("..data/h5/000140/sub-Jenkins"):
    !dandi download DANDI:000140 -o data/h5

# %% [markdown]
import h5py
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
import numpy as np

dataset_map = {
    "mc_maze": "000128",
    "mc_maze_small": "000140",
    "mc_maze_medium": "000139",
    "mc_maze_large": "000138",
    "mc_rtt": "000129",
    "area2_bump": "000127",
    "dmfc_rsg": "000130",
}

def load_dataset(dataset_name, path):
    if dataset_name == "lorenz":
        # This is a simple hdf5 file, no need to do anything fancy
        with open(f"{path}/h5/lfads_lorenz.h5", "rb") as f:
            h5_file = h5py.File(f, "r")
            loaded_dataset = {}
            for key in h5_file.keys():
                loaded_dataset[key.replace("valid_", "val_")] = h5_file[key][:]
            # All the tensors are in trials X time X neurons format
            return loaded_dataset

    elif dataset_name in dataset_map.keys():
        # Load the appropriate dataset
        fpath = f'{path}/h5/{dataset_map[dataset_name]}/sub-Jenkins'
        dataset = NWBDataset(fpath=fpath)

        # We use 10 ms bins to make the data more manageable
        dataset.resample(10)

        train_dict = make_train_input_tensors(dataset, dataset_name=dataset_name, trial_split="train", save_file=False, include_behavior=True)
        val_dict = make_train_input_tensors(dataset, dataset_name=dataset_name, trial_split="val", save_file=False, include_behavior=True)

        # Stack the heldin and heldout spikes. Although this is an important element of the design of the NLB, we won't touch on this here.
        data = {
            "train_data": np.concatenate([train_dict['train_spikes_heldin'], train_dict['train_spikes_heldout']], axis=2),
            "val_data": np.concatenate([val_dict['train_spikes_heldin'], val_dict['train_spikes_heldout']], axis=2),
            "train_behavior": train_dict['train_behavior'],
            "val_behavior": val_dict['train_behavior'],
            "spike_sources": list(dataset.data.spikes.columns) + list(dataset.data.heldout_spikes.columns)
        }

        return data

dataset = load_dataset("mc_maze_small", "data")

# %%
# Let's examine one trial
import matplotlib.pyplot as plot

# One figure, two axes stacked vertically
fig, (ax_top, ax_bottom) = plt.subplots(
    nrows=2, ncols=2,           # two rows, one column
    gridspec_kw={'height_ratios': [1, 2], 'width_ratios': [2, 1]},  # 1 : 2  ⇒ top = ⅓, bottom = ⅔
    figsize=(6, 8),             # any size you like
    sharex=False                 # optional: share the x-axis
)

bin_size = .01

ax_top[0].plot(np.arange(dataset['train_behavior'].shape[1]) * bin_size, dataset['train_behavior'][0, :])
ax_top[0].legend(['Velocity (x)', 'Velocity (y)'])
ax_top[0].set_xlim(0, dataset['train_behavior'].shape[1] * bin_size)
ax_top[0].set_ylim(-600, 600)

ax_top[1].plot(np.cumsum(dataset['train_behavior'][0, :, 0]), np.cumsum(dataset['train_behavior'][0, :, 1]), '-.')
ax_top[1].set_xlabel('position (x)')
ax_top[1].set_ylabel('position (y)')
ax_top[1].set_xlim([-10000, 10000])
ax_top[1].set_ylim([-10000, 10000])
ax_top[1].plot(np.cumsum(dataset['train_behavior'][0, :, 0])[-1], np.cumsum(dataset['train_behavior'][0, :, 1])[-1], 'gx')  # mark the end
ax_top[1].plot(0, 0, 'ro')

ax_bottom[0].imshow(dataset['train_data'][0, :, :].T, cmap='gray_r', aspect='auto', extent=[0, dataset['train_data'].shape[1] * bin_size, 0, dataset['train_data'].shape[2]])
ax_bottom[0].set_xlabel('Time (s)')
ax_bottom[0].set_ylabel('Neuron #')

# %%
dataset = load_dataset("lorenz", "data")

# %%
print("Train dataset shape:", dataset['train_data'].shape)
print("Val dataset shape:", dataset['val_data'].shape)

# %% [markdown]
# ## Building an autoencoder
import math
import torch
import torch.nn as nn

class _SinusoidalPositionEmb(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.sin(position * div_term)[:, :-1]  # last element is dropped for odd d_model
        self.register_buffer("pe", pe)

    def forward(self, t: int) -> torch.Tensor:
        return self.pe[:t]               # (t, d_model)


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 10_000,
        pos_encoding: str = "learned",   # "learned" | "sin"
    ):
        super().__init__()
        self.channels = channels
        self.max_len = max_len

        # positional encodings ------------------------------------------------
        if pos_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(max_len, channels))
            nn.init.normal_(self.pos_embed, std=0.02)
            self.get_pos = lambda t: self.pos_embed[:t]
        elif pos_encoding == "sin":
            self.sin_pos = _SinusoidalPositionEmb(channels, max_len)
            self.get_pos = self.sin_pos.forward
        else:
            raise ValueError("pos_encoding must be 'learned' or 'sin'")

        def new_layer():
            return nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )

        self.layers = nn.ModuleList([new_layer() for _ in range(num_layers)])

        self.out_proj = nn.Identity()

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        assert c == self.channels, "Mismatch in channel dimension"
        assert t <= self.max_len, "Sequence length exceeds `max_len`"

        x = x + self.get_pos(t).unsqueeze(0)  # add positional info

        # encode
        for layer in self.layers:
            x = layer(x)

        return self.out_proj(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy = torch.randn(4, 128, 64)
model = TransformerAutoencoder(channels=64, num_layers=4, pos_encoding="learned")
model.to(device)
out = model(dummy.to(device))
print(out.shape)          # torch.Size([4, 128, 64])

# %%
# Define a masking pattern
def do_masking(batch: torch.Tensor, mask_ratio: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly mask *mask_ratio* timesteps per trial (span width = 1)."""
    batch_size, num_timesteps = batch.shape[:2]

    width = torch.randint(1, 6, (1, )).item()  # random span length for each sample
    mask = torch.rand(batch_size, num_timesteps) < (mask_ratio / width)

    if width > 1:
        # Convolve to widen the mask
        kernel = torch.ones((width, ), dtype=torch.float32)
        mask = torch.nn.functional.conv1d(
            mask.unsqueeze(1).float(), kernel.unsqueeze(0).unsqueeze(0),
            padding=width // 2
        ).squeeze(1) > 0.5
        if width % 2 == 0:
            mask = mask[:, :-1]

    mask = ~mask  # True where we *keep* the original value

    # Replace some masked tokens with 0 (80%) or random spikes (20%)
    mask_token_ratio = 0.8
    random_token_ratio = 0.25

    replace_zero = (torch.rand_like(mask, dtype=float) < mask_token_ratio) & ~mask
    replace_rand = (torch.rand_like(mask, dtype=float) < random_token_ratio) & ~mask & ~replace_zero

    batch_mean = batch.to(float).mean().item()
    batch = batch.clone()  # avoid in‑place modification
    batch[replace_zero] = 0
    if replace_rand.any():
        rand_values = (torch.rand_like(batch, dtype=float) < batch_mean).to(torch.int)
        batch[replace_rand] = rand_values[replace_rand]

    return batch, mask

#A = (2 * torch.rand(20, 10, 5)).to(int)  # Example batch of shape (batch_size, num_timesteps, num_neurons)
#do_masking(A)

# %% Define a training step
def train_step(net, batch, device, criterion, mask_ratio=0.25):
    spikes, _ = batch
    ground_spikes = spikes.to(device).clone()
    masked_spikes, the_mask = do_masking(spikes, mask_ratio=mask_ratio)
    
    preds = net(masked_spikes.to(device).float())

    loss = criterion(preds[the_mask], ground_spikes[the_mask])
    loss = loss.mean()
    return loss, preds, the_mask

# %%
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

dropout = 0.2
batch_size = 64
num_epochs = 50
trial_len = dataset['train_data'].shape[1]  # number of timesteps in each trial

  # keep small for the tutorial
train_ds = TensorDataset(torch.from_numpy(dataset['train_data']), torch.from_numpy(dataset['train_truth']))
val_ds = TensorDataset(torch.from_numpy(dataset['val_data']), torch.from_numpy(dataset['val_truth']))

n_neurons = dataset['train_data'].shape[2]

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = TransformerAutoencoder(channels=n_neurons, num_layers=2, num_heads=1, max_len=trial_len, dropout=dropout, pos_encoding="sin").to(device)
net = net.to(device)
criterion = nn.PoissonNLLLoss(reduction='none', log_input=True)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

scheduler = StepLR(optimizer, step_size=num_epochs // 2, gamma=0.3)

for epoch in range(num_epochs):
    net.train()
    train_losses = []
    for batch in tqdm(train_loader, desc=f"Train epoch {epoch+1}"):
        optimizer.zero_grad()
        loss, preds, _ = train_step(net, batch, device, criterion, 0.25)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    net.eval()
    val_losses = []
    val_truth_losses = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val epoch {epoch+1}"):
            loss, preds, _ = train_step(net, batch, device, criterion, mask_ratio=1.0)
            val_losses.append(loss.item())

            # Use the correlation coefficient as a validation metric
            _, val_truth = batch
            val_truth_losses.append(
                torch.corrcoef(torch.stack([torch.exp(preds).flatten(), 
                                            val_truth.to(device).flatten()]))[0, 1].item()
            )


    print(
        f"Epoch {epoch+1}: train loss {sum(train_losses)/len(train_losses):.4f}, "
        f"val loss {sum(val_losses)/len(val_losses):.4f}, "
        f"val truth loss {sum(val_truth_losses)/len(val_losses):.4f}"
    )
    scheduler.step()

net.eval()
example_batch = next(iter(val_loader))
true_spikes = example_batch[0].clone()  # Get the spikes from the batch
with torch.no_grad():
    loss, preds, the_mask = train_step(net, example_batch, device, criterion, mask_ratio=1.0)

spikes, ground_truth = example_batch

plt.figure(figsize=(12, 4))
plt.subplot(1, 5, 1)
plt.imshow(true_spikes[0].numpy().T, aspect="auto", origin="lower")
plt.title("Input spikes")
plt.subplot(1, 5, 2)
plt.imshow(spikes[0].numpy().T, aspect="auto", origin="lower")
plt.title("Masked spikes")
plt.subplot(1, 5, 3)
plt.imshow(ground_truth[0].numpy().T, aspect="auto", origin="lower")
plt.title("Ground truth latents")
plt.subplot(1, 5, 4)
plt.imshow(np.exp(preds[0].cpu().numpy()).T, aspect="auto", origin="lower")
plt.title("Model prediction")
plt.subplot(1, 5, 5)
plt.imshow(the_mask.cpu().numpy(), aspect="auto", origin="lower")
plt.title("Mask")
plt.tight_layout()
plt.show()

# %% [markdown]
class TransformerLatentAutoencoder(nn.Module):
    def __init__(
        self,
        channels: int,
        inner_dim: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 10_000,
        pos_encoding: str = "learned",   # "learned" | "sin"
    ):
        super().__init__()
        self.channels = channels
        self.max_len = max_len

        # positional encodings ------------------------------------------------
        if pos_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(max_len, inner_dim))
            nn.init.normal_(self.pos_embed, std=0.02)
            self.get_pos = lambda t: self.pos_embed[:t]
        elif pos_encoding == "sin":
            print(inner_dim, max_len)
            self.sin_pos = _SinusoidalPositionEmb(inner_dim, max_len)
            self.get_pos = self.sin_pos.forward
        elif pos_encoding == "none":
            self.register_buffer("pos_embed", torch.zeros(max_len, inner_dim))
            self.get_pos = lambda t: self.pos_embed[:t, :]
            # no positional encoding, just zeros
        else:
            raise ValueError("pos_encoding must be 'learned' or 'sin'")
        self.linear_embed = nn.Linear(channels, inner_dim)

        def new_layer():
            return nn.TransformerEncoderLayer(
                d_model=inner_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )

        self.layers = nn.ModuleList([new_layer() for _ in range(num_layers)])

        self.out_proj = nn.Linear(inner_dim, channels)

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        assert c == self.channels, "Mismatch in channel dimension"
        assert t <= self.max_len, "Sequence length exceeds `max_len`"

        x = self.linear_embed(x) + self.get_pos(t).unsqueeze(0)

        # encode
        for layer in self.layers:
            x = layer(x)

        return self.out_proj(x)

# %%
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

dropout = 0.2
batch_size = 64
num_epochs = 100
inner_dim = 32
mask_ratio = 0.25
trial_len = dataset['train_data'].shape[1]  # number of timesteps in each trial

  # keep small for the tutorial
train_ds = TensorDataset(torch.from_numpy(dataset['train_data']), 
                         torch.from_numpy(dataset['train_truth']))
val_ds = TensorDataset(torch.from_numpy(dataset['val_data']), 
                       torch.from_numpy(dataset['val_truth']))

n_neurons = dataset['train_data'].shape[2]

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = TransformerLatentAutoencoder(channels=n_neurons, inner_dim=inner_dim, num_layers=2, num_heads=1, max_len=trial_len, dropout=dropout, pos_encoding="sin")
net = net.to(device)
criterion = nn.PoissonNLLLoss(reduction='none', log_input=True)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

scheduler = StepLR(optimizer, step_size=num_epochs // 2, gamma=0.3)

for epoch in range(num_epochs):
    net.train()
    train_losses = []
    for batch in tqdm(train_loader, desc=f"Train epoch {epoch+1}"):
        optimizer.zero_grad()
        loss, preds, _ = train_step(net, batch, device, criterion, mask_ratio)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    net.eval()
    val_losses = []
    val_truth_losses = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val epoch {epoch+1}"):
            loss, preds, _ = train_step(net, batch, device, criterion, mask_ratio=1.0)
            val_losses.append(loss.item())

            # Use the correlation coefficient as a validation metric
            _, val_truth = batch
            val_truth_losses.append(
                torch.corrcoef(torch.stack([torch.exp(preds).flatten(), val_truth.to(device).flatten()]))[0, 1].item() ** 2
            )


    print(
        f"Epoch {epoch+1}: train loss {sum(train_losses)/len(train_losses):.4f}, "
        f"val loss {sum(val_losses)/len(val_losses):.4f}, "
        f"val truth loss {sum(val_truth_losses)/len(val_losses):.4f}"
    )
    scheduler.step()

net.eval()
example_batch = next(iter(val_loader))
true_spikes = example_batch[0].clone()  # Get the spikes from the batch
with torch.no_grad():
    loss, preds, the_mask = train_step(net, example_batch, device, criterion, mask_ratio=1.0)

spikes, ground_truth = example_batch 

plt.figure(figsize=(12, 4))
plt.subplot(1, 5, 1)
plt.imshow(true_spikes[0].numpy().T, aspect="auto", origin="lower")
plt.title("Input spikes")
plt.subplot(1, 5, 2)
plt.imshow(spikes[0].numpy().T, aspect="auto", origin="lower")
plt.title("Masked spikes")
plt.subplot(1, 5, 3)
plt.imshow(ground_truth[0].numpy().T, aspect="auto", origin="lower")
plt.title("Ground truth latents")
plt.subplot(1, 5, 4)
plt.imshow(np.exp(preds[0].cpu().numpy()).T, aspect="auto", origin="lower")
plt.title("Model prediction")
plt.subplot(1, 5, 5)
plt.imshow(the_mask.cpu().numpy(), aspect="auto", origin="lower")
plt.title("Mask")
plt.tight_layout()
plt.show()
# %%
