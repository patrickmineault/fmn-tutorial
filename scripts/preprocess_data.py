import h5py
from pathlib import Path
from nlb_tools.nwb_interface import NWBDataset
import numpy as np
import pickle
import subprocess
from typing import Dict
from nlb_tools.make_tensors import make_train_input_tensors, _prep_mask
from scipy.ndimage import gaussian_filter1d

_DATASET_MAP: Dict[str, str] = {
    "mc_maze": "000128",
    "mc_maze_small": "000140",
    "mc_maze_medium": "000139",
    "mc_maze_large": "000138",
}


def download_datasets():
    """Download required datasets for preprocessing."""

    # Create data directory if it doesn't exist
    data_dir = Path("../data/h5")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download Lorenz dataset from GitHub
    lorenz_path = data_dir / "lfads_lorenz.h5"
    if not lorenz_path.exists():
        print("Downloading Lorenz dataset...")
        try:
            subprocess.run(
                [
                    "wget",
                    "-P",
                    str(data_dir),
                    "https://github.com/snel-repo/neural-data-transformers/raw/refs/heads/master/data/lfads_lorenz.h5",
                ],
                check=True,
            )
            print("✓ Lorenz dataset downloaded successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download Lorenz dataset: {e}")
            raise
        except FileNotFoundError:
            print("wget not found. Please install wget or download manually.")
            raise
    else:
        print("✓ Lorenz dataset already exists")

    # Download DANDI datasets
    for name, dandi_id in _DATASET_MAP.items():
        dataset_path = data_dir / dandi_id
        if not dataset_path.exists() or not any(dataset_path.iterdir()):
            print(f"Downloading {name} dataset (DANDI:{dandi_id})...")
            try:
                subprocess.run(
                    ["dandi", "download", f"DANDI:{dandi_id}", "-o", str(data_dir)],
                    check=True,
                )
                print(f"✓ {name} dataset downloaded successfully")
            except subprocess.CalledProcessError as e:
                print(f"Failed to download {name} dataset: {e}")
                raise
            except FileNotFoundError:
                print("dandi not found. Please install dandi: pip install dandi")
                raise
        else:
            print(f"✓ {name} dataset already exists")

    print("All datasets downloaded successfully!")


def _load_lorenz(path: Path) -> Dict[str, np.ndarray]:
    h5_path = path / "lfads_lorenz.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Expected Lorenz data at {h5_path}. Download it first."
        )
    with h5py.File(h5_path, "r") as f:
        data = {k.replace("valid_", "val_"): f[k][:] for k in f.keys()}
    data["val_truth"] = (data["val_truth"] * 20).astype(
        np.float32
    )  # This makes spike rates comparable to spikes
    data["train_data"] = data["train_data"].astype(np.uint8)
    data["val_data"] = data["val_data"].astype(np.uint8)
    del data[
        "train_truth"
    ]  # To save space and for the notebook to be faster to download
    return data  # keys: train_data, val_data, (optionally *_truth)


def preprocess_lfads_lorenz():
    dataset_name = "lfads_lorenz"
    dataset = _load_lorenz(Path("../data/h5/"))
    with open("../data/" + dataset_name + "_data.pkl", "wb") as f:
        pickle.dump(dataset, f)


def preprocess_mc_maze():
    """Preprocess the various `mc_maze` datasets to extract what we need for training models.
    It's possible to do this on the fly, however, it takes quite a bit of time. Rather than doing that,
    we preprocess the data and save it to disk.

    The preprocessing is inspired by the one proposed in the NDT paper. In particular, to derive a
    "ground truth" for what the traces should be, we use the full mc_maze dataset, average by trial type,
    and the smooth the average trace to get the estimated spike rate.
    """

    for dataset_name, dataset_id in _DATASET_MAP.items():
        fpath = f"../data/h5/{dataset_id}/sub-Jenkins"
        dataset = NWBDataset(fpath=fpath)
        dataset.resample(10)  # Bin at 10 ms resolution

        # Use the full mc_maze dataset to derive the ground truth for trial types
        trial_types = dataset.trial_info[
            ["trial_type", "trial_version", "maze_id"]
        ].drop_duplicates()
        trial_types["trial_unique_type"] = np.arange(len(trial_types)) - 1

        dataset.trial_info = dataset.trial_info.merge(
            trial_types, on=["trial_type", "trial_version", "maze_id"], how="left"
        )

        tr = make_train_input_tensors(
            dataset, dataset_name, "train", save_file=False, include_behavior=True
        )
        train_data = np.concatenate(
            [tr["train_spikes_heldin"], tr["train_spikes_heldout"]], axis=2
        )

        va = make_train_input_tensors(
            dataset, dataset_name, "val", save_file=False, include_behavior=True
        )
        val_data = np.concatenate(
            [va["train_spikes_heldin"], va["train_spikes_heldout"]], axis=2
        )

        trial_mask_train = _prep_mask(dataset, "train")
        trial_unique_type_train = dataset.trial_info["trial_unique_type"][
            trial_mask_train
        ]

        trial_mask_val = _prep_mask(dataset, "val")
        trial_unique_type_val = dataset.trial_info["trial_unique_type"][trial_mask_val]

        n_trial_types = len(np.unique(trial_unique_type_train))
        _, time, neurons = train_data.shape
        trial_avg = np.zeros((n_trial_types, time, neurons))
        counts = np.zeros((n_trial_types, 1, 1))  # for broadcasting

        # Accumulate sums and counts by trial type
        np.add.at(trial_avg, trial_unique_type_train, train_data)
        np.add.at(trial_avg, trial_unique_type_val, val_data)
        np.add.at(counts, trial_unique_type_train, 1)
        np.add.at(counts, trial_unique_type_val, 1)

        sigma = 3  # 30 ms, as in the paper
        counts[counts == 0] = 1  # Avoid division by zero
        trial_avg /= counts
        trial_avg_smooth = gaussian_filter1d(trial_avg, sigma=sigma, axis=1)

        # Create a synthetic ground truth by smoothing the raw data and aggregating by trial type
        val_data_truth = trial_avg_smooth[trial_unique_type_val]

        dataset = {
            "train_data": train_data.astype(np.uint8),
            "val_data": val_data.astype(np.uint8),
            "train_behavior": (tr["train_behavior"] / 1000.0).astype(
                np.float32
            ),  # In m/s
            "val_behavior": (va["train_behavior"] / 1000.0).astype(
                np.float32
            ),  # In m/s
            "spike_sources": list(dataset.data.spikes.columns)
            + list(dataset.data.heldout_spikes.columns),
            "val_truth": val_data_truth.astype(np.float32),
        }

        with open("../data/" + dataset_name + "_data.pkl", "wb") as f:
            pickle.dump(dataset, f)


def main():
    """Check if datasets exist"""
    download_datasets()
    preprocess_lfads_lorenz()
    preprocess_mc_maze()


if __name__ == "__main__":
    main()
