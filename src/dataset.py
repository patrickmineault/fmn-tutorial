#!/usr/bin/env python3
# Author: Joel Ye
import os.path as osp
from enum import Enum
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from munch import munchify
from torch.utils import data


# Verbatim from LFADS_TF2
def merge_train_valid(train_data, valid_data, train_ixs, valid_ixs):
    """Merges training and validation numpy arrays using indices.

    This function merges training and validation numpy arrays
    in the appropriate order using arrays of their indices. The
    lengths of the indices must be the same as the first dimension
    of the corresponding data.

    Parameters
    ----------
    train_data : np.ndarray
        An N-dimensional numpy array of training data with
        first dimension T.
    valid_data : np.ndarray
        An N-dimensional numpy array of validation data with
        first dimension V.
    train_ixs : np.ndarray
        A 1-D numpy array of training indices with length T.
    valid_ixs : np.ndarray
        A 1-D numpy array of validation indices with length V.

    Returns
    -------
    np.ndarray
        An N-dimensional numpy array with dimension T + V.

    """

    if (
        train_data.shape[0] == train_ixs.shape[0]
        and valid_data.shape[0] == valid_ixs.shape[0]
    ):
        # if the indices match up, then we can use them to merge
        data = np.full_like(np.concatenate([train_data, valid_data]), np.nan)
        if min(min(train_ixs), min(valid_ixs)) > 0:
            # we've got matlab data...
            train_ixs -= 1
            valid_ixs -= 1
        data[train_ixs.astype(int)] = train_data
        data[valid_ixs.astype(int)] = valid_data
    else:
        # if the indices do not match, train and
        # valid data may be the same (e.g. for priors)
        if np.all(train_data == valid_data):
            data = train_data
        else:
            raise ValueError(
                "shape mismatch: "
                f"Index shape {train_ixs.shape} does not "
                f"match the data shape {train_data.shape}."
            )
    return data


class DATASET_MODES(Enum):
    train = "train"
    val = "val"
    test = "test"
    trainval = "trainval"


def merge_config(base: dict, delta: dict):
    for k, v in delta.items():
        if isinstance(v, dict):
            merge_config(base[k], delta[k])
        else:
            base[k] = v
    return base


class SpikesDataset(data.Dataset):
    r"""
    Dataset for single file of spike times (loads into memory)
    Lorenz data is NxTxH (H being number of neurons) - we load to T x N x H
    # ! Note that codepath for forward but not heldout neurons is not tested and likely broken
    """

    def __init__(self, config_filename, mode=DATASET_MODES.train, logger=None):
        r"""
        args:
            config: dataset config
            filename: excluding path
            mode: used to extract the right indices from LFADS h5 data
        """
        super().__init__()
        with open(config_filename) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        with open(Path(config_filename).parent / "base.yaml") as f:
            base_config = yaml.load(f, Loader=yaml.FullLoader)

        config = munchify(merge_config(base_config, config))

        self.logger = logger
        self.config = config.DATA
        self.use_lograte = config.MODEL.LOGRATE
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.datapath = (
            Path(config_filename).parent.parent / "h5" / config.DATA.TRAIN_FILENAME
        )

        self.has_rates = False
        self.has_heldout = False
        self.has_forward = False

        spikes, rates, heldout_spikes, forward_spikes = self.get_data_from_h5(
            mode, self.datapath
        )

        spikes = torch.tensor(spikes).long()
        if rates is not None:
            rates = torch.tensor(rates)
        if heldout_spikes is not None:
            self.has_heldout = True
            heldout_spikes = torch.tensor(heldout_spikes).long()
        if forward_spikes is not None and not config.DATA.IGNORE_FORWARD:
            self.has_forward = True
            forward_spikes = torch.tensor(forward_spikes).long()
        else:
            forward_spikes = None

        self.num_trials, _, self.num_neurons = spikes.size()
        self.full_length = config.MODEL.TRIAL_LENGTH <= 0
        self.trial_length = (
            spikes.size(1) if self.full_length else config.MODEL.TRIAL_LENGTH
        )
        if self.has_heldout:
            self.num_neurons += heldout_spikes.size(-1)
        if self.has_forward:
            self.trial_length += forward_spikes.size(1)
        self.spikes = self.batchify(spikes)
        # Fake rates so we can skip None checks everywhere. Use `self.has_rates` when desired
        self.rates = (
            self.batchify(rates) if self.has_rates else torch.zeros_like(spikes)
        )
        # * else condition below is not precisely correctly shaped as correct shape isn't used
        self.heldout_spikes = (
            self.batchify(heldout_spikes)
            if self.has_heldout
            else torch.zeros_like(spikes)
        )
        self.forward_spikes = (
            self.batchify(forward_spikes)
            if self.has_forward
            else torch.zeros_like(spikes)
        )

        if config.DATA.OVERFIT_TEST:
            if self.logger is not None:
                self.logger.warning("Overfitting..")
            self.spikes = self.spikes[:2]
            self.rates = self.rates[:2]
            self.num_trials = 2
        elif (
            hasattr(config.DATA, "RANDOM_SUBSET_TRIALS")
            and config.DATA.RANDOM_SUBSET_TRIALS < 1.0
            and mode == DATASET_MODES.train
        ):
            if self.logger is not None:
                self.logger.warning(
                    f"!!!!! Training on {config.DATA.RANDOM_SUBSET_TRIALS} of the data with seed {config.SEED}."
                )
            reduced = int(self.num_trials * config.DATA.RANDOM_SUBSET_TRIALS)
            torch.random.manual_seed(config.SEED)
            random_subset = torch.randperm(self.num_trials)[:reduced]
            self.num_trials = reduced
            self.spikes = self.spikes[random_subset]
            self.rates = self.rates[random_subset]

    def batchify(self, x):
        r"""
        Chops data into uniform sizes as configured by trial_length.

        Returns:
            x reshaped as num_samples x trial_length x neurons
        """
        if self.full_length:
            return x
        trial_time = x.size(1)
        samples_per_trial = trial_time // self.trial_length
        if trial_time % self.trial_length != 0:
            if self.logger is not None:
                self.logger.debug(
                    f"Trimming dangling trial info. Data trial length {trial_time} \
                is not divisible by asked length {self.trial_length})"
                )
        x = x.narrow(1, 0, samples_per_trial * self.trial_length)

        # ! P sure this can be a view op
        # num_samples x trial_length x neurons
        return torch.cat(torch.split(x, self.trial_length, dim=1), dim=0)

    def get_num_neurons(self):
        return self.num_neurons

    def __len__(self):
        return self.spikes.size(0)

    def __getitem__(self, index):
        r"""
        Return spikes and rates, shaped T x N (num_neurons)
        """
        spikes = self.spikes[index].T.to(torch.long)
        rates = (
            None if self.rates is None else self.rates[index].T.to(torch.float32).exp()
        )
        heldout_spikes = (
            None
            if self.heldout_spikes is None
            else self.heldout_spikes[index].T.to(torch.long)
        )
        forward_spikes = (
            None
            if self.forward_spikes is None
            else self.forward_spikes[index].T.to(torch.long)
        )

        return (spikes, rates, heldout_spikes, forward_spikes)

    def get_dataset(self):
        return self.spikes, self.rates, self.heldout_spikes, self.forward_spikes

    def get_max_spikes(self):
        return self.spikes.max().item()

    def get_num_batches(self):
        return self.spikes.size(0) // self.batch_size

    def clip_spikes(self, max_val):
        self.spikes = torch.clamp(self.spikes, max=max_val)

    def get_data_from_h5(self, mode, filepath):
        r"""
        returns:
            spikes
            rates (None if not available)
            held out spikes (for cosmoothing, None if not available)
        * Note, rates and held out spikes codepaths conflict
        """
        NLB_KEY = "spikes"  # curiously, old code thought NLB data keys came as "train_data_heldin" and not "train_spikes_heldin"
        NLB_KEY_ALT = "data"

        with h5py.File(filepath, "r") as h5file:
            h5dict = {key: h5file[key][()] for key in h5file.keys()}
            if f"eval_{NLB_KEY}_heldin" not in h5dict:  # double check
                if f"eval_{NLB_KEY_ALT}_heldin" in h5dict:
                    NLB_KEY = NLB_KEY_ALT
            if (
                f"eval_{NLB_KEY}_heldin" in h5dict
            ):  # NLB data, presumes both heldout neurons and time are available
                get_key = lambda key: h5dict[key].astype(np.float32)
                train_data = get_key(f"train_{NLB_KEY}_heldin")
                train_data_fp = get_key(f"train_{NLB_KEY}_heldin_forward")
                train_data_heldout_fp = get_key(f"train_{NLB_KEY}_heldout_forward")
                train_data_all_fp = np.concatenate(
                    [train_data_fp, train_data_heldout_fp], -1
                )
                valid_data = get_key(f"eval_{NLB_KEY}_heldin")
                train_data_heldout = get_key(f"train_{NLB_KEY}_heldout")
                if f"eval_{NLB_KEY}_heldout" in h5dict:
                    valid_data_heldout = get_key(f"eval_{NLB_KEY}_heldout")
                else:
                    self.logger.warn(
                        "Substituting zero array for heldout neurons. Only done for evaluating models locally, i.e. will disrupt training due to early stopping."
                    )
                    valid_data_heldout = np.zeros(
                        (
                            valid_data.shape[0],
                            valid_data.shape[1],
                            train_data_heldout.shape[2],
                        ),
                        dtype=np.float32,
                    )
                if f"eval_{NLB_KEY}_heldin_forward" in h5dict:
                    valid_data_fp = get_key(f"eval_{NLB_KEY}_heldin_forward")
                    valid_data_heldout_fp = get_key(f"eval_{NLB_KEY}_heldout_forward")
                    valid_data_all_fp = np.concatenate(
                        [valid_data_fp, valid_data_heldout_fp], -1
                    )
                else:
                    self.logger.warn(
                        "Substituting zero array for heldout forward neurons. Only done for evaluating models locally, i.e. will disrupt training due to early stopping."
                    )
                    valid_data_all_fp = np.zeros(
                        (
                            valid_data.shape[0],
                            train_data_fp.shape[1],
                            valid_data.shape[2] + valid_data_heldout.shape[2],
                        ),
                        dtype=np.float32,
                    )

                # NLB data does not have ground truth rates
                if mode == DATASET_MODES.train:
                    return train_data, None, train_data_heldout, train_data_all_fp
                elif mode == DATASET_MODES.val:
                    return valid_data, None, valid_data_heldout, valid_data_all_fp
            train_data = h5dict["train_data"].astype(np.float32).squeeze()
            valid_data = h5dict["valid_data"].astype(np.float32).squeeze()
            train_rates = None
            valid_rates = None
            if (
                "train_truth" and "valid_truth" in h5dict
            ):  # original LFADS-type datasets
                self.has_rates = True
                train_rates = h5dict["train_truth"].astype(np.float32)
                valid_rates = h5dict["valid_truth"].astype(np.float32)
                train_rates = train_rates / h5dict["conversion_factor"]
                valid_rates = valid_rates / h5dict["conversion_factor"]
                if self.use_lograte:
                    train_rates = torch.log(
                        torch.tensor(train_rates) + self.config.LOG_EPSILON
                    )
                    valid_rates = torch.log(
                        torch.tensor(valid_rates) + self.config.LOG_EPSILON
                    )
        if mode == DATASET_MODES.train:
            return train_data, train_rates, None, None
        elif mode == DATASET_MODES.val:
            return valid_data, valid_rates, None, None
        elif mode == DATASET_MODES.trainval:
            # merge training and validation data
            if "train_inds" in h5dict and "valid_inds" in h5dict:
                # if there are index labels, use them to reassemble full data
                train_inds = h5dict["train_inds"].squeeze()
                valid_inds = h5dict["valid_inds"].squeeze()
                file_data = merge_train_valid(
                    train_data, valid_data, train_inds, valid_inds
                )
                if self.has_rates:
                    merged_rates = merge_train_valid(
                        train_rates, valid_rates, train_inds, valid_inds
                    )
            else:
                if self.logger is not None:
                    self.logger.info(
                        "No indices found for merge. "
                        "Concatenating training and validation samples."
                    )
                file_data = np.concatenate([train_data, valid_data], axis=0)
                if self.has_rates:
                    merged_rates = np.concatenate([train_rates, valid_rates], axis=0)
            return file_data, merged_rates if self.has_rates else None, None, None
        else:  # test unsupported
            return None, None, None, None
