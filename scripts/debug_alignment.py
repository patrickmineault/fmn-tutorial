#!/usr/bin/env python
"""
Debug script to test temporal alignment between neural spikes and behavior.

This script fits a simple linear decoder at various time delays to identify
potential misalignment between neural and behavioral data.

Example
-------
python debug_alignment.py --dataset mc_maze_medium --data-root ./data/h5

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import dataset loading functions
from train_tutorial_1 import load_dataset, _DATASET_MAP


def fit_decoder_at_delay(
    train_spikes: np.ndarray,
    train_behavior: np.ndarray,
    val_spikes: np.ndarray,
    val_behavior: np.ndarray,
    delay: int,
) -> Tuple[float, float, np.ndarray]:
    """
    Fit a linear decoder with a specific time delay using sklearn.
    
    Args:
        train_spikes: (n_trials, n_timesteps, n_neurons)
        train_behavior: (n_trials, n_timesteps, n_behaviors)
        delay: Time delay in bins (positive = behavior leads spikes)
    
    Returns:
        train_r2: Average R² on training set
        val_r2: Average R² on validation set
        val_r2_per_dim: R² for each behavior dimension
    """
    # Apply circular shift to behavior to test different delays
    # Positive delay means behavior is shifted forward (behavior leads spikes)
    train_behavior_shifted = np.roll(train_behavior, shift=delay, axis=1)
    val_behavior_shifted = np.roll(val_behavior, shift=delay, axis=1)
    
    # Flatten trials and timesteps
    n_train_samples = train_spikes.shape[0] * train_spikes.shape[1]
    n_val_samples = val_spikes.shape[0] * val_spikes.shape[1]
    
    train_spikes_flat = train_spikes.reshape(n_train_samples, -1)
    train_behavior_flat = train_behavior_shifted.reshape(n_train_samples, -1)
    val_spikes_flat = val_spikes.reshape(n_val_samples, -1)
    val_behavior_flat = val_behavior_shifted.reshape(n_val_samples, -1)
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(train_spikes_flat, train_behavior_flat)
    
    # Predict
    train_pred = model.predict(train_spikes_flat)
    val_pred = model.predict(val_spikes_flat)
    
    # Compute R² scores
    # Overall R²
    train_r2 = r2_score(train_behavior_flat, train_pred)
    val_r2 = r2_score(val_behavior_flat, val_pred)
    
    # Per-dimension R²
    n_behaviors = train_behavior.shape[-1]
    val_r2_per_dim = np.zeros(n_behaviors)
    for i in range(n_behaviors):
        val_r2_per_dim[i] = r2_score(val_behavior_flat[:, i], val_pred[:, i])
    
    return train_r2, val_r2, val_r2_per_dim


def main():
    parser = argparse.ArgumentParser(description="Debug neural-behavior alignment")
    parser.add_argument("--dataset", default="mc_maze_medium",
                       choices=["lorenz", *_DATASET_MAP.keys()],
                       help="Dataset name.")
    parser.add_argument("--data-root", default="data/h5",
                       help="Path containing dataset files.")
    parser.add_argument("--max-delay", type=int, default=50,
                       help="Maximum delay to test (in both directions).")
    parser.add_argument("--save-plot", type=str, default="alignment_debug.png",
                       help="Filename for saving the plot.")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset, args.data_root)
    
    train_spikes = data["train_data"]
    val_spikes = data["val_data"]
    
    try:
        train_behavior = data["train_behavior"]
        val_behavior = data["val_behavior"]
        n_behaviors = train_behavior.shape[-1]
        print(f"Behavior dimensions: {n_behaviors}")
    except KeyError:
        raise ValueError(f"Dataset {args.dataset} does not contain behavior data!")
    
    print(f"Data shapes:")
    print(f"  Train spikes: {train_spikes.shape}")
    print(f"  Train behavior: {train_behavior.shape}")
    print(f"  Val spikes: {val_spikes.shape}")
    print(f"  Val behavior: {val_behavior.shape}")
    
    # Test range of delays
    delays = list(range(-args.max_delay, args.max_delay + 1, 1))
    train_r2s = []
    val_r2s = []
    val_r2s_per_dim = []
    
    print(f"\nTesting {len(delays)} different delays...")
    for delay in tqdm(delays):
        train_r2, val_r2, val_r2_dims = fit_decoder_at_delay(
            train_spikes, train_behavior,
            val_spikes, val_behavior,
            delay
        )
        train_r2s.append(train_r2)
        val_r2s.append(val_r2)
        val_r2s_per_dim.append(val_r2_dims)
    
    # Convert to arrays
    train_r2s = np.array(train_r2s)
    val_r2s = np.array(val_r2s)
    val_r2s_per_dim = np.array(val_r2s_per_dim)
    
    # Find optimal delay
    best_idx = np.argmax(val_r2s)
    best_delay = delays[best_idx]
    best_r2 = val_r2s[best_idx]
    
    print(f"\nResults:")
    print(f"Best delay: {best_delay} bins")
    print(f"Best validation R²: {best_r2:.4f}")
    print(f"Training R² at best delay: {train_r2s[best_idx]:.4f}")
    print(f"Per-dimension R² at best delay: {val_r2s_per_dim[best_idx]}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot overall R²
    ax1.plot(delays, train_r2s, 'b-', label='Train R²', linewidth=2)
    ax1.plot(delays, val_r2s, 'r-', label='Val R²', linewidth=2)
    ax1.axvline(x=best_delay, color='g', linestyle='--', 
                label=f'Best delay = {best_delay}')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Delay (bins)')
    ax1.set_ylabel('R²')
    ax1.set_title(f'R² vs Time Delay - {args.dataset}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.0)
    
    # Plot per-dimension R²
    for i in range(n_behaviors):
        ax2.plot(delays, val_r2s_per_dim[:, i], label=f'Behavior dim {i}', linewidth=2)
    ax2.axvline(x=best_delay, color='g', linestyle='--', 
                label=f'Best delay = {best_delay}')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Delay (bins)')
    ax2.set_ylabel('Validation R²')
    ax2.set_title('Per-dimension R² vs Time Delay')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.0)
    
    plt.tight_layout()
    plt.savefig(args.save_plot, dpi=150)
    print(f"\nPlot saved to {args.save_plot}")
    
    # Additional analysis
    print("\nAdditional analysis:")
    print(f"R² at zero delay: Train={train_r2s[args.max_delay]:.4f}, Val={val_r2s[args.max_delay]:.4f}")
    
    # Find delays where R² > 0.5
    good_delays = np.where(val_r2s > 0.5)[0]
    if len(good_delays) > 0:
        delay_range = delays[good_delays[0]], delays[good_delays[-1]]
        print(f"Delay range with R² > 0.5: {delay_range[0]} to {delay_range[1]} bins")
    
    # Save results to file
    results = {
        'delays': delays,
        'train_r2s': train_r2s,
        'val_r2s': val_r2s,
        'val_r2s_per_dim': val_r2s_per_dim,
        'best_delay': best_delay,
        'best_r2': best_r2
    }
    np.savez(f'alignment_results_{args.dataset}.npz', **results)
    print(f"Results saved to alignment_results_{args.dataset}.npz")


if __name__ == "__main__":
    main()