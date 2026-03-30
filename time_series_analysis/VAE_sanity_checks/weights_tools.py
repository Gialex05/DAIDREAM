"""
weights_tools.py

Utility functions for saving, loading, and visualising the trainable weights
of the VAE model (vae_model.py) across different training phases and seeds.

Designed to support convergence studies where multiple random initialisations
are compared before and after training.

Exported functions
------------------
save_weights_npz          : save encoder and decoder weights to a .npz archive.
plot_violins              : violin plots of weight distributions across seeds.
plot_histograms           : histogram grid of weight distributions across seeds.
plot_violin_comparison    : side-by-side violins for two training phases, single seed.
plot_histograms_comparison: side-by-side histograms for two training phases, single seed.
plot_histograms_deltas    : histogram grid of weight differences between two phases.
"""

import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def save_weights_npz(vae, dataset_name, phase, seed, save_dir="./weights", overwrite=False):
    """
    Save all trainable weights (kernels and biases) of the VAE encoder and decoder
    into a single .npz file. Keys follow the pattern 'submodel__layername__kernel/bias'.

    Parameters
    ----------
    vae             : VAE
        Trained or initialized VAE instance.
    phase : str
        Label appended to the filename to distinguish pre-training from trained model. E.g. 'init' or 'trained'.
    seed            : int
        Seed used for this run, included in the filename.
    dataset_name    : str
        Name of the dataset, used as the base of the filename.
    save_dir        : str, optional (default='.')
        Directory where the .npz file will be saved.
    overwrite       : bool, optional (default=False)
        If True, overwrites an existing file with the same name.
        If False, raises a FileExistsError if the file already exists.

    Returns
    -------
    str
        Full path of the saved .npz file.

    Raises
    ------
    FileExistsError
        If a file with the same name already exists and overwrite=False.
    """
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{dataset_name}_{seed}_{phase}.npz")

    if os.path.exists(output_path):
        if overwrite:
            print(f"  [warning] overwriting existing file: {output_path}")
        else:
            raise FileExistsError(
                f"File already exists: '{output_path}'.\n"
                f"Use overwrite=True to overwrite it."
            )

    arrays    = {}
    submodels = {"encoder": vae.encoder, "decoder": vae.decoder}

    for submodel_name, submodel in submodels.items():
        for layer in submodel.layers:
            weights = layer.get_weights()
            if not weights:
                continue
            w_types = ["kernel", "bias"] + [f"w{k}" for k in range(2, len(weights))]
            for w_type, w_array in zip(w_types, weights):
                key = f"{submodel_name}__{layer.name}__{w_type}"
                arrays[key] = w_array

    np.savez(output_path, **arrays)

    print(f"\n[npz saved] → {output_path}  \nkeys: {list(arrays.keys())}\n")
    return output_path


def weights_to_longform(data_dict):
    """
    Convert a dictionary of weight arrays into a long-form DataFrame suitable for seaborn plotting.

    Parameters
    ----------
    data_dict : dict { int : np.ndarray }
        Dictionary mapping each seed (int) to a flattened 1D array of weight values.

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with two columns:
        - 'seed'  : str, the seed identifier associated with each weight value.
        - 'value' : float, a single weight value.
    """
    rows = []
    for seed, values in data_dict.items():
        for v in values:
            rows.append({"seed": str(seed), "value": v})
    return pd.DataFrame(rows)


def plot_violins(dataset_name, seeds, phase, submodel, layer, weight_type,
                 success=None, save_dir="./weights", out_dir="./plots"):
    """
    Plot a violins chart comparing the weight distribution of a given layer across different seeds.
    Each violin represents the distribution of all weights (flattened) for one seed.
    If an boolean array indicating successful and failing runs is given, the violins
    are colored dark green for successful training runs and dark red for failed ones.
    The plot is saved as a .png file in out_dir.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset, used to locate .npz files and label the plot.
    seeds        : array-like of int
        Seeds used during training, one violin is drawn per seed.
    phase        : str
        Training phase to load: 'init' (before training) or 'trained' (after training).
    submodel     : str
        Submodel to which the layer belongs: 'encoder' or 'decoder'.
    layer        : str
        Name of the layer whose weights are plotted. E.g. 'conv1', 'z_mean', 'dense_decoded'.
    weight_type  : str
        Type of weight array to load: 'kernel' or 'bias'.
    success      : array-like of bool or None, optional (default=None)
        Boolean array of the same length as seeds. True indicates a successful training run,
        False a failed one. If None, the default seaborn 'muted' palette is used.
    save_dir     : str, optional (default='./weights')
        Directory where the .npz weight files are stored.
    out_dir      : str, optional (default='./plots')
        Directory where the output .png plot is saved.

    Returns
    -------
    None
    """
    os.makedirs(out_dir, exist_ok=True)

    # Built the file name
    data = {}
    for seed in seeds:
        filepath = os.path.join(save_dir, f"{dataset_name}_{seed}_{phase}.npz")
        if not os.path.exists(filepath):
            print(f"  [warning] file not found: {filepath}")
            continue
        npz  = np.load(filepath)
        key  = f"{submodel}__{layer}__{weight_type}"
        if key not in npz:
            print(f"  [warning] key '{key}' not in {filepath}")
            continue
        data[seed] = npz[key].flatten()

    if not data:
        print("No data loaded.")
        return

    # Build palette seed → color based on success
    if success is not None:
        assert len(success) == len(seeds), f"Success and seeds have different lenghts: {len(success)} vs {len(seeds)}"    
        success_map = {str(seed): ok for seed, ok in zip(seeds, success)}
        palette = {
            str(seed): "forestgreen" if success_map[str(seed)] else "firebrick"
            for seed in data.keys()
        }
    else:
        palette = "muted"
    
    df = weights_to_longform(data)

    fig, ax = plt.subplots(figsize=(min(9, len(seeds) * 1.5), 5))

    sns.violinplot(
        data=df, x="seed", y="value",
        inner="box",
        linewidth=2,
        palette=palette,
        hue="seed", legend=False,
        ax=ax
    )

    # Linea a y=0 come riferimento
    ax.axhline(0, color="orange", linestyle="--", linewidth=2, alpha=0.8)

    ax.set_title(f"Parameters' distributions for {submodel} › {layer} › {weight_type}\n"
                 f"Dataset: {dataset_name}  |  Phase: {phase}", fontsize=11)
    ax.set_xlabel("seed")
    ax.set_ylabel("parameter values")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"violin_{dataset_name}_{phase}_{submodel}_{layer}_{weight_type}.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"  [plot saved] → {out_path}")
    
    
def plot_histograms(dataset_name, seeds, phase, submodel, layer, weight_type, 
                    success=None, bins=50, save_dir="./weights", out_dir="./plots"):
    """
    Plot a grid of histograms comparing the weight distribution of a given layer across different seeds.
    Each panel corresponds to one seed and shows the full distribution of all weights (flattened),
    with vertical lines marking the mean and median. Panels are arranged in a 2-column grid
    with a shared x-axis to make cross-seed comparisons straightforward.
    The plot is saved as a .png file in out_dir.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset, used to locate .npz files and label the plot.
    seeds        : array-like of int
        Seeds used during training, one histogram panel is drawn per seed.
    phase        : str
        Training phase to load: 'init' (before training) or 'trained' (after training).
    submodel     : str
        Submodel to which the layer belongs: 'encoder' or 'decoder'.
    layer        : str
        Name of the layer whose weights are plotted. E.g. 'conv1', 'z_mean', 'dense_decoded'.
    weight_type  : str
        Type of weight array to load: 'kernel' or 'bias'.
    success      : array-like of bool or None, optional (default=None)
        Boolean array of the same length as seeds. True indicates a successful training run,
        False a failed one. Colors each panel dark green or dark red accordingly,
        and adds a success/failure label to each panel title. If None, all panels are steel blue.
    bins         : int, optional (default=50)
        Number of bins for each histogram.
    save_dir     : str, optional (default='./weights')
        Directory where the .npz weight files are stored.
    out_dir      : str, optional (default='./plots')
        Directory where the output .png plot is saved.

    Returns
    -------
    None
    """
    os.makedirs(out_dir, exist_ok=True)

    data = {}
    for seed in seeds:
        filepath = os.path.join(save_dir, f"{dataset_name}_{seed}_{phase}.npz")
        if not os.path.exists(filepath):
            print(f"  [warning] file not found: {filepath}")
            continue
        npz = np.load(filepath)
        key = f"{submodel}__{layer}__{weight_type}"
        if key not in npz:
            print(f"  [warning] key '{key}' not in {filepath}")
            continue
        data[seed] = npz[key].flatten()

    if not data:
        print("No dato loaded.")
        return

    # Build map seed → color based on success    
    if success is not None:
        assert len(success) == len(seeds), f"Success and seeds have different lenghts: {len(success)} vs {len(seeds)}"
        success_map = {str(seed): ok for seed, ok in zip(seeds, success)}
        color_map = {
            str(seed): "forestgreen" if success_map[str(seed)] else "firebrick"
            for seed in data.keys()
        }
    else:
        color_map = {str(seed): "royalblue" for seed in data.keys()}
        
    n     = len(data)
    ncols = min(n, 2) 
    nrows = math.ceil(n / ncols)

    # Find common range for all the histograms
    all_values = np.concatenate(list(data.values()))
    x_min, x_max = all_values.min(), all_values.max()
    x_margin = (x_max - x_min) * 0.05
    shared_range = (x_min - x_margin, x_max + x_margin)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 3.5))
    axes = np.array(axes).flatten()

    for ax, (seed, values) in zip(axes, data.items()):
        ax.hist(values, bins=bins, range=shared_range,
                color=color_map[str(seed)], 
                edgecolor="white", linewidth=0.4, alpha=0.85)
        ax.axvline(0,            color="gray",   linestyle="--", linewidth=1)
        ax.axvline(values.mean(), color="yellow", linestyle="-",  linewidth=2,
                   label=f"mean={values.mean():.4f}")
        ax.axvline(np.median(values), color="orange", linestyle="-.", linewidth=2,
                   label=f"median={np.median(values):.4f}")
        ax.set_ylabel("counts")
        ax.legend(fontsize=10)
        
        if success is not None:
            outcome = "(success)" if success_map[str(seed)] else "(failure)"
            ax.set_title(f"seed {seed} {outcome}", fontsize=12)
        else:
            ax.set_title(f"seed {seed}", fontsize=12)
        
    # Hid last panel when n is odd
    if n%2 == 1:
        for ax in axes[n:]:
            ax.set_visible(False)        

    fig.suptitle(f" Parameters' distributions for {submodel} › {layer} › {weight_type}\n"
                 f"Dataset: {dataset_name}  |  Phase: {phase}",
                 fontsize=12, y=1.01)
    fig.supxlabel("parameter values")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"hist_{dataset_name}_{phase}_{submodel}_{layer}_{weight_type}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [plot saved] → {out_path}")
    
    
def plot_violin_comparison(dataset_name, seed, submodel, layer, weight_type,
                           phases=("init", "trained"), save_dir="./weights", out_dir="./plots"):
    """
    Plot two violin charts side by side on the same canvas, comparing the weight
    distribution of a given layer for two different training phases of the same seed.
    The plot is saved as a .png file in out_dir.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset, used to locate .npz files and label the plot.
    seed         : int
        Seed of the run to compare across phases.
    submodel     : str
        Submodel to which the layer belongs: 'encoder' or 'decoder'.
    layer        : str
        Name of the layer whose weights are plotted. E.g. 'conv1', 'z_mean', 'dense_decoded'.
    weight_type  : str
        Type of weight array to load: 'kernel' or 'bias'.
    phases       : tuple of str, optional (default=('init', 'trained'))
        The two training phases to compare. Each must match the phase label
        used when saving the corresponding .npz file.
    save_dir     : str, optional (default='./weights')
        Directory where the .npz weight files are stored.
    out_dir      : str, optional (default='./plots')
        Directory where the output .png plot is saved.

    Returns
    -------
    None
    """
    os.makedirs(out_dir, exist_ok=True)

    phase_colors = {phases[0]: "royalblue", phases[1]: "darkorange"}

    # Load weights for both phases
    data = {}
    for phase in phases:
        filepath = os.path.join(save_dir, f"{dataset_name}_{seed}_{phase}.npz")
        if not os.path.exists(filepath):
            print(f"  [warning] file not found: {filepath}")
            continue
        npz = np.load(filepath)
        key = f"{submodel}__{layer}__{weight_type}"
        if key not in npz:
            print(f"  [warning] key '{key}' not in {filepath}")
            continue
        data[phase] = npz[key].flatten()

    if len(data) < 2:
        print("  [error] could not load data for both phases.")
        return

    # Build long-form DataFrame with phase as grouping variable
    rows = []
    for phase, values in data.items():
        for v in values:
            rows.append({"phase": phase, "value": v})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5, 5))

    sns.violinplot(
        data=df, x="phase", y="value",
        hue="phase", legend=False,
        inner="box",
        linewidth=2,
        palette=phase_colors,
        ax=ax
    )

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_title(f"Parameters' distributions — {submodel} › {layer} › {weight_type}\n"
                 f"Dataset: {dataset_name}  |  Seed: {seed}", fontsize=11)
    ax.set_xlabel("phase")
    ax.set_ylabel("parameter values")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"violin_comparison_{dataset_name}_{seed}_{submodel}_{layer}_{weight_type}.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"  [plot saved] → {out_path}")


def plot_histograms_comparison(dataset_name, seed, submodel, layer, weight_type,
                               phases=("init", "trained"), bins=50,
                               save_dir="./weights", out_dir="./plots"):
    """
    Plot two histograms side by side on the same canvas, comparing the weight
    distribution of a given layer for two different training phases of the same seed.
    Both panels share the same x-axis range to make the comparison straightforward.
    Each panel includes vertical lines marking the mean and median of the distribution.
    The plot is saved as a .png file in out_dir.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset, used to locate .npz files and label the plot.
    seed         : int
        Seed of the run to compare across phases.
    submodel     : str
        Submodel to which the layer belongs: 'encoder' or 'decoder'.
    layer        : str
        Name of the layer whose weights are plotted. E.g. 'conv1', 'z_mean', 'dense_decoded'.
    weight_type  : str
        Type of weight array to load: 'kernel' or 'bias'.
    phases       : tuple of str, optional (default=('init', 'trained'))
        The two training phases to compare. Each must match the phase label
        used when saving the corresponding .npz file.
    bins         : int, optional (default=50)
        Number of bins for each histogram.
    save_dir     : str, optional (default='./weights')
        Directory where the .npz weight files are stored.
    out_dir      : str, optional (default='./plots')
        Directory where the output .png plot is saved.

    Returns
    -------
    None
    """
    os.makedirs(out_dir, exist_ok=True)

    phase_colors = {phases[0]: "royalblue", phases[1]: "darkorange"}

    # Load weights for both phases
    data = {}
    for phase in phases:
        filepath = os.path.join(save_dir, f"{dataset_name}_{seed}_{phase}.npz")
        if not os.path.exists(filepath):
            print(f"  [warning] file not found: {filepath}")
            continue
        npz = np.load(filepath)
        key = f"{submodel}__{layer}__{weight_type}"
        if key not in npz:
            print(f"  [warning] key '{key}' not in {filepath}")
            continue
        data[phase] = npz[key].flatten()

    if len(data) < 2:
        print("  [error] could not load data for both phases.")
        return

    # Shared x-axis range across both panels
    all_values = np.concatenate(list(data.values()))
    x_min, x_max = all_values.min(), all_values.max()
    x_margin = (x_max - x_min) * 0.05
    shared_range = (x_min - x_margin, x_max + x_margin)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, phase in zip(axes, phases):
        values = data[phase]
        ax.hist(values, bins=bins, range=shared_range,
                color=phase_colors[phase],
                edgecolor="white", linewidth=0.4, alpha=0.85)
        ax.axvline(0,                     color="gray",   linestyle="--", linewidth=1)
        ax.axvline(values.mean(),         color="yellow", linestyle="-",  linewidth=2,
                   label=f"mean={values.mean():.4f}")
        ax.axvline(np.median(values),     color="orange", linestyle="-.", linewidth=2,
                   label=f"median={np.median(values):.4f}")
        ax.set_title(f"phase: {phase}", fontsize=12)
        ax.set_ylabel("counts")
        ax.legend(fontsize=10)

    fig.suptitle(f"Parameters' distributions — {submodel} › {layer} › {weight_type}\n"
                 f"Dataset: {dataset_name}  |  Seed: {seed}",
                 fontsize=12, y=1.02)
    fig.supxlabel("parameter values")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"hist_comparison_{dataset_name}_{seed}_{submodel}_{layer}_{weight_type}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [plot saved] → {out_path}")
    
    
def plot_histograms_deltas(dataset_name, seeds, submodel, layer, weight_type,
                           phases=("init", "trained"), success=None, bins=50,
                           save_dir="./weights", out_dir="./plots"):
    """
    Plot a grid of histograms showing the element-wise difference (delta) between
    the weights of two training phases for a given layer, one panel per seed.
    Delta is computed as: weights[phases[1]] - weights[phases[0]], preserving the
    original weight ordering (i.e. weights are flattened consistently before subtraction).
    Each panel includes vertical lines marking mean and median of the delta distribution.
    All panels share the same x-axis range for cross-seed comparison.
    The plot is saved as a .png file in out_dir.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset, used to locate .npz files and label the plot.
    seeds        : array-like of int
        Seeds used during training, one histogram panel is drawn per seed.
    submodel     : str
        Submodel to which the layer belongs: 'encoder' or 'decoder'.
    layer        : str
        Name of the layer whose weight deltas are plotted. E.g. 'conv1', 'z_mean'.
    weight_type  : str
        Type of weight array to load: 'kernel' or 'bias'.
    phases       : tuple of str, optional (default=('init', 'trained'))
        The two phases to subtract. Delta = phases[1] - phases[0].
    success      : array-like of bool or None, optional (default=None)
        Boolean array of the same length as seeds. True indicates a successful training run,
        False a failed one. Colors each panel dark green or dark red accordingly,
        and adds a success/failure label to each panel title. If None, all panels are purple.
    bins         : int, optional (default=50)
        Number of bins for each histogram.
    save_dir     : str, optional (default='./weights')
        Directory where the .npz weight files are stored.
    out_dir      : str, optional (default='./plots')
        Directory where the output .png plot is saved.

    Returns
    -------
    None
    """
    os.makedirs(out_dir, exist_ok=True)

    if success is not None:
        assert len(success) == len(seeds), \
            f"success and seeds have different lengths: {len(success)} vs {len(seeds)}"
        success_map = {str(seed): ok for seed, ok in zip(seeds, success)}
        color_map = {
            str(seed): "forestgreen" if success_map[str(seed)] else "firebrick"
            for seed in seeds
        }
    else:
        color_map = {str(seed): "mediumpurple" for seed in seeds}

    # Load deltas for each seed
    key = f"{submodel}__{layer}__{weight_type}"
    deltas = {}

    for seed in seeds:
        arrays = {}
        for phase in phases:
            filepath = os.path.join(save_dir, f"{dataset_name}_{seed}_{phase}.npz")
            if not os.path.exists(filepath):
                print(f"  [warning] file not found: {filepath}")
                break
            npz = np.load(filepath)
            if key not in npz:
                print(f"  [warning] key '{key}' not in {filepath}")
                break
            arrays[phase] = npz[key].flatten()

        if len(arrays) < 2:
            print(f"  [warning] skipping seed {seed}: could not load both phases.")
            continue

        deltas[seed] = arrays[phases[1]] - arrays[phases[0]]

    if not deltas:
        print("  [error] no delta data available.")
        return

    n     = len(deltas)
    ncols = min(n, 2)
    nrows = math.ceil(n / ncols)

    # Shared x-axis range across all panels
    all_deltas = np.concatenate(list(deltas.values()))
    x_min, x_max = all_deltas.min(), all_deltas.max()
    x_margin = (x_max - x_min) * 0.05
    shared_range = (x_min - x_margin, x_max + x_margin)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 3.5))
    axes = np.array(axes).flatten()

    for ax, (seed, delta) in zip(axes, deltas.items()):
        ax.hist(delta, bins=bins, range=shared_range,
                color=color_map[str(seed)],
                edgecolor="white", linewidth=0.4, alpha=0.85)
        ax.axvline(0,                   color="gray",   linestyle="--", linewidth=1)
        ax.axvline(delta.mean(),        color="yellow", linestyle="-",  linewidth=2,
                   label=f"mean={delta.mean():.4f}")
        ax.axvline(np.median(delta),    color="orange", linestyle="-.", linewidth=2,
                   label=f"median={np.median(delta):.4f}")
        ax.set_ylabel("counts")
        ax.legend(fontsize=10)
        ax.set_yscale('log')

        if success is not None:
            outcome = "(success)" if success_map[str(seed)] else "(failure)"
            ax.set_title(f"seed {seed} {outcome}", fontsize=12)
        else:
            ax.set_title(f"seed {seed}", fontsize=12)

    # Hid last panel when n is odd
    if n%2 == 1:
        for ax in axes[n:]:
            ax.set_visible(False)     
            
    fig.suptitle(f"Weight differences ({phases[1]} − {phases[0]}) — "
                 f"{submodel} › {layer} › {weight_type}\n"
                 f"Dataset: {dataset_name}",
                 fontsize=12, y=1.01)
    fig.supxlabel(f"Δ {weight_type} ({phases[1]} − {phases[0]})")
    plt.tight_layout()

    out_path = os.path.join(
        out_dir,
        f"hist_deltas_{dataset_name}_{submodel}_{layer}_{weight_type}.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [plot saved] → {out_path}")