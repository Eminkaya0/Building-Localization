"""Generate publication-quality figures from experiment results."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# IEEE-friendly style
rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'results', 'all_experiments.json')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'paper', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def fig_baseline_comparison(data):
    """Bar chart: RMSE comparison across methods."""
    bl = data['baseline']
    methods = ['Odometry', 'EKF\nh=40m', 'EKF\nh=60m', 'EKF\nh=80m', 'EKF\nh=100m', 'Adaptive\nEKF']
    keys = ['odometry', 'ekf_fixed_40', 'ekf_fixed_60', 'ekf_fixed_80', 'ekf_fixed_100', 'adaptive']
    means = [bl[k]['mean_rmse'] for k in keys]
    stds = [bl[k]['std'] for k in keys]

    colors = ['#7f8c8d'] + ['#3498db']*4 + ['#e74c3c']

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    bars = ax.bar(range(len(methods)), means, yerr=stds, capsize=3,
                  color=colors, edgecolor='black', linewidth=0.5, width=0.65)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=7)
    ax.set_ylabel('RMSE (m)')
    ax.set_ylim(0, 8)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Value labels
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=6.5)

    fig.savefig(os.path.join(FIG_DIR, 'baseline_comparison.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'baseline_comparison.png'))
    plt.close(fig)
    print("  Saved baseline_comparison.pdf/png")


def fig_noise_sensitivity(data):
    """Line plot: RMSE vs pixel noise sigma."""
    nd = data['noise']
    sigmas = sorted([int(k) for k in nd.keys()])
    means = [nd[str(s)]['mean'] for s in sigmas]
    stds = [nd[str(s)]['std'] for s in sigmas]

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.errorbar(sigmas, means, yerr=stds, fmt='o-', color='#2c3e50',
                markersize=4, capsize=3, linewidth=1.2, markerfacecolor='#3498db',
                markeredgecolor='#2c3e50', markeredgewidth=0.8)
    ax.set_xlabel(r'Pixel noise $\sigma$ (px)')
    ax.set_ylabel('RMSE (m)')
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 0.75)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(FIG_DIR, 'noise_sensitivity.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'noise_sensitivity.png'))
    plt.close(fig)
    print("  Saved noise_sensitivity.pdf/png")


def fig_observability_comparison(data):
    """Side-by-side bar: circular vs straight observability."""
    obs = data['observability']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.0))

    # Condition number
    trajs = ['Circular', 'Straight']
    conds = [obs['circular']['cond'], obs['straight']['cond']]
    colors = ['#27ae60', '#e74c3c']
    bars1 = ax1.bar(trajs, conds, color=colors, edgecolor='black', linewidth=0.5, width=0.5)
    ax1.set_ylabel('Condition Number')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for bar, val in zip(bars1, conds):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    # Mean error
    errors = [obs['circular']['mean_error'], obs['straight']['mean_error']]
    bars2 = ax2.bar(trajs, errors, color=colors, edgecolor='black', linewidth=0.5, width=0.5)
    ax2.set_ylabel('Mean Error (m)')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    for bar, val in zip(bars2, errors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(FIG_DIR, 'observability_comparison.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'observability_comparison.png'))
    plt.close(fig)
    print("  Saved observability_comparison.pdf/png")


def fig_eigenvalue_spectrum(data):
    """Bar chart: eigenvalue spectrum for circular vs straight."""
    obs = data['observability']

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    x = np.arange(3)
    w = 0.3

    eig_c = obs['circular']['eigenvalues']
    eig_s = obs['straight']['eigenvalues']

    ax.bar(x - w/2, eig_c, w, label='Circular', color='#27ae60', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, eig_s, w, label='Straight', color='#e74c3c', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$'])
    ax.set_ylabel('Eigenvalue')
    ax.set_yscale('log')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(os.path.join(FIG_DIR, 'eigenvalue_spectrum.pdf'))
    fig.savefig(os.path.join(FIG_DIR, 'eigenvalue_spectrum.png'))
    plt.close(fig)
    print("  Saved eigenvalue_spectrum.pdf/png")


if __name__ == '__main__':
    print("Generating publication figures...")
    data = load_results()
    fig_baseline_comparison(data)
    fig_noise_sensitivity(data)
    fig_observability_comparison(data)
    fig_eigenvalue_spectrum(data)
    print(f"\nAll figures saved to {FIG_DIR}/")
