import argparse
import os

import numpy as np
import pandas as pd


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Plot state-level T1/T2 scatterplots for reliable metrics.')
    parser.add_argument('--reliability-csv', required=True)
    parser.add_argument('--wide-csv', required=True)
    parser.add_argument('--fig-out', required=True)
    parser.add_argument('--threshold', type=float, default=0.4)
    args = parser.parse_args()

    rel_df = pd.read_csv(args.reliability_csv)
    wide_df = pd.read_csv(args.wide_csv)

    keep = rel_df.loc[rel_df['pearson_r_t1_t2'] > args.threshold, 'state_metric'].tolist()
    d = wide_df[wide_df['state_metric'].isin(keep)].dropna(subset=['t1', 't2']).copy()
    if d.empty:
        raise RuntimeError('No state metrics passed the threshold.')

    n = len(keep)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax in axes.ravel():
        ax.set_visible(False)

    for i, metric in enumerate(keep):
        ax = axes.ravel()[i]
        ax.set_visible(True)
        dd = d[d['state_metric'] == metric]
        x = dd['t1'].to_numpy(dtype=float)
        y = dd['t2'].to_numpy(dtype=float)
        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        if lo == hi:
            pad = 0.1 if lo == 0 else abs(lo) * 0.05
            lo -= pad
            hi += pad
        rr = rel_df[rel_df['state_metric'] == metric].iloc[0]
        ax.scatter(x, y, s=28, alpha=0.75, color='#2B6CB0')
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_title(f'{metric}\nr={rr["pearson_r_t1_t2"]:.2f}, ICC={rr["icc3_1_t1_t2"]:.2f}')
        ax.set_xlabel('t1')
        ax.set_ylabel('t2')
        ax.grid(alpha=0.2)

    plt.tight_layout()
    _ensure_dir(os.path.dirname(args.fig_out))
    fig.savefig(args.fig_out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(args.fig_out)


if __name__ == '__main__':
    main()
