from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
IN_CSV = BASE / "behavior_test_retest_comparison" / "state_dependence_redeployment" / "behavior_with_state_long.csv"
OUT_DIR = BASE / "behavior_test_retest_comparison" / "state_dependence_redeployment"
OUT_CORR = OUT_DIR / "intra_behavior_correlations_t1_t2.csv"
OUT_DELTA = OUT_DIR / "intra_behavior_correlations_delta_t2_minus_t1.csv"
OUT_FIG = OUT_DIR / "figure_intra_behavior_correlations_t1_t2.png"

BEHAVIORS = [
    ("accuracy_rate", "Accuracy"),
    ("accuracy_first30", "Accuracy first 30"),
    ("accuracy_last20", "Accuracy last 20"),
    ("advice_taking_rate", "Advice-taking"),
    ("win_stay_rate", "Win-stay"),
    ("lose_switch_rate", "Lose-switch"),
    ("mean_wager", "Mean wager"),
]


def pairwise_corr(df: pd.DataFrame, timepoint: str) -> pd.DataFrame:
    sub = df[df["timepoint"] == timepoint].copy()
    rows = []
    for var1, lab1 in BEHAVIORS:
        for var2, lab2 in BEHAVIORS:
            pair = sub[[var1, var2]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 10:
                r, p = np.nan, np.nan
            elif var1 == var2:
                r, p = 1.0, 0.0
            else:
                r, p = stats.pearsonr(pair[var1], pair[var2])
            rows.append(
                {
                    "timepoint": timepoint,
                    "var1": var1,
                    "label1": lab1,
                    "var2": var2,
                    "label2": lab2,
                    "n": int(len(pair)),
                    "pearson_r": float(r) if pd.notna(r) else np.nan,
                    "p_value": float(p) if pd.notna(p) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_matrix(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    order = [b[1] for b in BEHAVIORS]
    return df.pivot(index="label1", columns="label2", values=value_col).reindex(index=order, columns=order)


def annotate(ax, mat: pd.DataFrame, pmat: pd.DataFrame | None = None) -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if pd.isna(val):
                txt = "NA"
            elif pmat is None or i == j:
                txt = f"{val:.2f}"
            else:
                p = pmat.iloc[i, j]
                stars = "***" if pd.notna(p) and p < 0.001 else "**" if pd.notna(p) and p < 0.01 else "*" if pd.notna(p) and p < 0.05 else ""
                txt = f"{val:.2f}{stars}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color="black")


def main() -> None:
    df = pd.read_csv(IN_CSV)
    t1 = pairwise_corr(df, "t1")
    t2 = pairwise_corr(df, "t2")
    corr = pd.concat([t1, t2], ignore_index=True)
    corr.to_csv(OUT_CORR, index=False)

    t1m = build_matrix(t1, "pearson_r")
    t2m = build_matrix(t2, "pearson_r")
    t1p = build_matrix(t1, "p_value")
    t2p = build_matrix(t2, "p_value")
    delta = t2m - t1m
    delta_long = delta.reset_index().melt(id_vars="label1", var_name="label2", value_name="delta_r")
    delta_long.to_csv(OUT_DELTA, index=False)

    vmax = max(np.nanmax(np.abs(t1m.to_numpy())), np.nanmax(np.abs(t2m.to_numpy())), 0.5)
    dmax = max(np.nanmax(np.abs(delta.to_numpy())), 0.15)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    panels = [
        (axes[0], t1m, t1p, "T1 correlations", vmax),
        (axes[1], t2m, t2p, "T2 correlations", vmax),
        (axes[2], delta, None, "T2 - T1 change", dmax),
    ]

    for ax, mat, pmat, title, lim in panels:
        im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xticks(range(len(BEHAVIORS)))
        ax.set_xticklabels([b[1] for b in BEHAVIORS], rotation=35, ha="right")
        ax.set_yticks(range(len(BEHAVIORS)))
        ax.set_yticklabels([b[1] for b in BEHAVIORS])
        annotate(ax, mat, pmat)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r")

    fig.suptitle("Redeployment intra-behavioral correlations at T1 and T2", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95], w_pad=2.2)
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    print(OUT_CORR)
    print(OUT_DELTA)
    print(OUT_FIG)


if __name__ == "__main__":
    main()
