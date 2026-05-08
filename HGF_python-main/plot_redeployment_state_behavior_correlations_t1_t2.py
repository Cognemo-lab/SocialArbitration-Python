from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
IN_CSV = BASE / "behavior_test_retest_comparison" / "state_dependence_redeployment" / "behavior_with_state_long.csv"
OUT_DIR = BASE / "behavior_test_retest_comparison" / "state_dependence_redeployment"
OUT_CORR = OUT_DIR / "state_behavior_correlations_t1_t2.csv"
OUT_DELTA = OUT_DIR / "state_behavior_correlations_delta_t2_minus_t1.csv"
OUT_FIG = OUT_DIR / "figure_state_behavior_correlations_t1_t2.png"

STATE_VARS = [
    ("sleep_hours", "Sleep hours"),
    ("subjective_rest", "Subjective rest"),
    ("motivation_level", "Motivation"),
    ("medication_taken_bin", "Medication"),
    ("caffeine_consumed_bin", "Caffeine"),
    ("alcohol_consumed_bin", "Alcohol"),
    ("nicotine_used_bin", "Nicotine"),
    ("drug_used_bin", "Drug use"),
]

BEHAVIOR_VARS = [
    ("accuracy_rate", "Accuracy"),
    ("accuracy_first30", "Accuracy first 30"),
    ("accuracy_last20", "Accuracy last 20"),
    ("advice_taking_rate", "Advice-taking"),
    ("win_stay_rate", "Win-stay"),
    ("lose_switch_rate", "Lose-switch"),
    ("mean_wager", "Mean wager"),
]


def corr_table(df: pd.DataFrame, timepoint: str) -> pd.DataFrame:
    sub = df[df["timepoint"] == timepoint].copy()
    rows = []
    for svar, slabel in STATE_VARS:
        for bvar, blabel in BEHAVIOR_VARS:
            pair = sub[[svar, bvar]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 10:
                r, p = np.nan, np.nan
            else:
                r, p = stats.pearsonr(pair[svar], pair[bvar])
            rows.append(
                {
                    "timepoint": timepoint,
                    "state_var": svar,
                    "state_label": slabel,
                    "behavior_var": bvar,
                    "behavior_label": blabel,
                    "n": int(len(pair)),
                    "pearson_r": float(r) if pd.notna(r) else np.nan,
                    "p_value": float(p) if pd.notna(p) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_matrix(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        df.pivot(index="state_label", columns="behavior_label", values=value_col)
        .reindex(index=[s[1] for s in STATE_VARS], columns=[b[1] for b in BEHAVIOR_VARS])
    )


def annotate(ax, mat: pd.DataFrame, pmat: pd.DataFrame | None = None) -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            if pd.isna(val):
                txt = "NA"
            elif pmat is None:
                txt = f"{val:.2f}"
            else:
                p = pmat.iloc[i, j]
                stars = "***" if pd.notna(p) and p < 0.001 else "**" if pd.notna(p) and p < 0.01 else "*" if pd.notna(p) and p < 0.05 else ""
                txt = f"{val:.2f}{stars}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")


def main() -> None:
    df = pd.read_csv(IN_CSV)
    t1 = corr_table(df, "t1")
    t2 = corr_table(df, "t2")
    corr = pd.concat([t1, t2], ignore_index=True)
    corr.to_csv(OUT_CORR, index=False)

    t1m = build_matrix(t1, "pearson_r")
    t2m = build_matrix(t2, "pearson_r")
    t1p = build_matrix(t1, "p_value")
    t2p = build_matrix(t2, "p_value")
    delta = t2m - t1m
    delta_long = delta.reset_index().melt(id_vars="state_label", var_name="behavior_label", value_name="delta_r")
    delta_long.to_csv(OUT_DELTA, index=False)

    vmax = max(np.nanmax(np.abs(t1m.to_numpy())), np.nanmax(np.abs(t2m.to_numpy())), 0.25)
    dmax = max(np.nanmax(np.abs(delta.to_numpy())), 0.15)

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    panels = [
        (axes[0], t1m, t1p, "T1 correlations", vmax),
        (axes[1], t2m, t2p, "T2 correlations", vmax),
        (axes[2], delta, None, "T2 - T1 change", dmax),
    ]

    for ax, mat, pmat, title, lim in panels:
        im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xticks(range(len(BEHAVIOR_VARS)))
        ax.set_xticklabels([b[1] for b in BEHAVIOR_VARS], rotation=35, ha="right")
        ax.set_yticks(range(len(STATE_VARS)))
        ax.set_yticklabels([s[1] for s in STATE_VARS])
        annotate(ax, mat, pmat)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r")

    fig.suptitle("Redeployment pre-task vs behavior correlations at T1 and T2", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95], w_pad=2.2)
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    print(OUT_CORR)
    print(OUT_DELTA)
    print(OUT_FIG)


if __name__ == "__main__":
    main()
