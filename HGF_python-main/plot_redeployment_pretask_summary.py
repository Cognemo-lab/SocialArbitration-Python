from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
IN_CSV = BASE / "behavior_test_retest_comparison" / "state_dependence_redeployment" / "pre_task_state_long.csv"
OUT_DIR = BASE / "behavior_test_retest_comparison" / "state_dependence_redeployment"
OUT_FIG = OUT_DIR / "figure_redeployment_pretask_summary_t1_t2.png"
OUT_CSV = OUT_DIR / "pretask_summary_t1_t2.csv"

CONTINUOUS = [
    ("sleep_hours", "Sleep hours"),
    ("subjective_rest", "Subjective rest"),
    ("motivation_level", "Motivation"),
]

BINARY = [
    ("medication_taken_bin", "Medication"),
    ("caffeine_consumed_bin", "Caffeine"),
    ("alcohol_consumed_bin", "Alcohol"),
    ("nicotine_used_bin", "Nicotine"),
    ("drug_used_bin", "Drug use"),
]

COLORS = {"t1": "#4C78A8", "t2": "#E76F51"}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tp in ["t1", "t2"]:
        sub = df[df["timepoint"] == tp].copy()
        for col, label in CONTINUOUS:
            s = pd.to_numeric(sub[col], errors="coerce").dropna()
            rows.append(
                {
                    "timepoint": tp,
                    "metric": col,
                    "label": label,
                    "type": "continuous",
                    "n": int(len(s)),
                    "mean": float(s.mean()),
                    "sd": float(s.std(ddof=0)),
                    "median": float(s.median()),
                    "q25": float(s.quantile(0.25)),
                    "q75": float(s.quantile(0.75)),
                }
            )
        for col, label in BINARY:
            s = pd.to_numeric(sub[col], errors="coerce").dropna()
            rows.append(
                {
                    "timepoint": tp,
                    "metric": col,
                    "label": label,
                    "type": "binary",
                    "n": int(len(s)),
                    "mean": float(s.mean()),
                    "sd": float(s.std(ddof=0)),
                    "median": float(s.median()),
                    "q25": float(s.quantile(0.25)),
                    "q75": float(s.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(IN_CSV)
    summary = summarize(df)
    summary.to_csv(OUT_CSV, index=False)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Continuous measures
    x = np.arange(len(CONTINUOUS))
    width = 0.36
    t1_vals = [summary.loc[(summary["timepoint"] == "t1") & (summary["metric"] == m), "mean"].iloc[0] for m, _ in CONTINUOUS]
    t2_vals = [summary.loc[(summary["timepoint"] == "t2") & (summary["metric"] == m), "mean"].iloc[0] for m, _ in CONTINUOUS]
    t1_sd = [summary.loc[(summary["timepoint"] == "t1") & (summary["metric"] == m), "sd"].iloc[0] for m, _ in CONTINUOUS]
    t2_sd = [summary.loc[(summary["timepoint"] == "t2") & (summary["metric"] == m), "sd"].iloc[0] for m, _ in CONTINUOUS]

    ax1.bar(x - width / 2, t1_vals, width, color=COLORS["t1"], alpha=0.9, label="T1")
    ax1.bar(x + width / 2, t2_vals, width, color=COLORS["t2"], alpha=0.9, label="T2")
    ax1.errorbar(x - width / 2, t1_vals, yerr=t1_sd, fmt="none", ecolor="#333", capsize=4, linewidth=1.0)
    ax1.errorbar(x + width / 2, t2_vals, yerr=t2_sd, fmt="none", ecolor="#333", capsize=4, linewidth=1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels([label for _, label in CONTINUOUS])
    ax1.set_title("Continuous pre-task measures", fontsize=13, pad=10)
    ax1.grid(axis="y", alpha=0.2)

    # Binary measures
    ypos = np.arange(len(BINARY))
    t1_prop = np.array([summary.loc[(summary["timepoint"] == "t1") & (summary["metric"] == m), "mean"].iloc[0] for m, _ in BINARY]) * 100
    t2_prop = np.array([summary.loc[(summary["timepoint"] == "t2") & (summary["metric"] == m), "mean"].iloc[0] for m, _ in BINARY]) * 100
    h = 0.36
    ax2.barh(ypos + h / 2, t1_prop, height=h, color=COLORS["t1"], alpha=0.9, label="T1")
    ax2.barh(ypos - h / 2, t2_prop, height=h, color=COLORS["t2"], alpha=0.9, label="T2")
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([label for _, label in BINARY])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.set_title("Recent use / state indicators (%)", fontsize=13, pad=10)
    ax2.grid(axis="x", alpha=0.2)

    for y, val in zip(ypos + h / 2, t1_prop):
        ax2.text(val + 1.2, y, f"{val:.1f}", va="center", fontsize=8)
    for y, val in zip(ypos - h / 2, t2_prop):
        ax2.text(val + 1.2, y, f"{val:.1f}", va="center", fontsize=8)

    n_t1 = int(summary.loc[(summary["timepoint"] == "t1") & (summary["metric"] == "sleep_hours"), "n"].iloc[0])
    n_t2 = int(summary.loc[(summary["timepoint"] == "t2") & (summary["metric"] == "sleep_hours"), "n"].iloc[0])
    fig.suptitle(
        f"Redeployment pre-task questionnaire summary\nT1 n={n_t1}, T2 n={n_t2}",
        fontsize=16,
        y=0.98,
    )
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.92))
    fig.tight_layout(rect=[0, 0, 1, 0.9], w_pad=2.5)
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    print(OUT_FIG)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
