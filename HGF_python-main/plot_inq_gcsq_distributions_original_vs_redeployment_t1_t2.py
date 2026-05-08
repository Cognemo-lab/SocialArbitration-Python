from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
OUT_DIR = BASE / "behavior_fod_moderation_comparison" / "state_and_prevalence_assessment"
OUT_FIG = OUT_DIR / "figure_inq_gcsq_distributions_original_vs_redeployment_t1_t2.png"
OUT_CSV = OUT_DIR / "inq_gcsq_distributions_original_vs_redeployment_t1_t2_summary.csv"

METRICS = [
    ("inq_grand_sum", "INQ total"),
    ("inq_thwarted_belongigness", "INQ thwarted belongingness"),
    ("inq_perceived_burdensomness", "INQ perceived burdensomeness"),
    ("gcsq_perceived_capability", "GCSQ acquired capability"),
    ("gcsq_pain_tolerance", "GCSQ pain tolerance"),
    ("gcsq_fearlesness_of_death", "GCSQ fearlessness of death"),
]

GROUPS = [
    ("Original", "t1", "Original T1", "#4C78A8"),
    ("Original", "t2", "Original T2", "#8FB9E0"),
    ("Redeployed", "t1", "Redeployed T1", "#E76F51"),
    ("Redeployed", "t2", "Redeployed T2", "#F2A07E"),
]


def load_long() -> pd.DataFrame:
    orig_t1 = pd.read_csv(BASE / "hitop" / "processed_data" / "additional_suicidality_scales_T1.csv")[
        ["prolific_id", *[m for m, _ in METRICS]]
    ].copy()
    orig_t1["dataset"] = "Original"
    orig_t1["timepoint"] = "t1"

    orig_t2 = pd.read_csv(BASE / "hitop" / "processed_data" / "additional_suicidality_scales_T2.csv")[
        ["prolific_id", *[m for m, _ in METRICS]]
    ].copy()
    orig_t2["dataset"] = "Original"
    orig_t2["timepoint"] = "t2"

    red_t1 = pd.read_csv(BASE / "hitop_batch2" / "processed_data" / "psych_scales_T1_extended.csv")[
        ["prolific_id", *[m for m, _ in METRICS]]
    ].copy()
    red_t1["dataset"] = "Redeployed"
    red_t1["timepoint"] = "t1"

    red_t2 = pd.read_csv(BASE / "hitop_batch2" / "processed_data" / "psych_scales_T2_extended.csv")[
        ["prolific_id", *[m for m, _ in METRICS]]
    ].copy()
    red_t2["dataset"] = "Redeployed"
    red_t2["timepoint"] = "t2"

    return pd.concat([orig_t1, orig_t2, red_t1, red_t2], ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, label in METRICS:
        for dataset, timepoint, group_label, _ in GROUPS:
            s = pd.to_numeric(
                df.loc[(df["dataset"] == dataset) & (df["timepoint"] == timepoint), metric],
                errors="coerce",
            ).dropna()
            if s.empty:
                continue
            rows.append(
                {
                    "metric": metric,
                    "label": label,
                    "dataset": dataset,
                    "timepoint": timepoint,
                    "group_label": group_label,
                    "n": int(len(s)),
                    "mean": float(s.mean()),
                    "sd": float(s.std(ddof=0)),
                    "median": float(s.median()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "q25": float(s.quantile(0.25)),
                    "q75": float(s.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_long()
    summary = summarize(df)
    summary.to_csv(OUT_CSV, index=False)

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(12, 20))
    rng = np.random.default_rng(42)

    for ax, (metric, label) in zip(axes, METRICS):
        values = []
        positions = np.arange(1, len(GROUPS) + 1)
        colors = []
        ns = []
        for dataset, timepoint, _group_label, color in GROUPS:
            s = pd.to_numeric(
                df.loc[(df["dataset"] == dataset) & (df["timepoint"] == timepoint), metric],
                errors="coerce",
            ).dropna()
            values.append(s.to_numpy())
            colors.append(color)
            ns.append(len(s))

        parts = ax.violinplot(values, positions=positions, showmeans=False, showmedians=False, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.35)

        box = ax.boxplot(
            values,
            positions=positions,
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.3},
            whiskerprops={"color": "#555", "linewidth": 1.0},
            capprops={"color": "#555", "linewidth": 1.0},
            boxprops={"linewidth": 1.0, "color": "#555"},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        for x, vals, color in zip(positions, values, colors):
            if len(vals) == 0:
                continue
            jitter = rng.normal(0, 0.04, size=len(vals))
            sample_idx = np.arange(len(vals))
            if len(vals) > 250:
                sample_idx = rng.choice(len(vals), size=250, replace=False)
            ax.scatter(
                np.full(len(sample_idx), x) + jitter[sample_idx],
                vals[sample_idx],
                s=9,
                color=color,
                alpha=0.18,
                edgecolors="none",
                zorder=3,
            )

        for x, n in zip(positions, ns):
            ax.text(x, 0.98, f"n={n}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9, color="#444")

        ax.set_title(label, fontsize=13, pad=8)
        ax.set_xticks(positions)
        ax.set_xticklabels([g[2] for g in GROUPS], fontsize=10)
        ax.grid(axis="y", alpha=0.18)
        ax.set_ylabel("Score")

    fig.suptitle("INQ and acquired capability measure distributions across original and redeployed samples", fontsize=17, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=1.5)
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    print(OUT_FIG)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
