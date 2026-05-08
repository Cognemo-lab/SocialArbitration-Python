from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
OUT_DIR = BASE / "behavior_fod_moderation_comparison" / "state_and_prevalence_assessment"
OUT = OUT_DIR / "figure_hitop_floor_effects_original_vs_redeployment.png"

METRICS = [
    ("hitop_mistrust_suspiciousness", "Suspiciousness"),
    ("hitop_reality_distortion", "Reality distortion"),
    ("hitop_reality_distortion_delusions", "Delusions"),
    ("hitop_reality_distortion_hallucinations", "Hallucinations"),
]
COLORS = {"Original": "#4C78A8", "Redeployed": "#E76F51"}


def load_df() -> pd.DataFrame:
    orig_t1 = pd.read_csv(BASE / "hitop" / "processed_data" / "hitop_scales_T1.csv")[[m for m, _ in METRICS]].copy()
    orig_t2 = pd.read_csv(BASE / "hitop" / "processed_data" / "hitop_scales_T2.csv")[[m for m, _ in METRICS]].copy()
    orig = pd.concat([orig_t1, orig_t2], ignore_index=True)
    orig["dataset"] = "Original"

    red_t1 = pd.read_csv(BASE / "hitop_batch2" / "processed_data" / "psych_scales_T1_extended.csv")[[m for m, _ in METRICS]].copy()
    red_t2 = pd.read_csv(BASE / "hitop_batch2" / "processed_data" / "psych_scales_T2_extended.csv")[[m for m, _ in METRICS]].copy()
    red = pd.concat([red_t1, red_t2], ignore_index=True)
    red["dataset"] = "Redeployed"

    return pd.concat([orig, red], ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_df()

    summary_rows = []
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for idx, (metric, label) in enumerate(METRICS):
        ax = axes[0, idx]
        sub = df[["dataset", metric]].dropna().copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        levels = sorted(sub[metric].dropna().astype(int).unique())
        x = range(len(levels))
        width = 0.38
        for dataset, off in [("Original", -width / 2), ("Redeployed", width / 2)]:
            counts = (
                sub.loc[sub["dataset"] == dataset, metric]
                .astype(int)
                .value_counts()
                .reindex(levels, fill_value=0)
                .to_numpy()
            )
            ax.bar([i + off for i in x], counts, width=width, color=COLORS[dataset], label=dataset if idx == 0 else None)
        floor = min(levels)
        floor_idx = levels.index(floor)
        ax.axvline(floor_idx, color="#666", linestyle="--", linewidth=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(levels, fontsize=9)
        ax.set_title(label)
        if idx == 0:
            ax.set_ylabel("Count")
            ax.legend(frameon=False, fontsize=9)

        ax2 = axes[1, idx]
        floor_props = []
        for dataset in ["Original", "Redeployed"]:
            s = sub.loc[sub["dataset"] == dataset, metric].astype(int)
            floor_props.append((s == floor).mean())
            summary_rows.append(
                {
                    "metric": metric,
                    "label": label,
                    "dataset": dataset,
                    "n": int(len(s)),
                    "mean": float(s.mean()),
                    "sd": float(s.std(ddof=0)),
                    "min": int(s.min()),
                    "median": float(s.median()),
                    "max": int(s.max()),
                    "prop_floor": float((s == floor).mean()),
                }
            )
        ax2.bar(["Original", "Redeployed"], floor_props, color=[COLORS["Original"], COLORS["Redeployed"]])
        for i, val in enumerate(floor_props):
            ax2.text(i, val + 0.02, f"{val*100:.1f}%", ha="center", va="bottom", fontsize=10)
        ax2.set_ylim(0, 1.0)
        if idx == 0:
            ax2.set_ylabel("Proportion at floor")
        ax2.set_title(f"{label}\nfloor")

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "hitop_floor_effects_summary.csv", index=False)
    fig.suptitle("HiTOP floor effects: original vs redeployed sample", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
