from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
IN_FILE = BASE / "behavior_fod_moderation_comparison" / "accuracy_fod_moderation_summary.csv"
OUT = BASE / "behavior_fod_moderation_comparison" / "figure_accuracy_fod_moderation_original_vs_redeployment.png"

SCOPE_ORDER = ["pooled", "subject_mean", "t1", "t2"]
SCOPE_LABELS = {
    "pooled": "Pooled clustered",
    "subject_mean": "Subject mean",
    "t1": "T1 only",
    "t2": "T2 only",
}
COLORS = {"Original": "#1d3557", "Redeployed": "#e76f51"}


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def main() -> None:
    df = pd.read_csv(IN_FILE)
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(SCOPE_ORDER))[::-1]
    for off, dataset in [(-0.12, "Original"), (0.12, "Redeployed")]:
        cur = df.loc[df["dataset"] == dataset].set_index("scope").reindex(SCOPE_ORDER)
        ax.errorbar(
            cur["interaction_beta"],
            y + off,
            xerr=1.96 * cur["interaction_se"],
            fmt="o",
            color=COLORS[dataset],
            ms=7,
            capsize=3,
            label=dataset,
        )
        for x, yy, p in zip(cur["interaction_beta"], y + off, cur["interaction_p"]):
            s = stars(float(p))
            if s:
                ax.text(x, yy + 0.08, s, ha="center", va="bottom", fontsize=10, color=COLORS[dataset])
    ax.axvline(0, color="#888", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([SCOPE_LABELS[s] for s in SCOPE_ORDER], fontsize=11)
    ax.set_xlabel("Interaction beta")
    ax.set_title("HiTOP suicidality moderating\naccuracy -> fearlessness of death")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
