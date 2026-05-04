from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
IN_DIR = BASE / "hitop" / "hierarchical_fod_moderation_features"

LABELS = {
    "abs_eps2_a_mean_z:hitop_suicidality_z": "Abs advice epsilon2 × suicidality",
    "inferv_a_mean_z:hitop_suicidality_z": "Advice inferential variance × suicidality",
    "ka_a_z:hitop_suicidality_z": "ka_a × suicidality",
    "om_a_z:hitop_suicidality_z": "om_a × suicidality",
    "eps3_a_mean_z:hitop_suicidality_z": "Advice epsilon3 × suicidality",
}


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def main() -> None:
    coef = pd.read_csv(IN_DIR / "hierarchical_moderation_interactions.csv").copy()
    coef["label"] = coef["term"].map(LABELS).fillna(coef["term"])
    coef = coef.sort_values("coef").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = ["#2a9d8f" if p < 0.05 else "#9aa1a8" for p in coef["p_value"]]
    y = range(len(coef))

    ax.barh(y, coef["coef"], color=colors, alpha=0.9)
    ax.errorbar(
        coef["coef"],
        y,
        xerr=[coef["coef"] - coef["ci_low"], coef["ci_high"] - coef["coef"]],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=3,
    )
    ax.axvline(0, color="0.25", linewidth=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(coef["label"])
    ax.set_xlabel("Hierarchical interaction beta")
    ax.set_title("Fearlessness of Death: Suicidality Interaction Terms")

    lim = max(abs(coef["ci_low"]).max(), abs(coef["ci_high"]).max()) * 1.1
    ax.set_xlim(-lim, lim)

    for i, row in coef.iterrows():
        txt = f"{row['coef']:.2f}{stars(row['p_value'])}\np={row['p_value']:.3g}"
        xpos = row["coef"] + 0.03 if row["coef"] >= 0 else row["coef"] - 0.03
        ha = "left" if row["coef"] >= 0 else "right"
        ax.text(xpos, i, txt, va="center", ha=ha, fontsize=9)

    note = "Green: p < .05; Gray: non-significant. Error bars show 95% CI."
    fig.text(0.5, 0.02, note, ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(IN_DIR / "figure_hierarchical_fod_interactions.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
