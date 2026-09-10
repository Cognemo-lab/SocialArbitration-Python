from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
OUT_DIR = BASE / "project_report" / "supporting_artifacts" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


ROWS = [
    {
        "task": "Jungle Adventure",
        "predictor": "Social-learning accuracy",
        "beta": -0.19,
        "p_text": "p = .04",
        "outcome": "HiTOP reality distortion",
        "interpretation": "lower accuracy -> higher reality distortion",
        "color": "#3E7CB1",
    },
    {
        "task": "Goin' Fishing",
        "predictor": r"Belief instability $\kappa_1$",
        "beta": 0.24,
        "p_text": "p = .03",
        "outcome": "HiTOP suspiciousness",
        "interpretation": "higher instability -> higher suspiciousness",
        "color": "#F06449",
    },
]


def main() -> None:
    y = np.array([1, 0], dtype=float)

    fig = plt.figure(figsize=(13.5, 4.8), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 2.4, 1.8], wspace=0.02)

    ax_left = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[0, 1], sharey=ax_left)
    ax_right = fig.add_subplot(gs[0, 2], sharey=ax_left)

    for ax in [ax_left, ax_right]:
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 1.5)
        ax.axis("off")

    # Left text panel
    for yy, row in zip(y, ROWS):
        ax_left.text(
            0.98,
            yy + 0.10,
            row["task"],
            ha="right",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=row["color"],
        )
        ax_left.text(
            0.98,
            yy - 0.02,
            row["predictor"],
            ha="right",
            va="top",
            fontsize=12.5,
            color="#243B53",
        )

    # Middle effect-size panel
    ax_mid.set_ylim(-0.5, 1.5)
    ax_mid.set_xlim(-0.30, 0.30)
    ax_mid.axvline(0, color="#748094", linewidth=1.4)
    ax_mid.set_yticks([])
    ax_mid.set_xlabel("standardized β", fontsize=12, color="#243B53", labelpad=10)
    ax_mid.set_xticks(np.arange(-0.3, 0.31, 0.1))
    ax_mid.tick_params(axis="x", labelsize=11, colors="#52667A")
    ax_mid.grid(axis="x", color="#D9E2EC", linewidth=1.0)
    ax_mid.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax_mid.spines[spine].set_visible(False)
    ax_mid.spines["bottom"].set_color("#7B8794")
    ax_mid.spines["bottom"].set_linewidth(1.2)

    for yy, row in zip(y, ROWS):
        beta = row["beta"]
        color = row["color"]
        ax_mid.barh(yy, beta, height=0.14, color=color, edgecolor="none", zorder=3)
        text_x = beta - 0.015 if beta < 0 else beta + 0.015
        ha = "right" if beta < 0 else "left"
        sign = "+" if beta > 0 else ""
        ax_mid.text(
            text_x,
            yy + 0.09,
            f"{sign}{beta:.2f}  {row['p_text']}",
            ha=ha,
            va="bottom",
            fontsize=12.5,
            fontweight="bold",
            color=color,
        )

    # Right text panel
    for yy, row in zip(y, ROWS):
        ax_right.text(
            0.02,
            yy + 0.12,
            row["outcome"],
            ha="left",
            va="bottom",
            fontsize=13.5,
            fontweight="bold",
            color="#1F3C5C",
        )
        ax_right.text(
            0.02,
            yy - 0.02,
            row["interpretation"],
            ha="left",
            va="top",
            fontsize=12,
            color="#3E4C59",
        )

    fig.suptitle(
        "Behavioral correlates of HiTOP symptom dimensions",
        fontsize=17,
        fontweight="bold",
        color="#102A43",
        y=0.97,
    )

    out_png = OUT_DIR / "figure_behavior_hitop_summary_improved.png"
    out_svg = OUT_DIR / "figure_behavior_hitop_summary_improved.svg"
    fig.savefig(out_png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(out_png)
    print(out_svg)


if __name__ == "__main__":
    main()
