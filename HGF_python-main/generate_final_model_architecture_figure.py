from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


OUT_BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/final_model_architecture")


def hexagon(center, w=0.055, h=0.045):
    x, y = center
    return Polygon(
        [
            (x - w * 0.55, y),
            (x - w * 0.28, y + h),
            (x + w * 0.28, y + h),
            (x + w * 0.55, y),
            (x + w * 0.28, y - h),
            (x - w * 0.28, y - h),
        ],
        closed=True,
        facecolor="white",
        edgecolor="#2f2f2f",
        linewidth=1.2,
    )


def diamond(center, w=0.05, h=0.05):
    x, y = center
    return Polygon(
        [(x, y + h), (x + w, y), (x, y - h), (x - w, y)],
        closed=True,
        facecolor="white",
        edgecolor="#2f2f2f",
        linewidth=1.2,
    )


def arrow(ax, p1, p2, style="-|>", color="#2f2f2f", lw=1.2, ls="-", rad=0.0):
    a = FancyArrowPatch(
        p1,
        p2,
        arrowstyle=style,
        mutation_scale=11,
        linewidth=lw,
        linestyle=ls,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)


def label(ax, x, y, text, size=13, weight="normal", ha="center", va="center", color="#111111"):
    ax.text(x, y, text, fontsize=size, fontweight=weight, ha=ha, va=va, color=color, family="DejaVu Serif")


def main() -> None:
    fig = plt.figure(figsize=(14, 8))
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background panels
    perceptual = FancyBboxPatch((0.03, 0.44), 0.94, 0.50, boxstyle="round,pad=0.012,rounding_size=0.01",
                                facecolor="#f6f7f8", edgecolor="#88b7a3", linewidth=1.8)
    response = FancyBboxPatch((0.03, 0.08), 0.94, 0.30, boxstyle="round,pad=0.012,rounding_size=0.01",
                              facecolor="#f8f9fb", edgecolor="#8eaad6", linewidth=1.8)
    ax.add_patch(perceptual)
    ax.add_patch(response)

    label(ax, 0.50, 0.95, "Final Baseline-Fixed Model Architecture", size=20, weight="bold")
    label(ax, 0.50, 0.91, "3-level HGF + volatility-modulated choice/wager model",
          size=12, color="#444444")

    label(ax, 0.50, 0.88, "Perceptual Model", size=18, weight="bold")
    label(ax, 0.50, 0.36, "Response Model", size=18, weight="bold")

    # Left explanatory text
    label(ax, 0.06, 0.80, "Advice branch", size=14, weight="bold", ha="left")
    label(ax, 0.06, 0.74, r"$x_{3,a}$ volatility", size=13, ha="left")
    label(ax, 0.06, 0.67, r"$x_{2,a}$ advice tendency", size=13, ha="left")
    label(ax, 0.06, 0.60, r"$x_{1,a}$ advice outcome", size=13, ha="left")

    label(ax, 0.06, 0.52, "Rewarding spatial location branch", size=14, weight="bold", ha="left")
    label(ax, 0.06, 0.46, r"$x_{3,c}$ volatility", size=13, ha="left")
    label(ax, 0.06, 0.39, r"$x_{2,c}$ location tendency", size=13, ha="left")
    label(ax, 0.06, 0.32, r"$x_{1,c}$ location outcome", size=13, ha="left")

    # Advice branch nodes
    advice_x = 0.38
    reward_x = 0.57
    y3, y2, y1 = 0.80, 0.67, 0.54

    for cx, cy, txt in [(advice_x, y3, r"$x_{3,a}^{(k)}$"), (advice_x, y2, r"$x_{2,a}^{(k)}$")]:
        ax.add_patch(hexagon((cx, cy)))
        label(ax, cx, cy, txt, size=16)
    ax.add_patch(diamond((advice_x, y1)))
    label(ax, advice_x, y1, r"$x_{1,a}^{(k)}$", size=16)

    # Reward branch nodes
    for cx, cy, txt in [(reward_x, y3, r"$x_{3,c}^{(k)}$"), (reward_x, y2, r"$x_{2,c}^{(k)}$")]:
        ax.add_patch(hexagon((cx, cy)))
        label(ax, cx, cy, txt, size=16)
    ax.add_patch(diamond((reward_x, y1)))
    label(ax, reward_x, y1, r"$x_{1,c}^{(k)}$", size=16)

    # Prior/parameter circles
    for cx, cy, txt in [
        (0.26, y3, r"$\vartheta_a$"),
        (0.26, y2, r"$\kappa_a$"),
        (0.66, y3, r"$\vartheta_c$"),
        (0.66, y2, r"$\kappa_c$"),
    ]:
        ax.add_patch(Circle((cx, cy), 0.026, facecolor="white", edgecolor="#2f2f2f", linewidth=1.2))
        label(ax, cx, cy, txt, size=15)

    # Arrows perceptual
    arrow(ax, (0.286, y3), (0.335, y3))
    arrow(ax, (0.286, y2), (0.335, y2))
    arrow(ax, (0.634, y3), (0.585, y3))
    arrow(ax, (0.634, y2), (0.585, y2))
    arrow(ax, (advice_x, y3 - 0.045), (advice_x, y2 + 0.045))
    arrow(ax, (advice_x, y2 - 0.045), (advice_x, y1 + 0.05))
    arrow(ax, (reward_x, y3 - 0.045), (reward_x, y2 + 0.045))
    arrow(ax, (reward_x, y2 - 0.045), (reward_x, y1 + 0.05))

    # Response model center
    label(ax, 0.19, 0.29, r"Arbitration", size=15, weight="bold", ha="left")
    label(ax, 0.19, 0.25, r"$b^{(k)}=\xi_a^{(k)}\hat{\mu}_{1,a}^{(k)}+\xi_c^{(k)}\hat{\mu}_{1,c}^{(k)}$", size=15, ha="left")
    label(ax, 0.19, 0.18, r"Choice noise", size=15, weight="bold", ha="left")
    label(ax, 0.19, 0.14, r"$\beta_{choice}^{(k)}=\exp(-\hat{\mu}_{3,a}^{(k)}-\hat{\mu}_{3,c}^{(k)})/be_{ch}$", size=14, ha="left")

    # Response nodes
    y_choice = (0.48, 0.24)
    y_wager = (0.56, 0.24)
    ax.add_patch(diamond(y_choice, w=0.04, h=0.04))
    ax.add_patch(diamond(y_wager, w=0.04, h=0.04))
    label(ax, *y_choice, r"$y_{choice}^{(k)}$", size=14)
    label(ax, *y_wager, r"$y_{wager}^{(k)}$", size=14)

    arrow(ax, (advice_x + 0.025, y1 - 0.03), (0.47, 0.28), rad=-0.15, ls="--")
    arrow(ax, (reward_x - 0.025, y1 - 0.03), (0.49, 0.28), rad=0.12, ls="--")
    arrow(ax, (advice_x + 0.025, y1 - 0.03), (0.55, 0.28), rad=-0.05, ls="--")
    arrow(ax, (reward_x - 0.025, y1 - 0.03), (0.57, 0.28), rad=0.05, ls="--")

    # Observation equations
    label(ax, 0.72, 0.28, "Choice", size=15, weight="bold", ha="left")
    label(ax, 0.73, 0.24,
          r"$p(y_{choice}^{(k)}=1)=\dfrac{b^{(k)\beta_{choice}^{(k)}}}{b^{(k)\beta_{choice}^{(k)}}+(1-b^{(k)})^{\beta_{choice}^{(k)}}}$",
          size=13, ha="left")
    label(ax, 0.72, 0.17, "Wager", size=15, weight="bold", ha="left")
    label(ax, 0.73, 0.13,
          r"$y_{wager}^{(k)}=\beta_0+\beta_1\,surp^{(k)}+\beta_2\,\xi^{(k)}+\beta_3 I\!V_a^{(k)}+\beta_4 I\!V_c^{(k)}+\beta_5 P\!V_a^{(k)}+\beta_6 P\!V_c^{(k)}+\varepsilon$",
          size=11.5, ha="left")

    # Final model callout
    callout = FancyBboxPatch((0.70, 0.72), 0.25, 0.10, boxstyle="round,pad=0.02,rounding_size=0.01",
                             facecolor="#d9dde3", edgecolor="none")
    ax.add_patch(callout)
    label(ax, 0.825, 0.785, "Final model", size=13, weight="bold")
    label(ax, 0.825, 0.755, "3-level HGF + linear wager model", size=10.5)
    label(ax, 0.825, 0.728, "baseline-fixed perceptual priors", size=10.5)

    # Fixed-parameter note
    fixed_box = FancyBboxPatch((0.69, 0.48), 0.25, 0.13, boxstyle="round,pad=0.02,rounding_size=0.01",
                               facecolor="#eef0f2", edgecolor="#c0c5ca", linewidth=1.0)
    ax.add_patch(fixed_box)
    label(ax, 0.815, 0.585, "Fixed perceptual priors", size=12, weight="bold")
    label(ax, 0.815, 0.545, r"$\mu_{2,r,0},\ \mu_{2,a,0},\ \mu_{3,a,0},\ \omega_r,\ \sigma^2_{2,a,0}$", size=12)
    label(ax, 0.815, 0.505, "Other reported parameters remained free", size=10.5, color="#444444")

    # Bottom note
    label(ax, 0.50, 0.05,
          "Dashed arrows indicate that both branches contribute to arbitration, choice probability, and wager generation.",
          size=11, color="#444444")

    for ext in ["svg", "png", "tiff"]:
        fig.savefig(OUT_BASE.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
