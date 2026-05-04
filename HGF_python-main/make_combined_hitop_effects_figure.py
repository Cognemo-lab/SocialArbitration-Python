from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
SUSPICIOUSNESS_FIG = BASE / "hitop" / "results_suspiciousness" / "figure_pca_suspiciousness_paper.png"
REALITY_FIG = BASE / "hitop" / "results_reality_distortion" / "figure_pca_reality_distortion_paper.png"
OUT_DIR = BASE / "hitop" / "paper_joint_summary"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_suspiciousness = mpimg.imread(SUSPICIOUSNESS_FIG)
    img_reality = mpimg.imread(REALITY_FIG)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.8))

    axes[0].imshow(img_suspiciousness)
    axes[0].axis("off")
    axes[0].set_title(
        "A. HiTOP Suspiciousness\n"
        "Strong shared advice-uncertainty / advice-learning signal",
        fontsize=13,
        pad=14,
    )

    axes[1].imshow(img_reality)
    axes[1].axis("off")
    axes[1].set_title(
        "B. HiTOP Reality Distortion\n"
        "Weaker effect, driven mainly by wager and choice terms",
        fontsize=13,
        pad=14,
    )

    fig.suptitle("Model-Based Correlates of Suspiciousness and Reality Distortion", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.5)

    fig.savefig(OUT_DIR / "figure_hitop_suspiciousness_reality_distortion_combined.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
