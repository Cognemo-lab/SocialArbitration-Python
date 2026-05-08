from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
OUT_DIR = ROOT / "hitop_cross_dataset_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


SUSPICIOUSNESS_PREDICTORS = [
    "inferv_a_mean",
    "om_a",
    "abs_eps2_a_mean",
    "abs_eps3_a_mean",
    "eps3_a_mean",
]

REALITY_PREDICTORS = ["be_wager", "be_ch"]


def zscore_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    scaler = StandardScaler()
    out = df.copy()
    out[cols] = scaler.fit_transform(out[cols])
    return out


def fit_pc1_effect(
    df: pd.DataFrame,
    predictor_cols: List[str],
    outcome_col: str,
    dataset_label: str,
    outcome_label: str,
) -> Tuple[dict, pd.DataFrame]:
    cols = predictor_cols + [outcome_col]
    work = df[cols].dropna().reset_index(drop=True).copy()
    work = zscore_frame(work, cols)

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(work[predictor_cols]).ravel()

    # Orient PC1 so higher values match the dominant loading direction.
    loadings = pd.Series(pca.components_[0], index=predictor_cols)
    if loadings.sum() < 0:
        pc1 *= -1
        loadings *= -1

    y = work[outcome_col]
    X = sm.add_constant(pd.DataFrame({"pc1": pc1}))
    model = sm.OLS(y, X).fit()
    beta = float(model.params["pc1"])
    se = float(model.bse["pc1"])
    ci_low, ci_high = model.conf_int().loc["pc1"].tolist()
    p_value = float(model.pvalues["pc1"])

    summary_row = {
        "dataset": dataset_label,
        "outcome": outcome_label,
        "n": int(len(work)),
        "beta": beta,
        "se": se,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
        "r_squared": float(model.rsquared),
        "pc1_variance_explained": float(pca.explained_variance_ratio_[0]),
    }
    loading_rows = pd.DataFrame(
        {
            "dataset": dataset_label,
            "outcome": outcome_label,
            "predictor": predictor_cols,
            "loading": loadings.values,
        }
    )
    return summary_row, loading_rows


def format_p(p_value: float) -> str:
    if p_value < 0.001:
        return "p<.001"
    return f"p={p_value:.3f}".replace("0.", ".")


def plot_effects(summary_df: pd.DataFrame, out_path: Path) -> None:
    outcome_order = [
        "Suspiciousness",
        "Overall reality distortion",
        "Delusions",
        "Hallucinations",
    ]
    dataset_order = ["Dataset 1", "Dataset 2"]
    colors = {"Dataset 1": "#2A6F97", "Dataset 2": "#C65D3B"}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True)
    axes = axes.flatten()

    for ax, outcome in zip(axes, outcome_order):
        subset = summary_df.loc[summary_df["outcome"] == outcome].copy()
        subset["dataset"] = pd.Categorical(subset["dataset"], dataset_order)
        subset = subset.sort_values("dataset")

        y_positions = np.array([1, 0])
        ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1)

        for y, (_, row) in zip(y_positions, subset.iterrows()):
            ax.errorbar(
                row["beta"],
                y,
                xerr=[[row["beta"] - row["ci_low"]], [row["ci_high"] - row["beta"]]],
                fmt="o",
                color=colors[row["dataset"]],
                ecolor=colors[row["dataset"]],
                elinewidth=2,
                capsize=4,
                markersize=8,
            )
            ax.text(
                row["ci_high"] + 0.025,
                y,
                f"{row['dataset']}: β={row['beta']:.2f}, {format_p(row['p_value'])}",
                va="center",
                ha="left",
                fontsize=9,
                color=colors[row["dataset"]],
            )

        ax.set_title(outcome, fontsize=12, pad=8)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(dataset_order, fontsize=10)
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlim(-0.4, 0.2)

        var_lines = []
        for _, row in subset.iterrows():
            var_lines.append(
                f"{row['dataset']} PC1 var={row['pc1_variance_explained']:.2f}, R²={row['r_squared']:.3f}"
            )
        ax.text(
            0.02,
            -0.30,
            "\n".join(var_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#444444",
        )

    for ax in axes[2:]:
        ax.set_xlabel("Standardized PCA composite effect (β)", fontsize=11)

    fig.suptitle(
        "HiTOP Associations: Dataset 1 vs Dataset 2\nLatent PCA composite effects at T1",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.2, w_pad=2.0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    original_susp = pd.read_csv(
        ROOT / "hitop/results_suspiciousness/merged_predictors_suspiciousness_long.csv"
    )
    original_real = pd.read_csv(
        ROOT / "hitop/results_reality_distortion/merged_predictors_reality_distortion_long.csv"
    )
    batch2_susp = pd.read_csv(
        ROOT / "hitop_batch2/results_suspiciousness/merged_predictors_suspiciousness_long.csv"
    )
    batch2_real = pd.read_csv(
        ROOT / "hitop_batch2/results_reality_distortion/merged_predictors_reality_distortion_long.csv"
    )

    original_susp = original_susp.loc[original_susp["timepoint"] == "t1"].copy()
    original_real = original_real.loc[original_real["timepoint"] == "t1"].copy()
    batch2_susp = batch2_susp.loc[batch2_susp["timepoint"] == "t1"].copy()
    batch2_real = batch2_real.loc[batch2_real["timepoint"] == "t1"].copy()

    analyses = [
        (
            original_susp,
            SUSPICIOUSNESS_PREDICTORS,
            "hitop_mistrust_suspiciousness",
            "Dataset 1",
            "Suspiciousness",
        ),
        (
            batch2_susp,
            SUSPICIOUSNESS_PREDICTORS,
            "hitop_mistrust_suspiciousness",
            "Dataset 2",
            "Suspiciousness",
        ),
        (
            original_real,
            REALITY_PREDICTORS,
            "hitop_reality_distortion",
            "Dataset 1",
            "Overall reality distortion",
        ),
        (
            batch2_real,
            REALITY_PREDICTORS,
            "hitop_reality_distortion",
            "Dataset 2",
            "Overall reality distortion",
        ),
        (
            original_real,
            REALITY_PREDICTORS,
            "hitop_reality_distortion_delusions",
            "Dataset 1",
            "Delusions",
        ),
        (
            batch2_real,
            REALITY_PREDICTORS,
            "hitop_reality_distortion_delusions",
            "Dataset 2",
            "Delusions",
        ),
        (
            original_real,
            REALITY_PREDICTORS,
            "hitop_reality_distortion_hallucinations",
            "Dataset 1",
            "Hallucinations",
        ),
        (
            batch2_real,
            REALITY_PREDICTORS,
            "hitop_reality_distortion_hallucinations",
            "Dataset 2",
            "Hallucinations",
        ),
    ]

    summary_rows = []
    loading_frames = []
    for args in analyses:
        summary_row, loading_df = fit_pc1_effect(*args)
        summary_rows.append(summary_row)
        loading_frames.append(loading_df)

    summary_df = pd.DataFrame(summary_rows)
    loadings_df = pd.concat(loading_frames, ignore_index=True)

    summary_path = OUT_DIR / "hitop_dataset1_vs_dataset2_pca_composites.csv"
    loadings_path = OUT_DIR / "hitop_dataset1_vs_dataset2_pca_composite_loadings.csv"
    figure_path = OUT_DIR / "figure_hitop_dataset1_vs_dataset2_pca_composites.png"

    summary_df.to_csv(summary_path, index=False)
    loadings_df.to_csv(loadings_path, index=False)
    plot_effects(summary_df, figure_path)

    print(f"Wrote {summary_path}")
    print(f"Wrote {loadings_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
