from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
IN_FILE = BASE / "hgf_baselinefixed_full" / "exported_estimates" / "reliable_parameters_and_states_long.csv"
OUT_DIR = BASE / "hgf_baselinefixed_full" / "exploratory_multilevel_reliable_features"


def load_session_wide() -> pd.DataFrame:
    long_df = pd.read_csv(IN_FILE)
    wide = (
        long_df.pivot_table(
            index=["prolific_id", "round_name", "timepoint"],
            columns="estimate_name",
            values="estimate",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    return wide


def fit_pca(df: pd.DataFrame, feature_cols: list[str], prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    x = scaler.fit_transform(df[feature_cols])
    n_comp = min(len(feature_cols), x.shape[0])
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(x)

    explained = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(n_comp)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained": np.cumsum(pca.explained_variance_ratio_),
            "analysis_level": prefix,
        }
    )
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f"PC{i+1}" for i in range(n_comp)],
    ).reset_index().rename(columns={"index": "feature"})
    loadings["analysis_level"] = prefix

    score_df = pd.DataFrame(scores[:, : min(3, n_comp)], columns=[f"PC{i+1}" for i in range(min(3, n_comp))])
    meta_cols = [c for c in ["prolific_id", "round_name", "timepoint"] if c in df.columns]
    score_df = pd.concat([df[meta_cols].reset_index(drop=True), score_df], axis=1)
    score_df["analysis_level"] = prefix
    return explained, loadings, score_df


def make_figures(explained: pd.DataFrame, loadings: pd.DataFrame, scores: pd.DataFrame) -> None:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Scree plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, level in zip(axes, ["overall", "between_subject", "within_subject"]):
        sub = explained.loc[explained["analysis_level"] == level].copy()
        ax.plot(range(1, len(sub) + 1), sub["explained_variance_ratio"], marker="o", color="#3a7ca5")
        ax.set_title(level.replace("_", " ").title())
        ax.set_xlabel("Principal component")
        ax.set_ylim(0, max(0.6, explained["explained_variance_ratio"].max() * 1.1))
    axes[0].set_ylabel("Explained variance ratio")
    fig.suptitle("Reliable Feature PCA Scree Plots", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fig_dir / "figure_multilevel_pca_scree.png", dpi=220)
    plt.close(fig)

    # PC1 loadings
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, level in zip(axes, ["overall", "between_subject", "within_subject"]):
        sub = loadings.loc[loadings["analysis_level"] == level].copy().sort_values("PC1")
        colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in sub["PC1"]]
        ax.barh(sub["feature"], sub["PC1"], color=colors, alpha=0.9)
        ax.axvline(0, color="0.3", lw=1)
        ax.set_title(f"{level.replace('_', ' ').title()} PC1")
        ax.set_xlabel("Loading")
    fig.suptitle("Reliable Feature PC1 Loadings", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fig_dir / "figure_multilevel_pca_pc1_loadings.png", dpi=220)
    plt.close(fig)

    # Overall score scatter
    overall_scores = scores.loc[scores["analysis_level"] == "overall"].copy()
    if {"PC1", "PC2"}.issubset(overall_scores.columns):
        fig, ax = plt.subplots(figsize=(6, 5.5))
        colors = overall_scores["timepoint"].map({"t1": "#3a7ca5", "t2": "#e07a5f"}).fillna("#777777")
        ax.scatter(overall_scores["PC1"], overall_scores["PC2"], c=colors, alpha=0.65, s=24)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Overall PCA Scores by Session")
        fig.tight_layout()
        fig.savefig(fig_dir / "figure_multilevel_pca_scores_overall.png", dpi=220)
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    session_wide = load_session_wide()
    feature_cols = [c for c in session_wide.columns if c not in ["prolific_id", "round_name", "timepoint"]]
    feature_cols = [c for c in feature_cols if c not in ["be0", "abs_eps3_a_mean"]]
    session_wide = session_wide[["prolific_id", "round_name", "timepoint"] + feature_cols].copy()
    session_wide.to_csv(OUT_DIR / "reliable_features_session_wide.csv", index=False)

    overall_expl, overall_load, overall_scores = fit_pca(session_wide, feature_cols, "overall")

    subject_mean = session_wide.groupby("prolific_id")[feature_cols].mean(numeric_only=True).reset_index()
    between_expl, between_load, between_scores = fit_pca(subject_mean, feature_cols, "between_subject")

    complete = session_wide.groupby("prolific_id")["timepoint"].nunique()
    complete_ids = complete.loc[complete >= 2].index
    within = session_wide.loc[session_wide["prolific_id"].isin(complete_ids)].copy()
    subj_means = within.groupby("prolific_id")[feature_cols].transform("mean")
    within_centered = within.copy()
    within_centered[feature_cols] = within[feature_cols] - subj_means
    within_expl, within_load, within_scores = fit_pca(within_centered, feature_cols, "within_subject")

    explained = pd.concat([overall_expl, between_expl, within_expl], ignore_index=True)
    loadings = pd.concat([overall_load, between_load, within_load], ignore_index=True)
    scores = pd.concat([overall_scores, between_scores, within_scores], ignore_index=True, sort=False)

    explained.to_csv(OUT_DIR / "multilevel_pca_explained_variance.csv", index=False)
    loadings.to_csv(OUT_DIR / "multilevel_pca_loadings.csv", index=False)
    scores.to_csv(OUT_DIR / "multilevel_pca_scores.csv", index=False)

    summary = {
        "n_features": int(len(feature_cols)),
        "features": feature_cols,
        "n_sessions": int(len(session_wide)),
        "n_subjects": int(session_wide["prolific_id"].nunique()),
        "n_subjects_complete_t1_t2": int(len(complete_ids)),
        "overall_pc1_explained": float(overall_expl.loc[0, "explained_variance_ratio"]),
        "between_pc1_explained": float(between_expl.loc[0, "explained_variance_ratio"]),
        "within_pc1_explained": float(within_expl.loc[0, "explained_variance_ratio"]),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    shared_report = (
        "Exploratory multilevel analysis of reliable model features\n"
        f"Number of reliable features: {summary['n_features']}\n"
        f"Sessions analyzed: {summary['n_sessions']}\n"
        f"Subjects analyzed: {summary['n_subjects']}\n"
        f"Subjects with both sessions: {summary['n_subjects_complete_t1_t2']}\n"
        f"Overall PC1 explained variance: {summary['overall_pc1_explained']:.3f}\n"
        f"Between-subject PC1 explained variance: {summary['between_pc1_explained']:.3f}\n"
        f"Within-subject PC1 explained variance: {summary['within_pc1_explained']:.3f}\n"
    )
    (OUT_DIR / "shared_report.txt").write_text(shared_report)

    make_figures(explained, loadings, scores)


if __name__ == "__main__":
    main()
