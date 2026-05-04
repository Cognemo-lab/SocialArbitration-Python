from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
STATE_WIDE = BASE / "hgf_baselinefixed_full" / "state_reliability" / "state_estimates_wide_t1_t2.csv"
HITOP_DIR = BASE / "hitop" / "processed_data"
OUT_DIR = BASE / "hitop" / "paper_moderation_summary"

STATE_METRICS = [
    "inferv_a_mean",
    "eps3_a_mean",
    "abs_eps2_a_mean",
]

STATE_LABELS = {
    "inferv_a_mean": "Advice inferential variance",
    "eps3_a_mean": "Advice epsilon3",
    "abs_eps2_a_mean": "Absolute advice epsilon2",
}


def load_clinical_long() -> pd.DataFrame:
    frames = []
    for timepoint, hitop_file, gcsq_file in [
        ("t1", "hitop_scales_T1.csv", "additional_suicidality_scales_T1.csv"),
        ("t2", "hitop_scales_T2.csv", "additional_suicidality_scales_T2.csv"),
    ]:
        hitop = pd.read_csv(HITOP_DIR / hitop_file)[["prolific_id", "hitop_suicidality"]].copy()
        gcsq = pd.read_csv(HITOP_DIR / gcsq_file)[["prolific_id", "gcsq_fearlesness_of_death"]].copy()
        merged = hitop.merge(gcsq, on="prolific_id", how="inner")
        merged["timepoint"] = timepoint
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def load_state_long(metric: str) -> pd.DataFrame:
    state = pd.read_csv(STATE_WIDE)
    df = state.loc[state["state_metric"] == metric].copy()
    df = df.melt(
        id_vars=["prolific_id", "state_metric"],
        value_vars=["t1", "t2"],
        var_name="timepoint",
        value_name=metric,
    ).drop(columns=["state_metric"])
    return df


def zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[f"{col}_z"] = (out[col] - out[col].mean()) / out[col].std(ddof=0)
    return out


def fit_models(metric: str, clinical: pd.DataFrame) -> tuple[pd.DataFrame, object, object]:
    state = load_state_long(metric)
    df = state.merge(clinical, on=["prolific_id", "timepoint"], how="inner").dropna().copy()
    df = zscore(df, [metric, "hitop_suicidality", "gcsq_fearlesness_of_death"])
    pooled = smf.ols(
        f"gcsq_fearlesness_of_death_z ~ {metric}_z * hitop_suicidality_z + C(timepoint)",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["prolific_id"]})

    subj = (
        df.groupby("prolific_id")[[metric, "hitop_suicidality", "gcsq_fearlesness_of_death"]]
        .mean(numeric_only=True)
        .dropna()
        .reset_index()
    )
    subj = zscore(subj, [metric, "hitop_suicidality", "gcsq_fearlesness_of_death"])
    subject_mean = smf.ols(
        f"gcsq_fearlesness_of_death_z ~ {metric}_z * hitop_suicidality_z",
        data=subj,
    ).fit()
    return df, pooled, subject_mean


def coefficient_row(metric: str, pooled, subject_mean, n_rows: int, n_subjects: int) -> dict:
    term = f"{metric}_z:hitop_suicidality_z"
    return {
        "state_metric": metric,
        "state_label": STATE_LABELS[metric],
        "n_rows_pooled": n_rows,
        "n_subjects": n_subjects,
        "pooled_interaction_beta": float(pooled.params[term]),
        "pooled_interaction_se": float(pooled.bse[term]),
        "pooled_interaction_p": float(pooled.pvalues[term]),
        "pooled_main_state_beta": float(pooled.params[f"{metric}_z"]),
        "pooled_main_state_p": float(pooled.pvalues[f"{metric}_z"]),
        "pooled_main_suicidality_beta": float(pooled.params["hitop_suicidality_z"]),
        "pooled_main_suicidality_p": float(pooled.pvalues["hitop_suicidality_z"]),
        "subject_mean_interaction_beta": float(subject_mean.params[term]),
        "subject_mean_interaction_se": float(subject_mean.bse[term]),
        "subject_mean_interaction_p": float(subject_mean.pvalues[term]),
    }


def simple_slope_rows(metric: str, pooled, subject_mean) -> list[dict]:
    rows = []
    term = f"{metric}_z"
    inter = f"{metric}_z:hitop_suicidality_z"
    for model_name, model in [("pooled_clustered", pooled), ("subject_mean", subject_mean)]:
        for mod_z, level in [(-1.0, "Low suicidality (-1 SD)"), (0.0, "Mean suicidality"), (1.0, "High suicidality (+1 SD)")]:
            rows.append(
                {
                    "state_metric": metric,
                    "state_label": STATE_LABELS[metric],
                    "model": model_name,
                    "moderator_level": level,
                    "suicidality_z": mod_z,
                    "simple_slope": float(model.params[term] + model.params[inter] * mod_z),
                }
            )
    return rows


def make_figure(plot_data: list[tuple[str, pd.DataFrame, object]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
    colors = {
        "Low suicidality (-1 SD)": "#2a9d8f",
        "Mean suicidality": "#264653",
        "High suicidality (+1 SD)": "#e76f51",
    }
    moderator_levels = [(-1.0, "Low suicidality (-1 SD)"), (0.0, "Mean suicidality"), (1.0, "High suicidality (+1 SD)")]

    for ax, (metric, df, pooled) in zip(axes, plot_data):
        x = df[f"{metric}_z"]
        y = df["gcsq_fearlesness_of_death_z"]
        ax.scatter(x, y, s=14, alpha=0.25, color="#5a6c84", edgecolor="none")
        grid = np.linspace(x.min(), x.max(), 120)
        b = pooled.params
        for mod_z, label in moderator_levels:
            pred = (
                b["Intercept"]
                + b.get("C(timepoint)[T.t2]", 0.0) * 0.5
                + b[f"{metric}_z"] * grid
                + b["hitop_suicidality_z"] * mod_z
                + b[f"{metric}_z:hitop_suicidality_z"] * grid * mod_z
            )
            ax.plot(grid, pred, lw=2, color=colors[label], label=label)
        p_int = pooled.pvalues[f"{metric}_z:hitop_suicidality_z"]
        beta_int = pooled.params[f"{metric}_z:hitop_suicidality_z"]
        ax.set_title(f"{STATE_LABELS[metric]}\ninteraction beta={beta_int:.3f}, p={p_int:.3f}")
        ax.set_xlabel("State summary (z)")
        ax.grid(alpha=0.15)

    axes[0].set_ylabel("Fearlessness of death (z)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT_DIR / "figure_combined_state_moderation.png", dpi=250)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clinical = load_clinical_long()

    summary_rows = []
    slope_rows = []
    plot_data = []

    for metric in STATE_METRICS:
        df, pooled, subject_mean = fit_models(metric, clinical)
        df.to_csv(OUT_DIR / f"merged_{metric}_long.csv", index=False)
        summary_rows.append(coefficient_row(metric, pooled, subject_mean, len(df), df["prolific_id"].nunique()))
        slope_rows.extend(simple_slope_rows(metric, pooled, subject_mean))
        plot_data.append((metric, df, pooled))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "table_moderation_summary.csv", index=False)

    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(OUT_DIR / "table_moderation_simple_slopes.csv", index=False)

    make_figure(plot_data)


if __name__ == "__main__":
    main()
