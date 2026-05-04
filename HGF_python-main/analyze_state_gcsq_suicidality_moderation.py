from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
STATE_WIDE = BASE / "hgf_baselinefixed_full" / "state_reliability" / "state_estimates_wide_t1_t2.csv"
HITOP_DIR = BASE / "hitop" / "processed_data"
OUT_DIR = BASE / "hitop" / "state_fearlessness_moderation"

STATE_METRICS = [
    "wager_pred_mean",
    "eps3_a_mean",
    "abs_eps2_a_mean",
]


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


def load_state_long(state_metric: str) -> pd.DataFrame:
    state = pd.read_csv(STATE_WIDE)
    df = state.loc[state["state_metric"] == state_metric].copy()
    df = df.melt(
        id_vars=["prolific_id", "state_metric"],
        value_vars=["t1", "t2"],
        var_name="timepoint",
        value_name=state_metric,
    ).drop(columns=["state_metric"])
    return df


def zscore_inplace(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)


def summarize_model(model, state_metric: str, model_name: str) -> pd.DataFrame:
    ci = model.conf_int()
    rows = []
    for term in model.params.index:
        rows.append(
            {
                "state_metric": state_metric,
                "model": model_name,
                "term": term,
                "coef": float(model.params[term]),
                "se": float(model.bse[term]),
                "stat": float(model.tvalues[term]),
                "p_value": float(model.pvalues[term]),
                "ci_low": float(ci.loc[term, 0]),
                "ci_high": float(ci.loc[term, 1]),
            }
        )
    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame, state_metric: str, pooled_model) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    x = df[f"{state_metric}_z"]
    y = df["gcsq_fearlesness_of_death_z"]
    ax.scatter(x, y, s=18, alpha=0.35, color="#4c6a92", edgecolor="none")

    grid = np.linspace(x.min(), x.max(), 100)
    moderators = [
        (-1.0, "Low suicidality (-1 SD)", "#2a9d8f"),
        (0.0, "Mean suicidality", "#264653"),
        (1.0, "High suicidality (+1 SD)", "#e76f51"),
    ]

    b = pooled_model.params
    for mod_z, label, color in moderators:
        pred = (
            b["Intercept"]
            + b.get("C(timepoint)[T.t2]", 0.0) * 0.5
            + b[f"{state_metric}_z"] * grid
            + b["hitop_suicidality_z"] * mod_z
            + b[f"{state_metric}_z:hitop_suicidality_z"] * grid * mod_z
        )
        ax.plot(grid, pred, lw=2, color=color, label=label)

    ax.set_xlabel(f"{state_metric} (z)")
    ax.set_ylabel("GCSQ fearlessness of death (z)")
    ax.set_title(f"Fearlessness of Death vs {state_metric}\nModerated by HiTOP suicidality")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"figure_{state_metric}_moderation.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clinical = load_clinical_long()

    all_coeffs = []
    all_slopes = []
    summaries = []

    for state_metric in STATE_METRICS:
        state_long = load_state_long(state_metric)
        df = state_long.merge(clinical, on=["prolific_id", "timepoint"], how="inner").dropna().copy()
        zscore_inplace(df, [state_metric, "hitop_suicidality", "gcsq_fearlesness_of_death"])
        df.to_csv(OUT_DIR / f"merged_{state_metric}_long.csv", index=False)

        pooled = smf.ols(
            f"gcsq_fearlesness_of_death_z ~ {state_metric}_z * hitop_suicidality_z + C(timepoint)",
            data=df,
        ).fit(cov_type="cluster", cov_kwds={"groups": df["prolific_id"]})

        subj = (
            df.groupby("prolific_id")[[state_metric, "hitop_suicidality", "gcsq_fearlesness_of_death"]]
            .mean(numeric_only=True)
            .dropna()
            .reset_index()
        )
        zscore_inplace(subj, [state_metric, "hitop_suicidality", "gcsq_fearlesness_of_death"])
        subj.to_csv(OUT_DIR / f"merged_{state_metric}_subject_mean.csv", index=False)

        subj_model = smf.ols(
            f"gcsq_fearlesness_of_death_z ~ {state_metric}_z * hitop_suicidality_z",
            data=subj,
        ).fit()

        all_coeffs.append(summarize_model(pooled, state_metric, "pooled_clustered"))
        all_coeffs.append(summarize_model(subj_model, state_metric, "subject_mean"))

        for model_name, model in [("pooled_clustered", pooled), ("subject_mean", subj_model)]:
            beta_x = model.params[f"{state_metric}_z"]
            beta_int = model.params[f"{state_metric}_z:hitop_suicidality_z"]
            for z, label in [(-1.0, "low"), (0.0, "mean"), (1.0, "high")]:
                all_slopes.append(
                    {
                        "state_metric": state_metric,
                        "model": model_name,
                        "moderator_level": label,
                        "suicidality_z": z,
                        "simple_slope_state": float(beta_x + beta_int * z),
                    }
                )

        summaries.append(
            {
                "state_metric": state_metric,
                "n_rows_pooled": int(len(df)),
                "n_subjects_pooled": int(df["prolific_id"].nunique()),
                "n_subjects_subject_mean": int(len(subj)),
                "pooled_clustered_interaction_beta": float(
                    pooled.params[f"{state_metric}_z:hitop_suicidality_z"]
                ),
                "pooled_clustered_interaction_p": float(
                    pooled.pvalues[f"{state_metric}_z:hitop_suicidality_z"]
                ),
                "pooled_clustered_main_state_beta": float(pooled.params[f"{state_metric}_z"]),
                "pooled_clustered_main_suicidality_beta": float(
                    pooled.params["hitop_suicidality_z"]
                ),
                "subject_mean_interaction_beta": float(
                    subj_model.params[f"{state_metric}_z:hitop_suicidality_z"]
                ),
                "subject_mean_interaction_p": float(
                    subj_model.pvalues[f"{state_metric}_z:hitop_suicidality_z"]
                ),
            }
        )

        make_plot(df, state_metric, pooled)

    coeffs = pd.concat(all_coeffs, ignore_index=True)
    coeffs.to_csv(OUT_DIR / "moderation_model_coefficients.csv", index=False)

    slopes = pd.DataFrame(all_slopes)
    slopes.to_csv(OUT_DIR / "moderation_simple_slopes.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "moderation_summary.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
