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
OUT_DIR = BASE / "hitop" / "inferv_a_fearlessness_moderation"


def load_long_data() -> pd.DataFrame:
    state = pd.read_csv(STATE_WIDE)
    inferv = state.loc[state["state_metric"] == "inferv_a_mean"].copy()
    inferv = inferv.melt(
        id_vars=["prolific_id", "state_metric"],
        value_vars=["t1", "t2"],
        var_name="timepoint",
        value_name="inferv_a_mean",
    ).drop(columns=["state_metric"])

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

    clinical = pd.concat(frames, ignore_index=True)
    df = inferv.merge(clinical, on=["prolific_id", "timepoint"], how="inner").dropna().copy()

    for col in ["inferv_a_mean", "hitop_suicidality", "gcsq_fearlesness_of_death"]:
        df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    return df


def simple_slopes(beta_x: float, beta_int: float, moderator_z: float) -> float:
    return beta_x + beta_int * moderator_z


def summarize_model(model, label: str) -> pd.DataFrame:
    ci = model.conf_int()
    rows = []
    for term in model.params.index:
        rows.append(
            {
                "model": label,
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


def make_plot(df: pd.DataFrame, pooled_model) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    x = df["inferv_a_mean_z"]
    y = df["gcsq_fearlesness_of_death_z"]
    ax.scatter(x, y, s=18, alpha=0.35, color="#4c6a92", edgecolor="none")

    grid = np.linspace(x.min(), x.max(), 100)
    moderators = [(-1.0, "Low suicidality (-1 SD)", "#2a9d8f"), (0.0, "Mean suicidality", "#264653"), (1.0, "High suicidality (+1 SD)", "#e76f51")]

    b = pooled_model.params
    for mod_z, label, color in moderators:
        pred = (
            b["Intercept"]
            + b.get("C(timepoint)[T.t2]", 0.0) * 0.5
            + b["inferv_a_mean_z"] * grid
            + b["hitop_suicidality_z"] * mod_z
            + b["inferv_a_mean_z:hitop_suicidality_z"] * grid * mod_z
        )
        ax.plot(grid, pred, lw=2, color=color, label=label)

    ax.set_xlabel("inferv_a_mean (z)")
    ax.set_ylabel("GCSQ fearlessness of death (z)")
    ax.set_title("Fearlessness of Death vs inferv_a_mean\nModerated by HiTOP suicidality")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_inferv_suicidality_moderation.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_long_data()
    df.to_csv(OUT_DIR / "merged_inferv_hitop_gcsq_long.csv", index=False)

    pooled = smf.ols(
        "gcsq_fearlesness_of_death_z ~ inferv_a_mean_z * hitop_suicidality_z + C(timepoint)",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["prolific_id"]})

    subj = (
        df.groupby("prolific_id")[["inferv_a_mean", "hitop_suicidality", "gcsq_fearlesness_of_death"]]
        .mean(numeric_only=True)
        .dropna()
        .reset_index()
    )
    for col in ["inferv_a_mean", "hitop_suicidality", "gcsq_fearlesness_of_death"]:
        subj[f"{col}_z"] = (subj[col] - subj[col].mean()) / subj[col].std(ddof=0)
    subj.to_csv(OUT_DIR / "merged_inferv_hitop_gcsq_subject_mean.csv", index=False)

    subj_model = smf.ols(
        "gcsq_fearlesness_of_death_z ~ inferv_a_mean_z * hitop_suicidality_z",
        data=subj,
    ).fit()

    coeffs = pd.concat(
        [
            summarize_model(pooled, "pooled_clustered"),
            summarize_model(subj_model, "subject_mean"),
        ],
        ignore_index=True,
    )
    coeffs.to_csv(OUT_DIR / "moderation_model_coefficients.csv", index=False)

    pooled_slopes = pd.DataFrame(
        [
            {
                "model": "pooled_clustered",
                "moderator_level": label,
                "suicidality_z": z,
                "simple_slope_inferv": simple_slopes(
                    pooled.params["inferv_a_mean_z"],
                    pooled.params["inferv_a_mean_z:hitop_suicidality_z"],
                    z,
                ),
            }
            for z, label in [(-1.0, "low"), (0.0, "mean"), (1.0, "high")]
        ]
    )
    subj_slopes = pd.DataFrame(
        [
            {
                "model": "subject_mean",
                "moderator_level": label,
                "suicidality_z": z,
                "simple_slope_inferv": simple_slopes(
                    subj_model.params["inferv_a_mean_z"],
                    subj_model.params["inferv_a_mean_z:hitop_suicidality_z"],
                    z,
                ),
            }
            for z, label in [(-1.0, "low"), (0.0, "mean"), (1.0, "high")]
        ]
    )
    slopes = pd.concat([pooled_slopes, subj_slopes], ignore_index=True)
    slopes.to_csv(OUT_DIR / "moderation_simple_slopes.csv", index=False)

    summary = {
        "n_rows_pooled": int(len(df)),
        "n_subjects_pooled": int(df["prolific_id"].nunique()),
        "n_subjects_subject_mean": int(len(subj)),
        "pooled_clustered_interaction_beta": float(
            pooled.params["inferv_a_mean_z:hitop_suicidality_z"]
        ),
        "pooled_clustered_interaction_p": float(
            pooled.pvalues["inferv_a_mean_z:hitop_suicidality_z"]
        ),
        "pooled_clustered_main_inferv_beta": float(pooled.params["inferv_a_mean_z"]),
        "pooled_clustered_main_suicidality_beta": float(
            pooled.params["hitop_suicidality_z"]
        ),
        "subject_mean_interaction_beta": float(
            subj_model.params["inferv_a_mean_z:hitop_suicidality_z"]
        ),
        "subject_mean_interaction_p": float(
            subj_model.pvalues["inferv_a_mean_z:hitop_suicidality_z"]
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    make_plot(df, pooled)


if __name__ == "__main__":
    main()
