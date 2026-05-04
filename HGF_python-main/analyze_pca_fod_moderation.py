from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
PARAM_LONG = BASE / "hgf_baselinefixed_full" / "parameter_estimates_long_t1_t2.csv"
STATE_WIDE = BASE / "hgf_baselinefixed_full" / "state_reliability" / "state_estimates_wide_t1_t2.csv"
HITOP_DIR = BASE / "hitop" / "processed_data"
OUT_DIR = BASE / "hitop" / "pca_fod_moderation"

PREDICTORS = ["om_a", "inferv_a_mean", "abs_eps2_a_mean"]


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


def load_predictors() -> pd.DataFrame:
    params = pd.read_csv(PARAM_LONG)
    om_a = params.loc[params["parameter"] == "om_a", ["prolific_id", "round_name", "timepoint", "estimate"]].copy()
    om_a = om_a.rename(columns={"estimate": "om_a"})

    states = pd.read_csv(STATE_WIDE)
    keep = states.loc[states["state_metric"].isin(["inferv_a_mean", "abs_eps2_a_mean"])].copy()
    keep = keep.melt(
        id_vars=["prolific_id", "state_metric"],
        value_vars=["t1", "t2"],
        var_name="timepoint",
        value_name="estimate",
    )
    keep = keep.pivot_table(
        index=["prolific_id", "timepoint"],
        columns="state_metric",
        values="estimate",
        aggfunc="first",
    ).reset_index()
    keep.columns.name = None
    return om_a.merge(keep, on=["prolific_id", "timepoint"], how="inner")


def zscore_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        sd = out[col].std(ddof=0)
        out[f"{col}_z"] = 0.0 if sd == 0 else (out[col] - out[col].mean()) / sd
    return out


def add_pc1(df: pd.DataFrame) -> tuple[pd.DataFrame, PCA]:
    z = zscore_frame(df, PREDICTORS)
    x = z[[f"{c}_z" for c in PREDICTORS]].to_numpy()
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(x).reshape(-1)
    out = z.copy()
    out["pc1_composite"] = pc1
    out = zscore_frame(out, ["pc1_composite"])
    return out, pca


def summarize_model(model, analysis_scope: str, n: int) -> dict:
    term = "pc1_composite_z:hitop_suicidality_z"
    return {
        "analysis_scope": analysis_scope,
        "n": int(n),
        "interaction_beta": float(model.params[term]),
        "interaction_se": float(model.bse[term]),
        "interaction_p": float(model.pvalues[term]),
        "main_pc1_beta": float(model.params["pc1_composite_z"]),
        "main_pc1_p": float(model.pvalues["pc1_composite_z"]),
        "main_suicidality_beta": float(model.params["hitop_suicidality_z"]),
        "main_suicidality_p": float(model.pvalues["hitop_suicidality_z"]),
        "r_squared": float(getattr(model, "rsquared", float("nan"))),
    }


def simple_slopes(model, analysis_scope: str) -> list[dict]:
    rows = []
    for z, label in [(-1.0, "low"), (0.0, "mean"), (1.0, "high")]:
        rows.append(
            {
                "analysis_scope": analysis_scope,
                "moderator_level": label,
                "suicidality_z": z,
                "simple_slope_pc1": float(
                    model.params["pc1_composite_z"] + model.params["pc1_composite_z:hitop_suicidality_z"] * z
                ),
            }
        )
    return rows


def fit_model(df: pd.DataFrame, include_timepoint: bool, clustered: bool):
    formula = "gcsq_fearlesness_of_death_z ~ pc1_composite_z * hitop_suicidality_z"
    if include_timepoint:
        formula += " + C(timepoint)"
    if clustered:
        return smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["prolific_id"]})
    return smf.ols(formula, data=df).fit()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clinical = load_clinical_long()
    preds = load_predictors()
    merged = preds.merge(clinical, on=["prolific_id", "timepoint"], how="inner").dropna().copy()

    pc_data, pca = add_pc1(merged)
    pc_data.to_csv(OUT_DIR / "pca_merged_long.csv", index=False)

    loadings = pd.DataFrame(
        {
            "predictor": PREDICTORS,
            "pc1_loading": pca.components_[0],
            "abs_loading": np.abs(pca.components_[0]),
            "explained_variance_ratio": pca.explained_variance_ratio_[0],
        }
    )
    loadings.to_csv(OUT_DIR / "pca_loadings.csv", index=False)

    summary_rows = []
    slope_rows = []

    pooled_model = fit_model(zscore_frame(pc_data, ["hitop_suicidality", "gcsq_fearlesness_of_death", "pc1_composite"]), True, True)
    summary_rows.append(summarize_model(pooled_model, "pooled_clustered", len(pc_data)))
    slope_rows.extend(simple_slopes(pooled_model, "pooled_clustered"))

    subj = (
        pc_data.groupby("prolific_id")[["pc1_composite", "hitop_suicidality", "gcsq_fearlesness_of_death"]]
        .mean(numeric_only=True)
        .dropna()
        .reset_index()
    )
    subj = zscore_frame(subj, ["pc1_composite", "hitop_suicidality", "gcsq_fearlesness_of_death"])
    subj.to_csv(OUT_DIR / "pca_subject_mean.csv", index=False)
    subj_model = fit_model(subj, False, False)
    summary_rows.append(summarize_model(subj_model, "subject_mean", len(subj)))
    slope_rows.extend(simple_slopes(subj_model, "subject_mean"))

    for tp in ["t1", "t2"]:
        df = pc_data.loc[pc_data["timepoint"] == tp].copy()
        df = zscore_frame(df, ["pc1_composite", "hitop_suicidality", "gcsq_fearlesness_of_death"])
        model = fit_model(df, False, False)
        summary_rows.append(summarize_model(model, tp, len(df)))
        slope_rows.extend(simple_slopes(model, tp))

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "pca_moderation_summary.csv", index=False)
    pd.DataFrame(slope_rows).to_csv(OUT_DIR / "pca_simple_slopes.csv", index=False)

    payload = {
        "predictors": PREDICTORS,
        "explained_variance_ratio_pc1": float(pca.explained_variance_ratio_[0]),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
