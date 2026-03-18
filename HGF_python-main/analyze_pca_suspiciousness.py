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
OUT_DIR = BASE / "hitop" / "pca_suspiciousness"

POOLED_PREDICTORS = ["inferv_a_mean", "om_a", "abs_eps2_a_mean", "abs_eps3_a_mean", "eps3_a_mean"]
T2_PREDICTORS = ["inferv_a_mean", "om_a", "abs_eps2_a_mean", "ka_a"]


def load_hitop_long() -> pd.DataFrame:
    frames = []
    for timepoint, hitop_file in [("t1", "hitop_scales_T1.csv"), ("t2", "hitop_scales_T2.csv")]:
        hitop = pd.read_csv(HITOP_DIR / hitop_file)[["prolific_id", "hitop_mistrust_suspiciousness"]].copy()
        hitop["timepoint"] = timepoint
        frames.append(hitop)
    return pd.concat(frames, ignore_index=True)


def load_predictors() -> pd.DataFrame:
    params = pd.read_csv(PARAM_LONG)
    om_a = params.loc[params["parameter"] == "om_a", ["prolific_id", "round_name", "timepoint", "estimate"]].copy()
    om_a = om_a.rename(columns={"estimate": "om_a"})
    ka_a = params.loc[params["parameter"] == "ka_a", ["prolific_id", "round_name", "timepoint", "estimate"]].copy()
    ka_a = ka_a.rename(columns={"estimate": "ka_a"})

    states = pd.read_csv(STATE_WIDE)
    keep = states.loc[
        states["state_metric"].isin(["inferv_a_mean", "abs_eps2_a_mean", "abs_eps3_a_mean", "eps3_a_mean"])
    ].copy()
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

    merged = om_a.merge(ka_a, on=["prolific_id", "round_name", "timepoint"], how="inner")
    merged = merged.merge(keep, on=["prolific_id", "timepoint"], how="inner")
    return merged


def zscore_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        sd = out[col].std(ddof=0)
        out[f"{col}_z"] = 0.0 if sd == 0 else (out[col] - out[col].mean()) / sd
    return out


def fit_pca(df: pd.DataFrame, predictors: list[str], prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = zscore_frame(df, predictors)
    x = z[[f"{c}_z" for c in predictors]].to_numpy()
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(x).reshape(-1)
    out = z.copy()
    out[f"{prefix}_pc1"] = pc1
    out = zscore_frame(out, [f"{prefix}_pc1"])
    loadings = pd.DataFrame(
        {
            "predictor": predictors,
            "pc1_loading": pca.components_[0],
            "abs_loading": np.abs(pca.components_[0]),
            "explained_variance_ratio": pca.explained_variance_ratio_[0],
            "prefix": prefix,
        }
    )
    return out, loadings


def summarize_model(model, analysis_scope: str, predictor_name: str, n: int) -> dict:
    term = f"{predictor_name}_z"
    return {
        "analysis_scope": analysis_scope,
        "predictor_name": predictor_name,
        "n": int(n),
        "beta": float(model.params[term]),
        "se": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
        "main_timepoint_t2_beta": float(model.params["C(timepoint)[T.t2]"]) if "C(timepoint)[T.t2]" in model.params.index else np.nan,
        "main_timepoint_t2_p": float(model.pvalues["C(timepoint)[T.t2]"]) if "C(timepoint)[T.t2]" in model.params.index else np.nan,
        "r_squared": float(getattr(model, "rsquared", float("nan"))),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hitop = load_hitop_long()
    preds = load_predictors()
    merged = preds.merge(hitop, on=["prolific_id", "timepoint"], how="inner").dropna().copy()
    merged.to_csv(OUT_DIR / "merged_predictors_suspiciousness_long.csv", index=False)

    pooled_df, pooled_loadings = fit_pca(merged, POOLED_PREDICTORS, "pooled")
    pooled_df = zscore_frame(pooled_df, ["hitop_mistrust_suspiciousness", "pooled_pc1"])
    pooled_model = smf.ols(
        "hitop_mistrust_suspiciousness_z ~ pooled_pc1_z + C(timepoint)",
        data=pooled_df,
    ).fit(cov_type="cluster", cov_kwds={"groups": pooled_df["prolific_id"]})

    t2_df = merged.loc[merged["timepoint"] == "t2"].copy()
    t2_df, t2_loadings = fit_pca(t2_df, T2_PREDICTORS, "t2")
    t2_df = zscore_frame(t2_df, ["hitop_mistrust_suspiciousness", "t2_pc1"])
    t2_model = smf.ols("hitop_mistrust_suspiciousness_z ~ t2_pc1_z", data=t2_df).fit()

    pooled_loadings.to_csv(OUT_DIR / "pooled_pca_loadings.csv", index=False)
    t2_loadings.to_csv(OUT_DIR / "t2_pca_loadings.csv", index=False)
    pooled_df.to_csv(OUT_DIR / "pooled_pca_data.csv", index=False)
    t2_df.to_csv(OUT_DIR / "t2_pca_data.csv", index=False)

    summary = pd.DataFrame(
        [
            summarize_model(pooled_model, "pooled_clustered", "pooled_pc1", len(pooled_df)),
            summarize_model(t2_model, "t2", "t2_pc1", len(t2_df)),
        ]
    )
    summary.to_csv(OUT_DIR / "pca_model_summary.csv", index=False)

    report = {
        "pooled": {
            "predictors": POOLED_PREDICTORS,
            "explained_variance_ratio_pc1": float(pooled_loadings["explained_variance_ratio"].iloc[0]),
            "beta_pc1": float(summary.loc[summary["analysis_scope"] == "pooled_clustered", "beta"].iloc[0]),
            "p_pc1": float(summary.loc[summary["analysis_scope"] == "pooled_clustered", "p_value"].iloc[0]),
            "r_squared": float(summary.loc[summary["analysis_scope"] == "pooled_clustered", "r_squared"].iloc[0]),
        },
        "t2": {
            "predictors": T2_PREDICTORS,
            "explained_variance_ratio_pc1": float(t2_loadings["explained_variance_ratio"].iloc[0]),
            "beta_pc1": float(summary.loc[summary["analysis_scope"] == "t2", "beta"].iloc[0]),
            "p_pc1": float(summary.loc[summary["analysis_scope"] == "t2", "p_value"].iloc[0]),
            "r_squared": float(summary.loc[summary["analysis_scope"] == "t2", "r_squared"].iloc[0]),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(report, indent=2))

    shared_report = (
        "PCA composite report for HiTOP suspiciousness\n"
        f"Pooled PC1 used predictors: {', '.join(POOLED_PREDICTORS)}\n"
        f"Pooled PC1 explained variance: {report['pooled']['explained_variance_ratio_pc1']:.3f}\n"
        f"Pooled PC1 association with suspiciousness: beta={report['pooled']['beta_pc1']:.3f}, "
        f"p={report['pooled']['p_pc1']:.6f}, R2={report['pooled']['r_squared']:.3f}\n"
        f"T2 PC1 used predictors: {', '.join(T2_PREDICTORS)}\n"
        f"T2 PC1 explained variance: {report['t2']['explained_variance_ratio_pc1']:.3f}\n"
        f"T2 PC1 association with suspiciousness: beta={report['t2']['beta_pc1']:.3f}, "
        f"p={report['t2']['p_pc1']:.6f}, R2={report['t2']['r_squared']:.3f}\n"
    )
    (OUT_DIR / "shared_report.txt").write_text(shared_report)


if __name__ == "__main__":
    main()
