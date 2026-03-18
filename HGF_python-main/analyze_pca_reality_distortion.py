from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
PARAM_LONG = BASE / "hgf_baselinefixed_full" / "parameter_estimates_long_t1_t2.csv"
HITOP_DIR = BASE / "hitop" / "processed_data"
OUT_DIR = BASE / "hitop" / "pca_reality_distortion"

OUTCOMES = [
    "hitop_reality_distortion",
    "hitop_reality_distortion_delusions",
    "hitop_reality_distortion_hallucinations",
]
PREDICTORS = ["be_wager", "be_ch"]


def load_hitop_long() -> pd.DataFrame:
    frames = []
    for timepoint, hitop_file in [("t1", "hitop_scales_T1.csv"), ("t2", "hitop_scales_T2.csv")]:
        hitop = pd.read_csv(HITOP_DIR / hitop_file)[["prolific_id"] + OUTCOMES].copy()
        hitop["timepoint"] = timepoint
        frames.append(hitop)
    return pd.concat(frames, ignore_index=True)


def load_predictors() -> pd.DataFrame:
    params = pd.read_csv(PARAM_LONG)
    out = []
    for name in PREDICTORS:
        df = params.loc[params["parameter"] == name, ["prolific_id", "round_name", "timepoint", "estimate"]].copy()
        df = df.rename(columns={"estimate": name})
        out.append(df)
    merged = out[0]
    for df in out[1:]:
        merged = merged.merge(df, on=["prolific_id", "round_name", "timepoint"], how="inner")
    return merged


def zscore_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        sd = out[col].std(ddof=0)
        out[f"{col}_z"] = 0.0 if sd == 0 else (out[col] - out[col].mean()) / sd
    return out


def fit_pca(df: pd.DataFrame, cols: list[str], prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = zscore_frame(df, cols)
    x = z[[f"{c}_z" for c in cols]].to_numpy()
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(x).reshape(-1)
    out = z.copy()
    out[f"{prefix}_pc1"] = pc1
    out = zscore_frame(out, [f"{prefix}_pc1"])
    loadings = pd.DataFrame(
        {
            "variable": cols,
            "pc1_loading": pca.components_[0],
            "abs_loading": np.abs(pca.components_[0]),
            "explained_variance_ratio": pca.explained_variance_ratio_[0],
            "prefix": prefix,
        }
    )
    return out, loadings


def coef_table(model, analysis_scope: str) -> pd.DataFrame:
    ci = model.conf_int()
    rows = []
    for term in model.params.index:
        rows.append(
            {
                "analysis_scope": analysis_scope,
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hitop = load_hitop_long()
    preds = load_predictors()
    merged = preds.merge(hitop, on=["prolific_id", "timepoint"], how="inner").dropna().copy()
    merged.to_csv(OUT_DIR / "merged_predictors_reality_distortion_long.csv", index=False)

    out_df, out_load = fit_pca(merged, OUTCOMES, "rd")
    pred_df, pred_load = fit_pca(out_df, PREDICTORS, "pred")

    pred_df.to_csv(OUT_DIR / "pca_data.csv", index=False)
    out_load.to_csv(OUT_DIR / "reality_distortion_pca_loadings.csv", index=False)
    pred_load.to_csv(OUT_DIR / "predictor_pca_loadings.csv", index=False)

    model1 = smf.ols("rd_pc1_z ~ be_wager_z + be_ch_z + C(timepoint)", data=pred_df).fit(
        cov_type="cluster", cov_kwds={"groups": pred_df["prolific_id"]}
    )
    model2 = smf.ols("rd_pc1_z ~ pred_pc1_z + C(timepoint)", data=pred_df).fit(
        cov_type="cluster", cov_kwds={"groups": pred_df["prolific_id"]}
    )

    coef = pd.concat(
        [coef_table(model1, "raw_predictors"), coef_table(model2, "predictor_pc1")],
        ignore_index=True,
    )
    coef.to_csv(OUT_DIR / "coefficients.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "model": "raw_predictors",
                "n": int(len(pred_df)),
                "r_squared": float(model1.rsquared),
                "adj_r_squared": float(model1.rsquared_adj),
            },
            {
                "model": "predictor_pc1",
                "n": int(len(pred_df)),
                "r_squared": float(model2.rsquared),
                "adj_r_squared": float(model2.rsquared_adj),
            },
        ]
    )
    summary.to_csv(OUT_DIR / "model_summary.csv", index=False)

    report = {
        "reality_distortion_outcome_pc1_explained_variance": float(out_load["explained_variance_ratio"].iloc[0]),
        "predictor_pc1_explained_variance": float(pred_load["explained_variance_ratio"].iloc[0]),
        "raw_predictor_model_r_squared": float(model1.rsquared),
        "predictor_pc1_model_r_squared": float(model2.rsquared),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(report, indent=2))

    shared_report = (
        "PCA composite report for HiTOP reality distortion\n"
        f"Outcome PC1 used variables: {', '.join(OUTCOMES)}\n"
        f"Outcome PC1 explained variance: {out_load['explained_variance_ratio'].iloc[0]:.3f}\n"
        f"Predictor PC1 used variables: {', '.join(PREDICTORS)}\n"
        f"Predictor PC1 explained variance: {pred_load['explained_variance_ratio'].iloc[0]:.3f}\n"
        f"Raw predictor model R2: {model1.rsquared:.3f}\n"
        f"Predictor PC1 model beta: {model2.params['pred_pc1_z']:.3f}, "
        f"p={model2.pvalues['pred_pc1_z']:.6f}, R2={model2.rsquared:.3f}\n"
    )
    (OUT_DIR / "shared_report.txt").write_text(shared_report)


if __name__ == "__main__":
    main()
