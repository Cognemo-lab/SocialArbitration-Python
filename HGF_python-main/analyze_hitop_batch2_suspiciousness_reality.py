from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
MODEL_DIR = BASE / "hgf_batch2_t1_baselinefixed_fullfidelity"
STATE_DIR = MODEL_DIR / "state_summaries"
HITOP_DIR = BASE / "hitop_batch2" / "processed_data"
OUT_BASE = BASE / "hitop_batch2"

ASSOC_DIR = OUT_BASE / "model_associations_reliable"
SUSP_DIR = OUT_BASE / "results_suspiciousness"
SUSP_MULTI_DIR = OUT_BASE / "multivariate_suspiciousness"
REALITY_DIR = OUT_BASE / "results_reality_distortion"

RELIABLE_PARAMETERS = ["be0", "be2", "be_ch", "be_wager", "ka_a", "om_a"]
RELIABLE_STATES = ["abs_eps2_a_mean", "abs_eps3_a_mean", "eps3_a_mean", "inferv_a_mean", "wager_pred_mean"]

HITOP_TARGETS = [
    "hitop_mistrust_suspiciousness",
    "hitop_reality_distortion",
    "hitop_reality_distortion_delusions",
    "hitop_reality_distortion_hallucinations",
]

SUSP_PREDICTORS = ["inferv_a_mean", "om_a", "abs_eps2_a_mean", "abs_eps3_a_mean", "eps3_a_mean"]
REALITY_PREDICTORS = ["be_wager", "be_ch"]


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = p_values.astype(float)
    mask = p.notna()
    result = pd.Series(np.nan, index=p.index, dtype=float)
    if mask.sum() == 0:
        return result
    ranked = p[mask].sort_values()
    m = float(len(ranked))
    q = ranked * m / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q.iloc[::-1])[::-1].clip(upper=1.0)
    result.loc[q.index] = q
    return result


def corr_stats(x: pd.Series, y: pd.Series) -> dict[str, float]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(df)
    if n < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return {"n": n, "pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
    pr, pp = pearsonr(df["x"], df["y"])
    sr, sp = spearmanr(df["x"], df["y"])
    return {
        "n": int(n),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
    }


def zscore_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        sd = out[col].std(ddof=0)
        out[f"{col}_z"] = 0.0 if sd == 0 else (out[col] - out[col].mean()) / sd
    return out


def load_predictors() -> pd.DataFrame:
    params = pd.read_csv(MODEL_DIR / "parameter_estimates_long_t1_t2.csv")
    params = params[(~params["is_fixed"]) & (params["parameter"].isin(RELIABLE_PARAMETERS))].copy()
    param_wide = params.pivot_table(
        index=["prolific_id", "timepoint"],
        columns="parameter",
        values="estimate",
        aggfunc="first",
    ).reset_index()
    param_wide.columns.name = None

    states = pd.read_csv(STATE_DIR / "state_estimates_wide.csv")
    keep = ["prolific_id", "timepoint"] + [c for c in RELIABLE_STATES if c in states.columns]
    states = states[keep].copy()
    return param_wide.merge(states, on=["prolific_id", "timepoint"], how="inner")


def load_hitop() -> pd.DataFrame:
    df = pd.read_csv(HITOP_DIR / "hitop_scales_T1.csv")
    df["timepoint"] = "t1"
    return df[["prolific_id", "timepoint"] + HITOP_TARGETS].copy()


def run_associations(predictors: pd.DataFrame, hitop: pd.DataFrame) -> pd.DataFrame:
    ASSOC_DIR.mkdir(parents=True, exist_ok=True)
    merged = hitop.merge(predictors, on=["prolific_id", "timepoint"], how="inner")
    merged.to_csv(ASSOC_DIR / "hitop_model_merged_long.csv", index=False)
    quantity_names = RELIABLE_PARAMETERS + RELIABLE_STATES
    quantity_kind = {name: "parameter" for name in RELIABLE_PARAMETERS}
    quantity_kind.update({name: "state" for name in RELIABLE_STATES})
    rows = []
    for target in HITOP_TARGETS:
        for q in quantity_names:
            stats = corr_stats(merged[target], merged[q])
            rows.append(
                {
                    "analysis_scope": "t1_only",
                    "hitop_measure": target,
                    "quantity_name": q,
                    "quantity_kind": quantity_kind[q],
                    **stats,
                }
            )
    assoc = pd.DataFrame(rows)
    assoc["fdr_q_pearson_within_scope"] = assoc.groupby("analysis_scope")["pearson_p"].transform(bh_fdr)
    assoc["fdr_q_spearman_within_scope"] = assoc.groupby("analysis_scope")["spearman_p"].transform(bh_fdr)
    assoc = assoc.sort_values(["hitop_measure", "quantity_kind", "quantity_name"]).reset_index(drop=True)
    assoc.to_csv(ASSOC_DIR / "hitop_reliable_quantity_associations.csv", index=False)
    assoc.reindex(assoc["pearson_r"].abs().sort_values(ascending=False).index).to_csv(
        ASSOC_DIR / "hitop_reliable_quantity_associations_ranked.csv", index=False
    )
    pd.DataFrame(
        {
            "quantity_name": quantity_names,
            "quantity_kind": [quantity_kind[q] for q in quantity_names],
        }
    ).to_csv(ASSOC_DIR / "selected_reliable_quantities.csv", index=False)
    summary = {
        "n_merged_rows": int(len(merged)),
        "n_unique_subjects_merged": int(merged["prolific_id"].nunique()),
        "selected_parameters": RELIABLE_PARAMETERS,
        "selected_states": RELIABLE_STATES,
    }
    (ASSOC_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    return merged


def fit_one_component_pca(df: pd.DataFrame, cols: list[str], prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def run_suspiciousness_pca(merged: pd.DataFrame) -> None:
    SUSP_DIR.mkdir(parents=True, exist_ok=True)
    df = merged[["prolific_id", "timepoint", "hitop_mistrust_suspiciousness"] + SUSP_PREDICTORS].dropna().copy()
    df.to_csv(SUSP_DIR / "merged_predictors_suspiciousness_long.csv", index=False)
    pca_df, loadings = fit_one_component_pca(df, SUSP_PREDICTORS, "pc")
    pca_df = zscore_frame(pca_df, ["hitop_mistrust_suspiciousness", "pc_pc1"])
    model = smf.ols("hitop_mistrust_suspiciousness_z ~ pc_pc1_z", data=pca_df).fit()
    loadings.to_csv(SUSP_DIR / "pooled_pca_loadings.csv", index=False)
    pca_df.to_csv(SUSP_DIR / "pooled_pca_data.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "analysis_scope": "t1_only",
                "predictor_name": "pc_pc1",
                "n": int(len(pca_df)),
                "beta": float(model.params["pc_pc1_z"]),
                "se": float(model.bse["pc_pc1_z"]),
                "p_value": float(model.pvalues["pc_pc1_z"]),
                "r_squared": float(model.rsquared),
            }
        ]
    )
    summary.to_csv(SUSP_DIR / "pca_model_summary.csv", index=False)
    report = {
        "t1": {
            "predictors": SUSP_PREDICTORS,
            "explained_variance_ratio_pc1": float(loadings["explained_variance_ratio"].iloc[0]),
            "beta_pc1": float(model.params["pc_pc1_z"]),
            "p_pc1": float(model.pvalues["pc_pc1_z"]),
            "r_squared": float(model.rsquared),
        }
    }
    (SUSP_DIR / "summary.json").write_text(json.dumps(report, indent=2))
    (SUSP_DIR / "shared_report.txt").write_text(
        "Batch 2 PCA composite report for HiTOP suspiciousness\n"
        f"T1 PC1 used predictors: {', '.join(SUSP_PREDICTORS)}\n"
        f"T1 PC1 explained variance: {report['t1']['explained_variance_ratio_pc1']:.3f}\n"
        f"T1 PC1 association with suspiciousness: beta={report['t1']['beta_pc1']:.3f}, "
        f"p={report['t1']['p_pc1']:.6f}, R2={report['t1']['r_squared']:.3f}\n"
    )


def run_suspiciousness_multivariate(merged: pd.DataFrame) -> None:
    SUSP_MULTI_DIR.mkdir(parents=True, exist_ok=True)
    df = merged[["prolific_id", "timepoint", "hitop_mistrust_suspiciousness"] + SUSP_PREDICTORS].dropna().copy()
    df = zscore_frame(df, ["hitop_mistrust_suspiciousness"] + SUSP_PREDICTORS)
    formula = "hitop_mistrust_suspiciousness_z ~ " + " + ".join(f"{p}_z" for p in SUSP_PREDICTORS)
    model = smf.ols(formula, data=df).fit()
    null_model = smf.ols("hitop_mistrust_suspiciousness_z ~ 1", data=df).fit()

    ci = model.conf_int()
    coef_rows = []
    for term in model.params.index:
        coef_rows.append(
            {
                "term": term,
                "coef": float(model.params[term]),
                "se": float(model.bse[term]),
                "stat": float(model.tvalues[term]),
                "p_value": float(model.pvalues[term]),
                "ci_low": float(ci.loc[term, 0]),
                "ci_high": float(ci.loc[term, 1]),
            }
        )
    pd.DataFrame(coef_rows).to_csv(SUSP_MULTI_DIR / "coefficients.csv", index=False)

    exog = df[[f"{p}_z" for p in SUSP_PREDICTORS]].copy()
    vif_df = pd.DataFrame(
        {
            "predictor": exog.columns,
            "vif": [float(variance_inflation_factor(exog.values, i)) for i in range(exog.shape[1])],
        }
    )
    vif_df.to_csv(SUSP_MULTI_DIR / "multicollinearity_vif.csv", index=False)

    f_test = model.compare_f_test(null_model)
    summary = pd.DataFrame(
        [
            {
                "analysis_scope": "t1_only",
                "n": int(len(df)),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "joint_f_statistic": float(f_test[0]),
                "joint_p_value": float(f_test[1]),
                "joint_df_diff": float(f_test[2]),
            }
        ]
    )
    summary.to_csv(SUSP_MULTI_DIR / "model_summary.csv", index=False)
    (SUSP_MULTI_DIR / "summary.json").write_text(
        json.dumps(
            {
                "predictors": SUSP_PREDICTORS,
                "n": int(len(df)),
                "r_squared": float(model.rsquared),
                "joint_f_statistic": float(f_test[0]),
                "joint_p_value": float(f_test[1]),
            },
            indent=2,
        )
    )


def run_reality_pca(merged: pd.DataFrame) -> None:
    REALITY_DIR.mkdir(parents=True, exist_ok=True)
    df = merged[["prolific_id", "timepoint"] + HITOP_TARGETS[1:] + REALITY_PREDICTORS].dropna().copy()
    df.to_csv(REALITY_DIR / "merged_predictors_reality_distortion_long.csv", index=False)
    out_df, out_load = fit_one_component_pca(df, HITOP_TARGETS[1:], "rd")
    pred_df, pred_load = fit_one_component_pca(out_df, REALITY_PREDICTORS, "pred")
    pred_df.to_csv(REALITY_DIR / "pca_data.csv", index=False)
    out_load.to_csv(REALITY_DIR / "reality_distortion_pca_loadings.csv", index=False)
    pred_load.to_csv(REALITY_DIR / "predictor_pca_loadings.csv", index=False)
    model1 = smf.ols("rd_pc1_z ~ be_wager_z + be_ch_z", data=pred_df).fit()
    model2 = smf.ols("rd_pc1_z ~ pred_pc1_z", data=pred_df).fit()
    coef_rows = []
    for scope, model in [("raw_predictors", model1), ("predictor_pc1", model2)]:
        ci = model.conf_int()
        for term in model.params.index:
            coef_rows.append(
                {
                    "analysis_scope": scope,
                    "term": term,
                    "coef": float(model.params[term]),
                    "se": float(model.bse[term]),
                    "stat": float(model.tvalues[term]),
                    "p_value": float(model.pvalues[term]),
                    "ci_low": float(ci.loc[term, 0]),
                    "ci_high": float(ci.loc[term, 1]),
                }
            )
    pd.DataFrame(coef_rows).to_csv(REALITY_DIR / "coefficients.csv", index=False)
    pd.DataFrame(
        [
            {"model": "raw_predictors", "n": int(len(pred_df)), "r_squared": float(model1.rsquared), "adj_r_squared": float(model1.rsquared_adj)},
            {"model": "predictor_pc1", "n": int(len(pred_df)), "r_squared": float(model2.rsquared), "adj_r_squared": float(model2.rsquared_adj)},
        ]
    ).to_csv(REALITY_DIR / "model_summary.csv", index=False)
    report = {
        "reality_distortion_outcome_pc1_explained_variance": float(out_load["explained_variance_ratio"].iloc[0]),
        "predictor_pc1_explained_variance": float(pred_load["explained_variance_ratio"].iloc[0]),
        "raw_predictor_model_r_squared": float(model1.rsquared),
        "predictor_pc1_model_r_squared": float(model2.rsquared),
        "predictor_pc1_beta": float(model2.params["pred_pc1_z"]),
        "predictor_pc1_p": float(model2.pvalues["pred_pc1_z"]),
    }
    (REALITY_DIR / "summary.json").write_text(json.dumps(report, indent=2))
    (REALITY_DIR / "shared_report.txt").write_text(
        "Batch 2 PCA composite report for HiTOP reality distortion\n"
        f"Outcome PC1 used variables: {', '.join(HITOP_TARGETS[1:])}\n"
        f"Outcome PC1 explained variance: {out_load['explained_variance_ratio'].iloc[0]:.3f}\n"
        f"Predictor PC1 used variables: {', '.join(REALITY_PREDICTORS)}\n"
        f"Predictor PC1 explained variance: {pred_load['explained_variance_ratio'].iloc[0]:.3f}\n"
        f"Predictor PC1 model beta: {model2.params['pred_pc1_z']:.3f}, "
        f"p={model2.pvalues['pred_pc1_z']:.6f}, R2={model2.rsquared:.3f}\n"
    )


def main() -> None:
    predictors = load_predictors()
    hitop = load_hitop()
    merged = run_associations(predictors, hitop)
    run_suspiciousness_pca(merged)
    run_suspiciousness_multivariate(merged)
    run_reality_pca(merged)


if __name__ == "__main__":
    main()
