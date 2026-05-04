from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
ASSOC_CSV = BASE / "hitop" / "model_associations_reliable" / "hitop_reliable_quantity_associations.csv"
PARAM_LONG = BASE / "hgf_baselinefixed_full" / "parameter_estimates_long_t1_t2.csv"
STATE_WIDE = BASE / "hgf_baselinefixed_full" / "state_reliability" / "state_estimates_wide_t1_t2.csv"
HITOP_DIR = BASE / "hitop" / "processed_data"
OUT_DIR = BASE / "hitop" / "consistent_domain_analysis"

OUTCOMES = [
    "hitop_mistrust_suspiciousness",
    "hitop_reality_distortion",
    "hitop_reality_distortion_delusions",
    "hitop_reality_distortion_hallucinations",
]


def zscore_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        sd = out[col].std(ddof=0)
        out[f"{col}_z"] = 0.0 if sd == 0 or pd.isna(sd) else (out[col] - out[col].mean()) / sd
    return out


def load_predictors_long() -> pd.DataFrame:
    params = pd.read_csv(PARAM_LONG)
    params = params[["prolific_id", "round_name", "timepoint", "parameter", "estimate"]].copy()
    params["quantity_kind"] = "parameter"
    params = params.rename(columns={"parameter": "quantity_name"})

    states = pd.read_csv(STATE_WIDE)
    states = states.melt(
        id_vars=["prolific_id", "state_metric"],
        value_vars=["t1", "t2"],
        var_name="timepoint",
        value_name="estimate",
    )
    states["round_name"] = states["timepoint"].map({"t1": "round1", "t2": "round2"})
    states["quantity_kind"] = "state"
    states = states.rename(columns={"state_metric": "quantity_name"})
    states = states[["prolific_id", "round_name", "timepoint", "quantity_name", "estimate", "quantity_kind"]]

    return pd.concat([params, states], ignore_index=True)


def load_hitop_long() -> pd.DataFrame:
    frames = []
    for timepoint, hitop_file in [("t1", "hitop_scales_T1.csv"), ("t2", "hitop_scales_T2.csv")]:
        cols = ["prolific_id"] + OUTCOMES
        hitop = pd.read_csv(HITOP_DIR / hitop_file)[cols].copy()
        hitop["timepoint"] = timepoint
        frames.append(hitop)
    return pd.concat(frames, ignore_index=True)


def select_predictors(assoc: pd.DataFrame, outcome: str, scope: str) -> list[str]:
    sub = assoc[
        (assoc["hitop_measure"] == outcome)
        & (assoc["analysis_scope"] == scope)
        & (assoc["fdr_q_pearson_within_scope"] < 0.05)
    ].copy()
    if sub.empty:
        return []
    sub["abs_r"] = sub["pearson_r"].abs()
    sub = sub.sort_values(["abs_r", "quantity_name"], ascending=[False, True], kind="stable")
    return sub["quantity_name"].tolist()


def wide_from_predictors(pred_long: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    sub = pred_long[pred_long["quantity_name"].isin(predictors)].copy()
    wide = sub.pivot_table(
        index=["prolific_id", "round_name", "timepoint"],
        columns="quantity_name",
        values="estimate",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def coef_table(model, analysis_scope: str, outcome: str) -> pd.DataFrame:
    ci = model.conf_int()
    rows = []
    for term in model.params.index:
        rows.append(
            {
                "outcome": outcome,
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


def vif_table(df: pd.DataFrame, predictors: list[str], analysis_scope: str, outcome: str) -> pd.DataFrame:
    cols = [f"{p}_z" for p in predictors]
    if len(cols) < 2:
        return pd.DataFrame(
            [{"outcome": outcome, "analysis_scope": analysis_scope, "variable": cols[0], "vif": np.nan}]
        )
    x = df[cols].copy()
    rows = []
    for i, col in enumerate(cols):
        rows.append(
            {
                "outcome": outcome,
                "analysis_scope": analysis_scope,
                "variable": col,
                "vif": float(variance_inflation_factor(x.values, i)),
            }
        )
    return pd.DataFrame(rows)


def run_joint_test(model, predictors: list[str], analysis_scope: str, outcome: str) -> dict:
    constraints = [f"{p}_z = 0" for p in predictors]
    if not constraints:
        return {
            "outcome": outcome,
            "analysis_scope": analysis_scope,
            "test": "all_predictors_zero",
            "statistic": np.nan,
            "df_denom": np.nan,
            "df_num": np.nan,
            "p_value": np.nan,
        }
    wt = model.wald_test(constraints)
    return {
        "outcome": outcome,
        "analysis_scope": analysis_scope,
        "test": "all_predictors_zero",
        "statistic": float(wt.statistic),
        "df_denom": float(getattr(wt, "df_denom", float("nan"))),
        "df_num": float(getattr(wt, "df_num", len(constraints))),
        "p_value": float(wt.pvalue),
    }


def run_multivariate(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    analysis_scope: str,
    include_timepoint: bool,
    clustered: bool,
):
    work = zscore_frame(df, predictors + [outcome])
    formula = f"{outcome}_z ~ " + " + ".join([f"{p}_z" for p in predictors])
    if include_timepoint:
        formula += " + C(timepoint)"
    if clustered:
        model = smf.ols(formula, data=work).fit(cov_type="cluster", cov_kwds={"groups": work["prolific_id"]})
    else:
        model = smf.ols(formula, data=work).fit()
    return {
        "summary": {
            "outcome": outcome,
            "analysis_scope": analysis_scope,
            "n": int(len(work)),
            "n_predictors": int(len(predictors)),
            "predictors": ", ".join(predictors),
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
        },
        "coef": coef_table(model, analysis_scope, outcome),
        "vif": vif_table(work, predictors, analysis_scope, outcome),
        "joint": run_joint_test(model, predictors, analysis_scope, outcome),
    }


def fit_pc1(df: pd.DataFrame, predictors: list[str], prefix: str):
    if len(predictors) == 1:
        out = zscore_frame(df, predictors)
        out[f"{prefix}_pc1"] = out[f"{predictors[0]}_z"]
        out = zscore_frame(out, [f"{prefix}_pc1"])
        loadings = pd.DataFrame(
            {
                "predictor": predictors,
                "pc1_loading": [1.0],
                "abs_loading": [1.0],
                "explained_variance_ratio": [1.0],
                "prefix": prefix,
            }
        )
        return out, loadings

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


def run_pca_model(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    analysis_scope: str,
    include_timepoint: bool,
    clustered: bool,
    prefix: str,
):
    work, loadings = fit_pc1(df, predictors, prefix)
    work = zscore_frame(work, [outcome, f"{prefix}_pc1"])
    formula = f"{outcome}_z ~ {prefix}_pc1_z"
    if include_timepoint:
        formula += " + C(timepoint)"
    if clustered:
        model = smf.ols(formula, data=work).fit(cov_type="cluster", cov_kwds={"groups": work["prolific_id"]})
    else:
        model = smf.ols(formula, data=work).fit()
    return {
        "summary": {
            "outcome": outcome,
            "analysis_scope": analysis_scope,
            "n": int(len(work)),
            "n_predictors": int(len(predictors)),
            "predictors": ", ".join(predictors),
            "explained_variance_ratio_pc1": float(loadings["explained_variance_ratio"].iloc[0]),
            "beta_pc1": float(model.params[f"{prefix}_pc1_z"]),
            "se_pc1": float(model.bse[f"{prefix}_pc1_z"]),
            "p_pc1": float(model.pvalues[f"{prefix}_pc1_z"]),
            "r_squared": float(model.rsquared),
        },
        "loadings": loadings.assign(outcome=outcome, analysis_scope=analysis_scope),
        "data": work.assign(outcome=outcome, analysis_scope=analysis_scope),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    assoc = pd.read_csv(ASSOC_CSV)
    pred_long = load_predictors_long()
    hitop_long = load_hitop_long()

    all_selection_rows = []
    multivar_summary_rows = []
    multivar_coef_rows = []
    multivar_vif_rows = []
    multivar_joint_rows = []
    pca_summary_rows = []
    pca_loading_rows = []

    for outcome in OUTCOMES:
        pooled_predictors = select_predictors(assoc, outcome, "pooled_t1_t2")
        t2_predictors = select_predictors(assoc, outcome, "t2")

        all_selection_rows.extend(
            [
                {"outcome": outcome, "analysis_scope": "pooled_t1_t2", "predictor": p, "rank": i + 1}
                for i, p in enumerate(pooled_predictors)
            ]
        )
        all_selection_rows.extend(
            [{"outcome": outcome, "analysis_scope": "t2", "predictor": p, "rank": i + 1} for i, p in enumerate(t2_predictors)]
        )

        if pooled_predictors:
            pooled_wide = wide_from_predictors(pred_long, pooled_predictors)
            pooled_df = pooled_wide.merge(
                hitop_long[["prolific_id", "timepoint", outcome]],
                on=["prolific_id", "timepoint"],
                how="inner",
            ).dropna()
            multivar = run_multivariate(
                pooled_df,
                outcome,
                pooled_predictors,
                analysis_scope="pooled_clustered",
                include_timepoint=True,
                clustered=True,
            )
            multivar_summary_rows.append(multivar["summary"])
            multivar_coef_rows.append(multivar["coef"])
            multivar_vif_rows.append(multivar["vif"])
            multivar_joint_rows.append(multivar["joint"])

            pca = run_pca_model(
                pooled_df,
                outcome,
                pooled_predictors,
                analysis_scope="pooled_clustered",
                include_timepoint=True,
                clustered=True,
                prefix="pooled",
            )
            pca_summary_rows.append(pca["summary"])
            pca_loading_rows.append(pca["loadings"])

        if t2_predictors:
            t2_wide = wide_from_predictors(pred_long, t2_predictors)
            t2_df = t2_wide.merge(
                hitop_long[["prolific_id", "timepoint", outcome]],
                on=["prolific_id", "timepoint"],
                how="inner",
            )
            t2_df = t2_df[t2_df["timepoint"] == "t2"].dropna()
            if len(t2_df):
                multivar = run_multivariate(
                    t2_df,
                    outcome,
                    t2_predictors,
                    analysis_scope="t2",
                    include_timepoint=False,
                    clustered=False,
                )
                multivar_summary_rows.append(multivar["summary"])
                multivar_coef_rows.append(multivar["coef"])
                multivar_vif_rows.append(multivar["vif"])
                multivar_joint_rows.append(multivar["joint"])

                pca = run_pca_model(
                    t2_df,
                    outcome,
                    t2_predictors,
                    analysis_scope="t2",
                    include_timepoint=False,
                    clustered=False,
                    prefix="t2",
                )
                pca_summary_rows.append(pca["summary"])
                pca_loading_rows.append(pca["loadings"])

    selection_df = pd.DataFrame(all_selection_rows)
    selection_df.to_csv(OUT_DIR / "selected_predictors_by_outcome.csv", index=False)
    pd.DataFrame(multivar_summary_rows).to_csv(OUT_DIR / "multivariate_model_summary.csv", index=False)
    pd.concat(multivar_coef_rows, ignore_index=True).to_csv(OUT_DIR / "multivariate_coefficients.csv", index=False)
    pd.concat(multivar_vif_rows, ignore_index=True).to_csv(OUT_DIR / "multicollinearity_vif.csv", index=False)
    pd.DataFrame(multivar_joint_rows).to_csv(OUT_DIR / "multivariate_joint_wald_tests.csv", index=False)
    pd.DataFrame(pca_summary_rows).to_csv(OUT_DIR / "pca_model_summary.csv", index=False)
    pd.concat(pca_loading_rows, ignore_index=True).to_csv(OUT_DIR / "pca_loadings.csv", index=False)

    report = {
        "outcomes": OUTCOMES,
        "pooled_predictors_by_outcome": {
            outcome: select_predictors(assoc, outcome, "pooled_t1_t2") for outcome in OUTCOMES
        },
        "t2_predictors_by_outcome": {outcome: select_predictors(assoc, outcome, "t2") for outcome in OUTCOMES},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
