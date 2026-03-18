from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
PARAM_LONG = BASE / "hgf_baselinefixed_full" / "parameter_estimates_long_t1_t2.csv"
STATE_WIDE = BASE / "hgf_baselinefixed_full" / "state_reliability" / "state_estimates_wide_t1_t2.csv"
HITOP_DIR = BASE / "hitop" / "processed_data"
OUT_DIR = BASE / "hitop" / "multivariate_suspiciousness"

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


def vif_table(df: pd.DataFrame, predictors: list[str], analysis_scope: str) -> pd.DataFrame:
    cols = [f"{p}_z" for p in predictors]
    x = df[cols].copy()
    rows = []
    for i, col in enumerate(cols):
        rows.append(
            {
                "analysis_scope": analysis_scope,
                "variable": col,
                "vif": float(variance_inflation_factor(x.values, i)),
            }
        )
    return pd.DataFrame(rows)


def run_joint_test(model, predictors: list[str], analysis_scope: str) -> dict:
    constraints = [f"{p}_z = 0" for p in predictors]
    wt = model.wald_test(constraints)
    return {
        "analysis_scope": analysis_scope,
        "test": "all_predictors_zero",
        "statistic": float(wt.statistic),
        "df_denom": float(getattr(wt, "df_denom", float("nan"))),
        "df_num": float(getattr(wt, "df_num", len(constraints))),
        "p_value": float(wt.pvalue),
    }


def run_model(df: pd.DataFrame, predictors: list[str], analysis_scope: str, include_timepoint: bool, clustered: bool):
    formula = "hitop_mistrust_suspiciousness_z ~ " + " + ".join([f"{p}_z" for p in predictors])
    if include_timepoint:
        formula += " + C(timepoint)"
    if clustered:
        model = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["prolific_id"]})
    else:
        model = smf.ols(formula, data=df).fit()
    return {
        "summary": {
            "analysis_scope": analysis_scope,
            "n": int(len(df)),
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
        },
        "coef": coef_table(model, analysis_scope),
        "vif": vif_table(df, predictors, analysis_scope),
        "joint": run_joint_test(model, predictors, analysis_scope),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hitop = load_hitop_long()
    preds = load_predictors()
    merged = preds.merge(hitop, on=["prolific_id", "timepoint"], how="inner").dropna().copy()
    merged.to_csv(OUT_DIR / "merged_predictors_suspiciousness_long.csv", index=False)

    pooled = zscore_frame(merged, POOLED_PREDICTORS + ["hitop_mistrust_suspiciousness"])
    pooled_res = run_model(pooled, POOLED_PREDICTORS, "pooled_clustered", include_timepoint=True, clustered=True)

    t2 = merged.loc[merged["timepoint"] == "t2"].copy()
    t2 = zscore_frame(t2, T2_PREDICTORS + ["hitop_mistrust_suspiciousness"])
    t2_res = run_model(t2, T2_PREDICTORS, "t2", include_timepoint=False, clustered=False)

    pd.DataFrame([pooled_res["summary"], t2_res["summary"]]).to_csv(OUT_DIR / "model_summary.csv", index=False)
    pd.concat([pooled_res["coef"], t2_res["coef"]], ignore_index=True).to_csv(OUT_DIR / "coefficients.csv", index=False)
    pd.concat([pooled_res["vif"], t2_res["vif"]], ignore_index=True).to_csv(OUT_DIR / "multicollinearity_vif.csv", index=False)
    pd.DataFrame([pooled_res["joint"], t2_res["joint"]]).to_csv(OUT_DIR / "joint_wald_tests.csv", index=False)

    payload = {
        "pooled_predictors": POOLED_PREDICTORS,
        "t2_predictors": T2_PREDICTORS,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
