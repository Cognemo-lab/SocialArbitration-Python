from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
PARAM_LONG = BASE / "hgf_baselinefixed_full" / "parameter_estimates_long_t1_t2.csv"
STATE_WIDE = BASE / "hgf_baselinefixed_full" / "state_reliability" / "state_estimates_wide_t1_t2.csv"
HITOP_T1 = BASE / "hitop" / "processed_data" / "hitop_scales_T1.csv"
HITOP_T2 = BASE / "hitop" / "processed_data" / "hitop_scales_T2.csv"
PFACTOR = (
    BASE
    / "behavior_fod_moderation_comparison"
    / "state_and_prevalence_assessment"
    / "hitop_bifactor_cfa"
    / "hitop_bifactor_cfa_scores.csv"
)
OUT_DIR = BASE / "hitop" / "consistent_domain_analysis" / "p_factor_adjustment"

OUTCOME = "hitop_mistrust_suspiciousness"
CURRENT_PREDICTORS = ["inferv_a_mean", "om_a", "abs_eps2_a_mean", "eps3_a_mean", "ka_a"]
LEGACY_PREDICTORS = CURRENT_PREDICTORS[:3] + ["abs_eps3_a_mean"] + CURRENT_PREDICTORS[3:]


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / sd


def load_data() -> pd.DataFrame:
    params = pd.read_csv(PARAM_LONG)
    param_wide = params.loc[
        params["parameter"].isin(["om_a", "ka_a"]),
        ["prolific_id", "round_name", "timepoint", "parameter", "estimate"],
    ].pivot_table(
        index=["prolific_id", "round_name", "timepoint"],
        columns="parameter",
        values="estimate",
        aggfunc="first",
    ).reset_index()
    param_wide.columns.name = None

    states = pd.read_csv(STATE_WIDE)
    states = states[states["state_metric"].isin(set(LEGACY_PREDICTORS) - {"om_a"})]
    states = states.melt(
        id_vars=["prolific_id", "state_metric"],
        value_vars=["t1", "t2"],
        var_name="timepoint",
        value_name="estimate",
    )
    states = states.pivot_table(
        index=["prolific_id", "timepoint"], columns="state_metric", values="estimate", aggfunc="first"
    ).reset_index()
    states.columns.name = None

    hitop_frames = []
    for timepoint, path in [("t1", HITOP_T1), ("t2", HITOP_T2)]:
        frame = pd.read_csv(path)[["prolific_id", OUTCOME]].copy()
        frame["timepoint"] = timepoint
        hitop_frames.append(frame)
    hitop = pd.concat(hitop_frames, ignore_index=True)

    scores = pd.read_csv(PFACTOR)
    scores = scores[scores["dataset"] == "Original"][["prolific_id", "timepoint", "p_factor"]].copy()

    for frame in (param_wide, states, hitop, scores):
        frame["prolific_id"] = frame["prolific_id"].astype(str)

    return param_wide.merge(states, on=["prolific_id", "timepoint"], how="inner").merge(
        hitop, on=["prolific_id", "timepoint"], how="inner"
    ).merge(scores, on=["prolific_id", "timepoint"], how="inner")


def fit_specification(data: pd.DataFrame, label: str, predictors: list[str]) -> tuple[dict, pd.DataFrame]:
    work = data[["prolific_id", "timepoint", OUTCOME, "p_factor", *predictors]].dropna().copy()
    standardized = pd.DataFrame(index=work.index)
    for predictor in predictors:
        standardized[predictor] = zscore(work[predictor])

    pca = PCA(n_components=1)
    work["pc1"] = pca.fit_transform(standardized[predictors]).reshape(-1)
    # Orient PC1 so higher scores represent the common positive advice-uncertainty direction.
    if pca.components_[0][predictors.index("inferv_a_mean")] < 0:
        work["pc1"] *= -1
        pca.components_[0] *= -1

    work["pc1_z"] = zscore(work["pc1"])
    work["outcome_z"] = zscore(work[OUTCOME])
    work["p_factor_z"] = zscore(work["p_factor"])

    base = smf.ols("outcome_z ~ pc1_z + C(timepoint)", data=work).fit(
        cov_type="cluster", cov_kwds={"groups": work["prolific_id"]}
    )
    adjusted = smf.ols("outcome_z ~ pc1_z + p_factor_z + C(timepoint)", data=work).fit(
        cov_type="cluster", cov_kwds={"groups": work["prolific_id"]}
    )
    pc1_p = smf.ols("p_factor_z ~ pc1_z + C(timepoint)", data=work).fit(
        cov_type="cluster", cov_kwds={"groups": work["prolific_id"]}
    )

    def effect(model, term: str) -> tuple[float, float, float, float, float]:
        ci = model.conf_int().loc[term]
        return (
            float(model.params[term]),
            float(model.bse[term]),
            float(model.pvalues[term]),
            float(ci.iloc[0]),
            float(ci.iloc[1]),
        )

    b_beta, b_se, b_p, b_low, b_high = effect(base, "pc1_z")
    a_beta, a_se, a_p, a_low, a_high = effect(adjusted, "pc1_z")
    pf_beta, pf_se, pf_p, pf_low, pf_high = effect(adjusted, "p_factor_z")
    assoc_beta, assoc_se, assoc_p, assoc_low, assoc_high = effect(pc1_p, "pc1_z")

    summary = {
        "specification": label,
        "predictors": ", ".join(predictors),
        "n_observations": int(adjusted.nobs),
        "n_participants": int(work["prolific_id"].nunique()),
        "pc1_variance_explained": float(pca.explained_variance_ratio_[0]),
        "base_beta": b_beta,
        "base_se": b_se,
        "base_p": b_p,
        "base_ci_low": b_low,
        "base_ci_high": b_high,
        "adjusted_beta": a_beta,
        "adjusted_se": a_se,
        "adjusted_p": a_p,
        "adjusted_ci_low": a_low,
        "adjusted_ci_high": a_high,
        "p_factor_beta": pf_beta,
        "p_factor_se": pf_se,
        "p_factor_p": pf_p,
        "p_factor_ci_low": pf_low,
        "p_factor_ci_high": pf_high,
        "pc1_pfactor_beta": assoc_beta,
        "pc1_pfactor_se": assoc_se,
        "pc1_pfactor_p": assoc_p,
        "pc1_pfactor_ci_low": assoc_low,
        "pc1_pfactor_ci_high": assoc_high,
        "base_r_squared": float(base.rsquared),
        "adjusted_r_squared": float(adjusted.rsquared),
        "delta_r_squared": float(adjusted.rsquared - base.rsquared),
    }
    loadings = pd.DataFrame(
        {
            "specification": label,
            "predictor": predictors,
            "pc1_loading": pca.components_[0],
            "pc1_variance_explained": pca.explained_variance_ratio_[0],
        }
    )
    return summary, loadings


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    summaries = []
    loadings = []
    for label, predictors in [
        ("current_five_predictor_pc1", CURRENT_PREDICTORS),
        ("legacy_six_predictor_pc1", LEGACY_PREDICTORS),
    ]:
        summary, loading = fit_specification(data, label, predictors)
        summaries.append(summary)
        loadings.append(loading)

    result = pd.DataFrame(summaries)
    result["adjusted_fdr_q_across_two_pc1_specs"] = multipletests(result["adjusted_p"], method="fdr_bh")[1]
    result.to_csv(OUT_DIR / "suspiciousness_pc1_pfactor_adjusted_models.csv", index=False)
    pd.concat(loadings, ignore_index=True).to_csv(
        OUT_DIR / "suspiciousness_pc1_pfactor_adjusted_loadings.csv", index=False
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
