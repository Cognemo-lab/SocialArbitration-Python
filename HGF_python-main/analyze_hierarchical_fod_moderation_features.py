from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
PARAM_LONG = BASE / "hgf_baselinefixed_full" / "parameter_estimates_long_t1_t2.csv"
STATE_WIDE = BASE / "hgf_baselinefixed_full" / "state_reliability" / "state_estimates_wide_t1_t2.csv"
HITOP_DIR = BASE / "hitop" / "processed_data"
OUT_DIR = BASE / "hitop" / "hierarchical_fod_moderation_features"

PREDICTORS = ["abs_eps2_a_mean", "inferv_a_mean", "ka_a", "om_a", "eps3_a_mean"]


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
    ka_a = params.loc[params["parameter"] == "ka_a", ["prolific_id", "round_name", "timepoint", "estimate"]].copy()
    ka_a = ka_a.rename(columns={"estimate": "ka_a"})
    om_a = params.loc[params["parameter"] == "om_a", ["prolific_id", "round_name", "timepoint", "estimate"]].copy()
    om_a = om_a.rename(columns={"estimate": "om_a"})

    states = pd.read_csv(STATE_WIDE)
    keep = states.loc[
        states["state_metric"].isin(["abs_eps2_a_mean", "inferv_a_mean", "eps3_a_mean"])
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

    merged = ka_a.merge(om_a, on=["prolific_id", "round_name", "timepoint"], how="inner")
    merged = merged.merge(keep, on=["prolific_id", "timepoint"], how="inner")
    return merged


def zscore_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        sd = out[col].std(ddof=0)
        out[f"{col}_z"] = 0.0 if sd == 0 else (out[col] - out[col].mean()) / sd
    return out


def coef_table(result) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "term": result.params.index,
            "coef": result.params.values,
            "se": result.bse.values,
            "z_value": result.tvalues.values,
            "p_value": result.pvalues.values,
        }
    )
    try:
        ci = result.conf_int()
        out["ci_low"] = ci[0].values
        out["ci_high"] = ci[1].values
    except Exception:
        out["ci_low"] = pd.NA
        out["ci_high"] = pd.NA
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clinical = load_clinical_long()
    predictors = load_predictors()
    merged = predictors.merge(clinical, on=["prolific_id", "timepoint"], how="inner").dropna().copy()
    merged = zscore_frame(merged, PREDICTORS + ["hitop_suicidality", "gcsq_fearlesness_of_death"])
    merged.to_csv(OUT_DIR / "hierarchical_merged_long.csv", index=False)

    formula = (
        "gcsq_fearlesness_of_death_z ~ C(timepoint) + hitop_suicidality_z + "
        "abs_eps2_a_mean_z + inferv_a_mean_z + ka_a_z + om_a_z + eps3_a_mean_z + "
        "abs_eps2_a_mean_z:hitop_suicidality_z + "
        "inferv_a_mean_z:hitop_suicidality_z + "
        "ka_a_z:hitop_suicidality_z + "
        "om_a_z:hitop_suicidality_z + "
        "eps3_a_mean_z:hitop_suicidality_z"
    )

    full_model = smf.mixedlm(formula, data=merged, groups=merged["prolific_id"])
    full_result = full_model.fit(method="lbfgs", reml=False, maxiter=500, disp=False)

    reduced_formula = (
        "gcsq_fearlesness_of_death_z ~ C(timepoint) + hitop_suicidality_z + "
        "abs_eps2_a_mean_z + inferv_a_mean_z + ka_a_z + om_a_z + eps3_a_mean_z"
    )
    reduced_model = smf.mixedlm(reduced_formula, data=merged, groups=merged["prolific_id"])
    reduced_result = reduced_model.fit(method="lbfgs", reml=False, maxiter=500, disp=False)

    coef = coef_table(full_result)
    coef.to_csv(OUT_DIR / "hierarchical_moderation_coefficients.csv", index=False)

    interaction_rows = coef.loc[coef["term"].str.contains(":hitop_suicidality_z", regex=False)].copy()
    interaction_rows.to_csv(OUT_DIR / "hierarchical_moderation_interactions.csv", index=False)

    lr_stat = 2 * (full_result.llf - reduced_result.llf)
    df_diff = len(full_result.params) - len(reduced_result.params)

    summary = {
        "n_rows": int(len(merged)),
        "n_subjects": int(merged["prolific_id"].nunique()),
        "predictors": PREDICTORS,
        "full_model_loglik": float(full_result.llf),
        "reduced_model_loglik": float(reduced_result.llf),
        "interaction_block_lr_stat": float(lr_stat),
        "interaction_block_df_diff": int(df_diff),
        "aic_full": float(full_result.aic),
        "aic_reduced": float(reduced_result.aic),
        "bic_full": float(full_result.bic),
        "bic_reduced": float(reduced_result.bic),
        "converged_full": bool(full_result.converged),
        "converged_reduced": bool(reduced_result.converged),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    report_lines = [
        "Hierarchical moderation model for fearlessness of death",
        f"Rows: {summary['n_rows']}",
        f"Subjects: {summary['n_subjects']}",
        f"Predictors: {', '.join(PREDICTORS)}",
        f"Full model converged: {summary['converged_full']}",
        f"Reduced model converged: {summary['converged_reduced']}",
        f"LogLik full: {summary['full_model_loglik']:.3f}",
        f"LogLik reduced: {summary['reduced_model_loglik']:.3f}",
        f"Interaction block LR stat: {summary['interaction_block_lr_stat']:.3f} (df={summary['interaction_block_df_diff']})",
        "Interaction coefficients:",
    ]
    for _, row in interaction_rows.iterrows():
        report_lines.append(
            f"  {row['term']}: beta={row['coef']:.3f}, SE={row['se']:.3f}, p={row['p_value']:.6f}"
        )
    (OUT_DIR / "shared_report.txt").write_text("\n".join(report_lines))


if __name__ == "__main__":
    main()
