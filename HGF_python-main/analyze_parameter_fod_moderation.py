from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
PARAM_LONG = BASE / "hgf_baselinefixed_full" / "parameter_estimates_long_t1_t2.csv"
PARAM_REL = BASE / "hgf_baselinefixed_full" / "test_retest_reliability_by_parameter.csv"
HITOP_DIR = BASE / "hitop" / "processed_data"
OUT_DIR = BASE / "hitop" / "parameter_fod_moderation"

PARAMETERS = ["om_a", "m_a", "mu3a_0"]


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


def zscore_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        sd = out[col].std(ddof=0)
        out[f"{col}_z"] = 0.0 if sd == 0 else (out[col] - out[col].mean()) / sd
    return out


def summarize_model(model, parameter: str, analysis_scope: str, n: int) -> dict:
    term = f"{parameter}_z:hitop_suicidality_z"
    return {
        "parameter": parameter,
        "analysis_scope": analysis_scope,
        "n": int(n),
        "interaction_beta": float(model.params[term]),
        "interaction_se": float(model.bse[term]),
        "interaction_p": float(model.pvalues[term]),
        "main_parameter_beta": float(model.params[f"{parameter}_z"]),
        "main_parameter_p": float(model.pvalues[f"{parameter}_z"]),
        "main_suicidality_beta": float(model.params["hitop_suicidality_z"]),
        "main_suicidality_p": float(model.pvalues["hitop_suicidality_z"]),
        "r_squared": float(getattr(model, "rsquared", float("nan"))),
    }


def simple_slopes(model, parameter: str, analysis_scope: str) -> list[dict]:
    term = f"{parameter}_z"
    inter = f"{parameter}_z:hitop_suicidality_z"
    rows = []
    for z, label in [(-1.0, "low"), (0.0, "mean"), (1.0, "high")]:
        rows.append(
            {
                "parameter": parameter,
                "analysis_scope": analysis_scope,
                "moderator_level": label,
                "suicidality_z": z,
                "simple_slope_parameter": float(model.params[term] + model.params[inter] * z),
            }
        )
    return rows


def fit_pooled_model(df: pd.DataFrame, parameter: str):
    return smf.ols(
        f"gcsq_fearlesness_of_death_z ~ {parameter}_z * hitop_suicidality_z + C(timepoint)",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["prolific_id"]})


def fit_timepoint_model(df: pd.DataFrame, parameter: str):
    return smf.ols(
        f"gcsq_fearlesness_of_death_z ~ {parameter}_z * hitop_suicidality_z",
        data=df,
    ).fit()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    params = pd.read_csv(PARAM_LONG)
    rel = pd.read_csv(PARAM_REL)
    clinical = load_clinical_long()

    availability_rows = []
    summary_rows = []
    slope_rows = []

    for parameter in PARAMETERS:
        rel_row = rel.loc[rel["parameter"] == parameter]
        if rel_row.empty:
            availability_rows.append(
                {"parameter": parameter, "status": "missing", "reason": "parameter_not_found"}
            )
            continue

        is_fixed = bool(rel_row["is_fixed"].iloc[0])
        availability_rows.append(
            {
                "parameter": parameter,
                "status": "ok" if not is_fixed else "not_estimable",
                "reason": "fixed_in_model" if is_fixed else "",
                "pearson_r_t1_t2": float(rel_row["pearson_r_t1_t2"].iloc[0]) if pd.notna(rel_row["pearson_r_t1_t2"].iloc[0]) else None,
                "icc3_1_t1_t2": float(rel_row["icc3_1_t1_t2"].iloc[0]) if pd.notna(rel_row["icc3_1_t1_t2"].iloc[0]) else None,
            }
        )
        if is_fixed:
            continue

        p_long = params.loc[params["parameter"] == parameter, ["prolific_id", "round_name", "timepoint", "estimate"]].copy()
        p_long = p_long.rename(columns={"estimate": parameter})
        merged = p_long.merge(clinical, on=["prolific_id", "timepoint"], how="inner").dropna().copy()
        merged.to_csv(OUT_DIR / f"merged_{parameter}_long.csv", index=False)

        pooled = zscore_frame(merged, [parameter, "hitop_suicidality", "gcsq_fearlesness_of_death"])
        pooled_model = fit_pooled_model(pooled, parameter)
        summary_rows.append(summarize_model(pooled_model, parameter, "pooled_clustered", len(pooled)))
        slope_rows.extend(simple_slopes(pooled_model, parameter, "pooled_clustered"))

        subj = (
            merged.groupby("prolific_id")[[parameter, "hitop_suicidality", "gcsq_fearlesness_of_death"]]
            .mean(numeric_only=True)
            .dropna()
            .reset_index()
        )
        subj = zscore_frame(subj, [parameter, "hitop_suicidality", "gcsq_fearlesness_of_death"])
        subj_model = fit_timepoint_model(subj, parameter)
        summary_rows.append(summarize_model(subj_model, parameter, "subject_mean", len(subj)))
        slope_rows.extend(simple_slopes(subj_model, parameter, "subject_mean"))
        subj.to_csv(OUT_DIR / f"merged_{parameter}_subject_mean.csv", index=False)

        for timepoint in ["t1", "t2"]:
            tp = merged.loc[merged["timepoint"] == timepoint].copy()
            tp = zscore_frame(tp, [parameter, "hitop_suicidality", "gcsq_fearlesness_of_death"])
            tp_model = fit_timepoint_model(tp, parameter)
            summary_rows.append(summarize_model(tp_model, parameter, timepoint, len(tp)))
            slope_rows.extend(simple_slopes(tp_model, parameter, timepoint))

    pd.DataFrame(availability_rows).to_csv(OUT_DIR / "parameter_availability.csv", index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).sort_values(["parameter", "analysis_scope"]).to_csv(
            OUT_DIR / "moderation_summary_all_scopes.csv", index=False
        )
        pd.DataFrame(slope_rows).sort_values(["parameter", "analysis_scope", "suicidality_z"]).to_csv(
            OUT_DIR / "moderation_simple_slopes_all_scopes.csv", index=False
        )

    payload = {
        "availability": availability_rows,
        "n_analyzed_parameters": int(sum(1 for row in availability_rows if row["status"] == "ok")),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
