from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
STATE_WIDE = BASE / "hgf_baselinefixed_full" / "state_reliability" / "state_estimates_wide_t1_t2.csv"
HITOP_DIR = BASE / "hitop" / "processed_data"
OUT_DIR = BASE / "hitop" / "reliable_state_fod_moderation"

STATE_METRICS = ["inferv_a_mean", "abs_eps2_a_mean"]


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


def load_state_long(state_metric: str) -> pd.DataFrame:
    state = pd.read_csv(STATE_WIDE)
    df = state.loc[state["state_metric"] == state_metric].copy()
    return df.melt(
        id_vars=["prolific_id", "state_metric"],
        value_vars=["t1", "t2"],
        var_name="timepoint",
        value_name=state_metric,
    ).drop(columns=["state_metric"])


def zscore_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        sd = out[col].std(ddof=0)
        out[f"{col}_z"] = 0.0 if sd == 0 else (out[col] - out[col].mean()) / sd
    return out


def summarize_model(model, state_metric: str, analysis_scope: str, n: int) -> dict:
    term = f"{state_metric}_z:hitop_suicidality_z"
    return {
        "state_metric": state_metric,
        "analysis_scope": analysis_scope,
        "n": int(n),
        "interaction_beta": float(model.params[term]),
        "interaction_se": float(model.bse[term]),
        "interaction_p": float(model.pvalues[term]),
        "main_state_beta": float(model.params[f"{state_metric}_z"]),
        "main_state_p": float(model.pvalues[f"{state_metric}_z"]),
        "main_suicidality_beta": float(model.params["hitop_suicidality_z"]),
        "main_suicidality_p": float(model.pvalues["hitop_suicidality_z"]),
        "r_squared": float(getattr(model, "rsquared", float("nan"))),
    }


def simple_slopes(model, state_metric: str, analysis_scope: str) -> list[dict]:
    term = f"{state_metric}_z"
    inter = f"{state_metric}_z:hitop_suicidality_z"
    rows = []
    for z, label in [(-1.0, "low"), (0.0, "mean"), (1.0, "high")]:
        rows.append(
            {
                "state_metric": state_metric,
                "analysis_scope": analysis_scope,
                "moderator_level": label,
                "suicidality_z": z,
                "simple_slope_state": float(model.params[term] + model.params[inter] * z),
            }
        )
    return rows


def fit_pooled_model(df: pd.DataFrame, state_metric: str):
    return smf.ols(
        f"gcsq_fearlesness_of_death_z ~ {state_metric}_z * hitop_suicidality_z + C(timepoint)",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["prolific_id"]})


def fit_timepoint_model(df: pd.DataFrame, state_metric: str):
    return smf.ols(
        f"gcsq_fearlesness_of_death_z ~ {state_metric}_z * hitop_suicidality_z",
        data=df,
    ).fit()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clinical = load_clinical_long()

    summary_rows = []
    slope_rows = []

    for state_metric in STATE_METRICS:
        state_long = load_state_long(state_metric)
        merged = state_long.merge(clinical, on=["prolific_id", "timepoint"], how="inner").dropna().copy()
        merged.to_csv(OUT_DIR / f"merged_{state_metric}_long.csv", index=False)

        pooled = zscore_frame(merged, [state_metric, "hitop_suicidality", "gcsq_fearlesness_of_death"])
        pooled_model = fit_pooled_model(pooled, state_metric)
        summary_rows.append(summarize_model(pooled_model, state_metric, "pooled_clustered", len(pooled)))
        slope_rows.extend(simple_slopes(pooled_model, state_metric, "pooled_clustered"))

        subj = (
            merged.groupby("prolific_id")[[state_metric, "hitop_suicidality", "gcsq_fearlesness_of_death"]]
            .mean(numeric_only=True)
            .dropna()
            .reset_index()
        )
        subj = zscore_frame(subj, [state_metric, "hitop_suicidality", "gcsq_fearlesness_of_death"])
        subj_model = fit_timepoint_model(subj, state_metric)
        summary_rows.append(summarize_model(subj_model, state_metric, "subject_mean", len(subj)))
        slope_rows.extend(simple_slopes(subj_model, state_metric, "subject_mean"))
        subj.to_csv(OUT_DIR / f"merged_{state_metric}_subject_mean.csv", index=False)

        for timepoint in ["t1", "t2"]:
            tp = merged.loc[merged["timepoint"] == timepoint].copy()
            tp = zscore_frame(tp, [state_metric, "hitop_suicidality", "gcsq_fearlesness_of_death"])
            tp_model = fit_timepoint_model(tp, state_metric)
            summary_rows.append(summarize_model(tp_model, state_metric, timepoint, len(tp)))
            slope_rows.extend(simple_slopes(tp_model, state_metric, timepoint))

    summary_df = pd.DataFrame(summary_rows).sort_values(["state_metric", "analysis_scope"])
    slope_df = pd.DataFrame(slope_rows).sort_values(["state_metric", "analysis_scope", "suicidality_z"])
    summary_df.to_csv(OUT_DIR / "moderation_summary_all_scopes.csv", index=False)
    slope_df.to_csv(OUT_DIR / "moderation_simple_slopes_all_scopes.csv", index=False)

    grouped = {}
    for state_metric in STATE_METRICS:
        grouped[state_metric] = {
            "summary_rows": summary_df.loc[summary_df["state_metric"] == state_metric].to_dict(orient="records"),
            "simple_slopes": slope_df.loc[slope_df["state_metric"] == state_metric].to_dict(orient="records"),
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(grouped, indent=2))


if __name__ == "__main__":
    main()
