from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
STATE_WIDE = BASE / "hgf_baselinefixed_full" / "state_reliability" / "state_estimates_wide_t1_t2.csv"
HITOP_DIR = BASE / "hitop" / "processed_data"
OUT_DIR = BASE / "hitop" / "state_suicidality_fod_by_timepoint"

STATE_METRICS = ["inferv_a_mean", "abs_eps2_a_mean"]


def load_clinical() -> pd.DataFrame:
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


def summarize_model(model, state_metric: str, timepoint: str, n: int) -> dict:
    term = f"{state_metric}_z:hitop_suicidality_z"
    return {
        "state_metric": state_metric,
        "timepoint": timepoint,
        "n": int(n),
        "interaction_beta": float(model.params[term]),
        "interaction_se": float(model.bse[term]),
        "interaction_p": float(model.pvalues[term]),
        "main_state_beta": float(model.params[f"{state_metric}_z"]),
        "main_state_p": float(model.pvalues[f"{state_metric}_z"]),
        "main_suicidality_beta": float(model.params["hitop_suicidality_z"]),
        "main_suicidality_p": float(model.pvalues["hitop_suicidality_z"]),
        "r_squared": float(model.rsquared),
    }


def simple_slopes(model, state_metric: str) -> list[dict]:
    term = f"{state_metric}_z"
    inter = f"{state_metric}_z:hitop_suicidality_z"
    rows = []
    for z, label in [(-1.0, "low"), (0.0, "mean"), (1.0, "high")]:
        rows.append(
            {
                "moderator_level": label,
                "suicidality_z": z,
                "simple_slope_state": float(model.params[term] + model.params[inter] * z),
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clinical = load_clinical()
    summaries = []
    slope_rows = []

    for state_metric in STATE_METRICS:
        state_long = load_state_long(state_metric)
        merged = state_long.merge(clinical, on=["prolific_id", "timepoint"], how="inner").dropna().copy()
        merged.to_csv(OUT_DIR / f"merged_{state_metric}_long.csv", index=False)

        for timepoint in ["t1", "t2"]:
            df = merged.loc[merged["timepoint"] == timepoint].copy()
            df = zscore_frame(df, [state_metric, "hitop_suicidality", "gcsq_fearlesness_of_death"])
            model = smf.ols(
                f"gcsq_fearlesness_of_death_z ~ {state_metric}_z * hitop_suicidality_z",
                data=df,
            ).fit()
            summaries.append(summarize_model(model, state_metric, timepoint, len(df)))
            for row in simple_slopes(model, state_metric):
                slope_rows.append({"state_metric": state_metric, "timepoint": timepoint, **row})

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "timepoint_specific_moderation_summary.csv", index=False)
    pd.DataFrame(slope_rows).to_csv(OUT_DIR / "timepoint_specific_simple_slopes.csv", index=False)

    grouped = {}
    for state_metric in STATE_METRICS:
        sub = summary_df.loc[summary_df["state_metric"] == state_metric].copy()
        grouped[state_metric] = sub.to_dict(orient="records")
    (OUT_DIR / "summary.json").write_text(json.dumps(grouped, indent=2))


if __name__ == "__main__":
    main()
