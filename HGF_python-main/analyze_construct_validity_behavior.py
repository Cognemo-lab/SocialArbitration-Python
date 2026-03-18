from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
TRIALS_CSV = BASE / "hgf_baselinefixed_full" / "extracted_model_trials.csv"
PARAM_CSV = BASE / "hgf_baselinefixed_full" / "parameter_estimates_long_t1_t2.csv"
OUT_DIR = BASE / "hgf_baselinefixed_full" / "construct_validity_behavior"

BEHAVIOR_METRICS = [
    "accuracy_rate",
    "advice_taking_rate",
    "win_stay_rate",
    "lose_switch_rate",
]


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


def safe_corr(x: pd.Series, y: pd.Series) -> dict[str, float]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(df)
    if n < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return {
            "n": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }
    pr, pp = pearsonr(df["x"], df["y"])
    sr, sp = spearmanr(df["x"], df["y"])
    return {
        "n": n,
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
    }


def behavior_summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pid, round_name), g in trials.groupby(["prolific_id", "round_name"], sort=False):
        g = g.sort_values("trial_index_choice").reset_index(drop=True).copy()
        correct = (g["choice_side"] == g["input_reward"]).astype(float)
        prev_correct = correct.shift(1)
        prev_choice = g["choice_side"].shift(1)
        stay = (g["choice_side"] == prev_choice).astype(float)
        switch = 1.0 - stay
        win_mask = prev_correct == 1
        lose_mask = prev_correct == 0

        rows.append(
            {
                "prolific_id": pid,
                "round_name": round_name,
                "timepoint": "t1" if round_name == "round1" else "t2",
                "n_trials": int(len(g)),
                "accuracy_rate": float(correct.mean()),
                "advice_taking_rate": float(g["choice_advice_taken"].mean()),
                "win_stay_rate": float(stay.loc[win_mask].mean()) if win_mask.any() else np.nan,
                "lose_switch_rate": float(switch.loc[lose_mask].mean()) if lose_mask.any() else np.nan,
                "mean_wager": float(g["wager"].mean()),
            }
        )
    return pd.DataFrame(rows)


def parameter_wide(params: pd.DataFrame) -> pd.DataFrame:
    free = params.loc[~params["is_fixed"]].copy()
    wide = free.pivot_table(
        index=["prolific_id", "round_name", "timepoint"],
        columns="parameter",
        values="estimate",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def association_table(df: pd.DataFrame, parameter_names: list[str], scope: str) -> pd.DataFrame:
    rows = []
    for behavior in BEHAVIOR_METRICS:
        for parameter in parameter_names:
            stats = safe_corr(df[parameter], df[behavior])
            rows.append(
                {
                    "analysis_scope": scope,
                    "behavior_metric": behavior,
                    "parameter": parameter,
                    **stats,
                }
            )
    out = pd.DataFrame(rows)
    out["fdr_q_pearson_within_behavior_scope"] = (
        out.groupby(["analysis_scope", "behavior_metric"])["pearson_p"].transform(bh_fdr)
    )
    return out.sort_values(["behavior_metric", "parameter"]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    trials = pd.read_csv(TRIALS_CSV)
    params = pd.read_csv(PARAM_CSV)

    beh = behavior_summary(trials)
    beh.to_csv(OUT_DIR / "behavior_session_summary.csv", index=False)

    wide = parameter_wide(params)
    merged = beh.merge(wide, on=["prolific_id", "round_name", "timepoint"], how="inner")
    merged.to_csv(OUT_DIR / "behavior_parameter_merged.csv", index=False)

    parameter_names = sorted(
        params.loc[~params["is_fixed"], "parameter"].drop_duplicates().tolist()
    )

    results = []
    for scope_name in ["t1", "t2"]:
        sub = merged.loc[merged["timepoint"] == scope_name].copy()
        results.append(association_table(sub, parameter_names, scope_name))
    results.append(association_table(merged, parameter_names, "pooled_t1_t2"))

    mean_cols = BEHAVIOR_METRICS + parameter_names
    subj = merged.groupby("prolific_id")[mean_cols].mean(numeric_only=True).reset_index()
    subj.to_csv(OUT_DIR / "behavior_parameter_subject_mean.csv", index=False)
    results.append(association_table(subj, parameter_names, "subject_mean_t1_t2"))

    assoc = pd.concat(results, ignore_index=True)
    assoc.to_csv(OUT_DIR / "construct_validity_parameter_behavior.csv", index=False)

    top = []
    for scope in ["pooled_t1_t2", "subject_mean_t1_t2"]:
        sub = assoc.loc[assoc["analysis_scope"] == scope].copy()
        sub = sub.assign(abs_r=sub["pearson_r"].abs())
        top.append(
            sub.sort_values(["behavior_metric", "abs_r"], ascending=[True, False])
            .groupby("behavior_metric", as_index=False)
            .head(5)
        )
    top_df = pd.concat(top, ignore_index=True)
    top_df.to_csv(OUT_DIR / "top_construct_validity_hits.csv", index=False)

    summary = {
        "n_sessions": int(len(beh)),
        "n_subjects": int(beh["prolific_id"].nunique()),
        "n_free_parameters": int(len(parameter_names)),
        "behavior_metric_means": {
            metric: float(beh[metric].mean()) for metric in BEHAVIOR_METRICS
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
