from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
BEHAVIOR_CSV = BASE / "hgf_baselinefixed_full" / "construct_validity_behavior" / "behavior_session_summary.csv"
STATE_WIDE_CSV = BASE / "hgf_baselinefixed_full" / "state_reliability" / "state_estimates_wide_t1_t2.csv"
STATE_REL_CSV = BASE / "hgf_baselinefixed_full" / "state_reliability" / "state_test_retest_reliability.csv"
OUT_DIR = BASE / "hgf_baselinefixed_full" / "construct_validity_states"

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


def load_reliable_states() -> list[str]:
    rel = pd.read_csv(STATE_REL_CSV)
    keep = rel.loc[
        (rel["pearson_r_t1_t2"] >= 0.4) & (rel["icc3_1_t1_t2"] >= 0.4),
        "state_metric",
    ].tolist()
    return sorted(keep)


def load_state_wide(states: list[str]) -> pd.DataFrame:
    wide = pd.read_csv(STATE_WIDE_CSV)
    wide = wide.loc[wide["state_metric"].isin(states)].copy()
    long = wide.melt(
        id_vars=["prolific_id", "state_metric"],
        value_vars=["t1", "t2"],
        var_name="timepoint",
        value_name="estimate",
    )
    out = long.pivot_table(
        index=["prolific_id", "timepoint"],
        columns="state_metric",
        values="estimate",
        aggfunc="first",
    ).reset_index()
    out.columns.name = None
    return out


def association_table(df: pd.DataFrame, states: list[str], scope: str) -> pd.DataFrame:
    rows = []
    for behavior in BEHAVIOR_METRICS:
        for state in states:
            stats = safe_corr(df[state], df[behavior])
            rows.append(
                {
                    "analysis_scope": scope,
                    "behavior_metric": behavior,
                    "state_metric": state,
                    **stats,
                }
            )
    out = pd.DataFrame(rows)
    out["fdr_q_pearson_within_behavior_scope"] = (
        out.groupby(["analysis_scope", "behavior_metric"])["pearson_p"].transform(bh_fdr)
    )
    return out.sort_values(["behavior_metric", "state_metric"]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    behavior = pd.read_csv(BEHAVIOR_CSV)
    reliable_states = load_reliable_states()
    state_wide = load_state_wide(reliable_states)

    merged = behavior.merge(state_wide, on=["prolific_id", "timepoint"], how="inner")
    merged.to_csv(OUT_DIR / "behavior_state_merged.csv", index=False)

    results = []
    for scope in ["t1", "t2"]:
        sub = merged.loc[merged["timepoint"] == scope].copy()
        results.append(association_table(sub, reliable_states, scope))
    results.append(association_table(merged, reliable_states, "pooled_t1_t2"))

    mean_cols = BEHAVIOR_METRICS + reliable_states
    subj = merged.groupby("prolific_id")[mean_cols].mean(numeric_only=True).reset_index()
    subj.to_csv(OUT_DIR / "behavior_state_subject_mean.csv", index=False)
    results.append(association_table(subj, reliable_states, "subject_mean_t1_t2"))

    assoc = pd.concat(results, ignore_index=True)
    assoc.to_csv(OUT_DIR / "construct_validity_state_behavior.csv", index=False)

    top = []
    for scope in ["pooled_t1_t2", "subject_mean_t1_t2"]:
        sub = assoc.loc[assoc["analysis_scope"] == scope].copy()
        sub = sub.assign(abs_r=sub["pearson_r"].abs())
        top.append(
            sub.sort_values(["behavior_metric", "abs_r"], ascending=[True, False])
            .groupby("behavior_metric", as_index=False)
            .head(5)
        )
    pd.concat(top, ignore_index=True).to_csv(OUT_DIR / "top_construct_validity_state_hits.csv", index=False)

    summary = {
        "n_sessions": int(len(behavior)),
        "n_subjects": int(behavior["prolific_id"].nunique()),
        "n_reliable_states": int(len(reliable_states)),
        "reliable_states": reliable_states,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
