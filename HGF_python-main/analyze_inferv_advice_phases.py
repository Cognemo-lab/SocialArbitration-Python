from __future__ import annotations

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))

from HGF.code_inversion.tapas_sgm import tapas_sgm
from HGF.code_model_wagad.prc_model.hgf_binary3l_freekappa_reward_social import (
    hgf_binary3l_freekappa_reward_social,
)


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
PARAM_CSV = BASE / "hgf_baselinefixed_full" / "parameter_estimates_long_t1_t2.csv"
TRIAL_CSV = BASE / "hgf_baselinefixed_full" / "extracted_model_trials.csv"
CANONICAL_CSV = BASE / "canonical_100_trial_schedule" / "canonical_100_trial_sequence.csv"
HITOP_DIR = BASE / "hitop" / "processed_data"
OUT_DIR = BASE / "hitop" / "inferv_advice_phase_analysis"

OUTCOMES = [
    "hitop_mistrust_suspiciousness",
    "hitop_reality_distortion",
    "hitop_reality_distortion_delusions",
    "hitop_reality_distortion_hallucinations",
]

OUTCOME_LABELS = {
    "hitop_mistrust_suspiciousness": "Suspiciousness",
    "hitop_reality_distortion": "Reality distortion",
    "hitop_reality_distortion_delusions": "Delusions",
    "hitop_reality_distortion_hallucinations": "Hallucinations",
}

PREDICTOR_LABELS = {
    "inferv_a_mean": "Overall",
    "inferv_a_stable_mean": "Stable phases",
    "inferv_a_volatile_mean": "Volatile phase",
}

PREDICTORS = list(PREDICTOR_LABELS)
TRIPLE_COLS = ["input_advice", "input_reward", "advice_card_space"]

PRC_ORDER = [
    "mu2r_0", "sa2r_0", "mu3r_0", "sa3r_0", "ka_r", "om_r", "th_r",
    "mu2a_0", "sa2a_0", "mu3a_0", "sa3a_0", "ka_a", "om_a", "th_a",
    "phi_r", "m_r", "phi_a", "m_a",
]


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / sd


def load_parameter_pivot() -> pd.DataFrame:
    params = pd.read_csv(PARAM_CSV)
    pivot = (
        params[params["group"] == "prc"]
        .pivot_table(
            index=["prolific_id", "round_name", "timepoint"],
            columns="parameter",
            values="estimate",
            aggfunc="first",
        )
        .reset_index()
    )
    pivot.columns.name = None
    return pivot


def token_sequence(df: pd.DataFrame) -> list[tuple[int, int, int]]:
    return [tuple(int(round(v)) for v in row) for row in df[TRIPLE_COLS].to_numpy(float)]


def align_to_canonical(observed: pd.DataFrame, canonical: pd.DataFrame) -> dict[int, int]:
    matcher = SequenceMatcher(
        None,
        token_sequence(canonical),
        token_sequence(observed),
        autojunk=False,
    )
    mapping: dict[int, int] = {}
    for canonical_start, observed_start, length in matcher.get_matching_blocks():
        for offset in range(length):
            mapping[canonical_start + offset] = observed_start + offset
    return mapping


def derive_phase_states() -> pd.DataFrame:
    params = load_parameter_pivot()
    trials = pd.read_csv(TRIAL_CSV)
    canonical = pd.read_csv(CANONICAL_CSV).sort_values("canonical_trial").reset_index(drop=True)
    rows = []

    for row in params.itertuples(index=False):
        pid = str(row.prolific_id)
        round_name = str(row.round_name)
        g = trials[
            (trials["prolific_id"].astype(str) == pid)
            & (trials["round_name"].astype(str) == round_name)
        ].copy()
        if g.empty:
            continue
        g = g.sort_values("trial_index_choice").reset_index(drop=True)
        u = g[TRIPLE_COLS].to_numpy(float)
        p_prc = np.array([float(getattr(row, name)) for name in PRC_ORDER], dtype=float)
        traj, _ = hgf_binary3l_freekappa_reward_social({"u": u, "ign": []}, p_prc)

        mu2hat_a = traj["muhat_a"][:, 1]
        sa2hat_a = traj["sahat_a"][:, 1]
        prob_a = tapas_sgm(mu2hat_a, 1.0)
        inferv_a = prob_a * (1.0 - prob_a) * sa2hat_a

        mapping = align_to_canonical(g, canonical)
        stable_obs = [obs for can, obs in mapping.items() if can < 30 or can >= 70]
        volatile_obs = [obs for can, obs in mapping.items() if 30 <= can < 70]
        all_obs = [obs for can, obs in mapping.items() if can < 100]

        # Require at least half of each phase so truncated or poorly aligned sessions
        # cannot contribute a phase estimate based on only a few trials.
        stable_mean = float(np.nanmean(inferv_a[stable_obs])) if len(stable_obs) >= 30 else np.nan
        volatile_mean = float(np.nanmean(inferv_a[volatile_obs])) if len(volatile_obs) >= 20 else np.nan
        overall_mean = float(np.nanmean(inferv_a[all_obs])) if len(all_obs) >= 50 else np.nan

        rows.append(
            {
                "prolific_id": pid,
                "round_name": round_name,
                "timepoint": str(row.timepoint).lower(),
                "n_observed_trials": int(len(g)),
                "n_canonical_matched": int(len(all_obs)),
                "n_stable_matched": int(len(stable_obs)),
                "n_volatile_matched": int(len(volatile_obs)),
                "inferv_a_mean": overall_mean,
                "inferv_a_stable_mean": stable_mean,
                "inferv_a_volatile_mean": volatile_mean,
            }
        )
    return pd.DataFrame(rows)


def load_hitop() -> pd.DataFrame:
    frames = []
    for timepoint, filename in [("t1", "hitop_scales_T1.csv"), ("t2", "hitop_scales_T2.csv")]:
        frame = pd.read_csv(HITOP_DIR / filename)[["prolific_id"] + OUTCOMES].copy()
        frame["prolific_id"] = frame["prolific_id"].astype(str)
        frame["timepoint"] = timepoint
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def icc3_1(x: np.ndarray, y: np.ndarray) -> float:
    values = np.column_stack([x, y]).astype(float)
    n, k = values.shape
    subject_means = values.mean(axis=1)
    session_means = values.mean(axis=0)
    grand = values.mean()
    ms_subject = k * np.sum((subject_means - grand) ** 2) / (n - 1)
    residual = values - subject_means[:, None] - session_means[None, :] + grand
    ms_error = np.sum(residual**2) / ((n - 1) * (k - 1))
    return float((ms_subject - ms_error) / (ms_subject + (k - 1) * ms_error))


def reliability_table(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for predictor in PREDICTORS:
        wide = states.pivot_table(index="prolific_id", columns="timepoint", values=predictor, aggfunc="first")
        wide = wide.dropna(subset=["t1", "t2"])
        r, p = pearsonr(wide["t1"], wide["t2"])
        rows.append(
            {
                "predictor": predictor,
                "n": len(wide),
                "pearson_r": r,
                "pearson_p": p,
                "icc3_1": icc3_1(wide["t1"].to_numpy(), wide["t2"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def phase_mean_comparison(states: pd.DataFrame) -> pd.DataFrame:
    work = states.dropna(subset=["inferv_a_stable_mean", "inferv_a_volatile_mean"]).copy()
    work["volatile_minus_stable"] = work["inferv_a_volatile_mean"] - work["inferv_a_stable_mean"]
    model = smf.ols("volatile_minus_stable ~ C(timepoint)", data=work).fit(
        cov_type="cluster", cov_kwds={"groups": work["prolific_id"]}
    )
    ci = model.conf_int()
    rows = []
    for term in model.params.index:
        rows.append(
            {
                "term": term,
                "n_sessions": len(work),
                "stable_mean": work["inferv_a_stable_mean"].mean(),
                "volatile_mean": work["inferv_a_volatile_mean"].mean(),
                "estimate": model.params[term],
                "se": model.bse[term],
                "p_value": model.pvalues[term],
                "ci_low": ci.loc[term, 0],
                "ci_high": ci.loc[term, 1],
            }
        )
    return pd.DataFrame(rows)


def model_row(model, outcome: str, predictor: str, scope: str) -> dict:
    term = f"{predictor}_z"
    ci = model.conf_int().loc[term]
    return {
        "outcome": outcome,
        "predictor": predictor,
        "analysis_scope": scope,
        "n": int(model.nobs),
        "beta": float(model.params[term]),
        "se": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "r_squared": float(model.rsquared),
    }


def run_separate_models(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOMES:
        for predictor in PREDICTORS:
            work = merged[["prolific_id", "timepoint", outcome, predictor]].dropna().copy()
            work[f"{outcome}_z"] = zscore(work[outcome])
            work[f"{predictor}_z"] = zscore(work[predictor])
            pooled = smf.ols(
                f"{outcome}_z ~ {predictor}_z + C(timepoint)", data=work
            ).fit(cov_type="cluster", cov_kwds={"groups": work["prolific_id"]})
            rows.append(model_row(pooled, outcome, predictor, "pooled_clustered"))

            t2 = work[work["timepoint"] == "t2"].copy()
            t2[f"{outcome}_z"] = zscore(t2[outcome])
            t2[f"{predictor}_z"] = zscore(t2[predictor])
            model_t2 = smf.ols(f"{outcome}_z ~ {predictor}_z", data=t2).fit()
            rows.append(model_row(model_t2, outcome, predictor, "t2"))
    result = pd.DataFrame(rows)
    result["fdr_q_within_scope"] = np.nan
    for scope, index in result.groupby("analysis_scope").groups.items():
        result.loc[index, "fdr_q_within_scope"] = multipletests(
            result.loc[index, "p_value"], method="fdr_bh"
        )[1]
    return result


def run_phase_contrast_models(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work_all = merged.copy()
    work_all["inferv_a_volatile_minus_stable"] = (
        work_all["inferv_a_volatile_mean"] - work_all["inferv_a_stable_mean"]
    )
    predictor = "inferv_a_volatile_minus_stable"
    for outcome in OUTCOMES:
        work = work_all[["prolific_id", "timepoint", outcome, predictor]].dropna().copy()
        work[f"{outcome}_z"] = zscore(work[outcome])
        work[f"{predictor}_z"] = zscore(work[predictor])
        pooled = smf.ols(
            f"{outcome}_z ~ {predictor}_z + C(timepoint)", data=work
        ).fit(cov_type="cluster", cov_kwds={"groups": work["prolific_id"]})
        rows.append(model_row(pooled, outcome, predictor, "pooled_clustered"))

        t2 = work[work["timepoint"] == "t2"].copy()
        t2[f"{outcome}_z"] = zscore(t2[outcome])
        t2[f"{predictor}_z"] = zscore(t2[predictor])
        model_t2 = smf.ols(f"{outcome}_z ~ {predictor}_z", data=t2).fit()
        rows.append(model_row(model_t2, outcome, predictor, "t2"))
    return pd.DataFrame(rows)


def run_joint_models(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    coefficients = []
    predictors = ["inferv_a_stable_mean", "inferv_a_volatile_mean"]
    for outcome in OUTCOMES:
        for scope in ["pooled_clustered", "t2"]:
            cols = ["prolific_id", "timepoint", outcome] + predictors
            work = merged[cols].dropna().copy()
            if scope == "t2":
                work = work[work["timepoint"] == "t2"].copy()
            work[f"{outcome}_z"] = zscore(work[outcome])
            for predictor in predictors:
                work[f"{predictor}_z"] = zscore(work[predictor])
            formula = (
                f"{outcome}_z ~ inferv_a_stable_mean_z + inferv_a_volatile_mean_z"
                + (" + C(timepoint)" if scope == "pooled_clustered" else "")
            )
            model = smf.ols(formula, data=work).fit(
                cov_type="cluster" if scope == "pooled_clustered" else "nonrobust",
                cov_kwds={"groups": work["prolific_id"]} if scope == "pooled_clustered" else None,
            )
            test = model.wald_test(
                "inferv_a_stable_mean_z = 0, inferv_a_volatile_mean_z = 0"
            )
            slope_difference = model.wald_test(
                "inferv_a_stable_mean_z = inferv_a_volatile_mean_z"
            )
            summaries.append(
                {
                    "outcome": outcome,
                    "analysis_scope": scope,
                    "n": int(model.nobs),
                    "stable_volatile_r": float(work[predictors].corr().iloc[0, 1]),
                    "joint_statistic": float(test.statistic),
                    "joint_p_value": float(test.pvalue),
                    "stable_vs_volatile_slope_statistic": float(slope_difference.statistic),
                    "stable_vs_volatile_slope_p_value": float(slope_difference.pvalue),
                    "r_squared": float(model.rsquared),
                }
            )
            for predictor in predictors:
                coefficients.append(model_row(model, outcome, predictor, scope))
    return pd.DataFrame(summaries), pd.DataFrame(coefficients)


def make_figure(states: pd.DataFrame, reliability: pd.DataFrame, models: pd.DataFrame) -> None:
    suspicious = models[models["outcome"] == "hitop_mistrust_suspiciousness"].copy()
    colors = {"inferv_a_mean": "#687786", "inferv_a_stable_mean": "#34745b", "inferv_a_volatile_mean": "#c6812f"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))

    ax = axes[0]
    y = np.arange(len(PREDICTORS))
    ax.scatter(reliability.set_index("predictor").loc[PREDICTORS, "pearson_r"], y - 0.10, label="Pearson r", color="#356aa0", s=55)
    ax.scatter(reliability.set_index("predictor").loc[PREDICTORS, "icc3_1"], y + 0.10, label="ICC(3,1)", color="#d47b28", s=55)
    ax.set_yticks(y)
    ax.set_yticklabels([PREDICTOR_LABELS[p] for p in PREDICTORS])
    ax.invert_yaxis()
    ax.axvline(0.4, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Test-retest reliability")
    ax.set_title("A. Reliability", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)

    for ax, scope, title in zip(axes[1:], ["pooled_clustered", "t2"], ["B. Suspiciousness: pooled", "C. Suspiciousness: T2"]):
        sub = suspicious[suspicious["analysis_scope"] == scope].set_index("predictor").loc[PREDICTORS].reset_index()
        y = np.arange(len(sub))
        ax.errorbar(sub["beta"], y, xerr=1.96 * sub["se"], fmt="none", color="#222222", capsize=3, linewidth=1.2)
        for i, row in sub.iterrows():
            ax.scatter(row["beta"], i, color=colors[row["predictor"]], s=70, zorder=3)
            ax.text(row["beta"] + (0.012 if row["beta"] >= 0 else -0.012), i, f"{row['beta']:.2f}\np={row['p_value']:.3g}", ha="left" if row["beta"] >= 0 else "right", va="center", fontsize=8.5)
        ax.set_yticks(y)
        ax.set_yticklabels([PREDICTOR_LABELS[p] for p in sub["predictor"]])
        ax.invert_yaxis()
        ax.axvline(0, color="#777777", linewidth=1)
        ax.set_xlim(-0.35, 0.12)
        ax.set_xlabel("Standardized beta")
        ax.set_title(title, loc="left", fontweight="bold")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Advice inferential variance by programmed phase", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "figure_inferv_advice_phase_analysis.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure_inferv_advice_phase_analysis.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    states = derive_phase_states()
    hitop = load_hitop()
    merged = states.merge(hitop, on=["prolific_id", "timepoint"], how="inner")
    reliability = reliability_table(states)
    phase_comparison = phase_mean_comparison(states)
    separate = run_separate_models(merged)
    phase_contrast = run_phase_contrast_models(merged)
    joint_summary, joint_coef = run_joint_models(merged)

    states.to_csv(OUT_DIR / "inferv_advice_phase_states.csv", index=False)
    merged.to_csv(OUT_DIR / "inferv_advice_phase_hitop_merged.csv", index=False)
    reliability.to_csv(OUT_DIR / "inferv_advice_phase_reliability.csv", index=False)
    phase_comparison.to_csv(OUT_DIR / "inferv_advice_phase_mean_comparison.csv", index=False)
    separate.to_csv(OUT_DIR / "inferv_advice_phase_separate_models.csv", index=False)
    phase_contrast.to_csv(OUT_DIR / "inferv_advice_phase_contrast_models.csv", index=False)
    joint_summary.to_csv(OUT_DIR / "inferv_advice_phase_joint_model_summary.csv", index=False)
    joint_coef.to_csv(OUT_DIR / "inferv_advice_phase_joint_coefficients.csv", index=False)
    make_figure(states, reliability, separate)

    payload = {
        "phase_definition": {"stable": "canonical trials 1-30 and 71-100", "volatile": "canonical trials 31-70"},
        "n_sessions": int(len(states)),
        "median_canonical_trials_matched": float(states["n_canonical_matched"].median()),
        "minimum_phase_trials_required": {"stable": 30, "volatile": 20},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(OUT_DIR / "figure_inferv_advice_phase_analysis.png")


if __name__ == "__main__":
    main()
