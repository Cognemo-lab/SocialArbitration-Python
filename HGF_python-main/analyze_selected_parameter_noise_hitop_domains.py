from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
PARAM_CSV = BASE / "hgf_baselinefixed_full" / "parameter_estimates_long_t1_t2.csv"
HITOP_DIR = BASE / "hitop" / "processed_data"
OUT_DIR = BASE / "hitop" / "selected_parameter_noise_domain_analysis"

PREDICTOR_BY_OUTCOME = {
    "hitop_mistrust_suspiciousness": "om_a",
    "hitop_reality_distortion": "be_wager",
    "hitop_reality_distortion_delusions": "be_ch",
    "hitop_reality_distortion_hallucinations": "be_wager",
}
OUTCOME_LABELS = {
    "hitop_mistrust_suspiciousness": "Suspiciousness",
    "hitop_reality_distortion": "Reality distortion",
    "hitop_reality_distortion_delusions": "Delusions",
    "hitop_reality_distortion_hallucinations": "Hallucinations",
}
PREDICTOR_LABELS = {
    "om_a": "Omega-Advice",
    "be_wager": "Wager noise",
    "be_ch": "Choice noise",
}
OUTCOMES = list(PREDICTOR_BY_OUTCOME)


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=0)
    return (series - series.mean()) / sd if np.isfinite(sd) and sd > 0 else pd.Series(0.0, index=series.index)


def load_data() -> pd.DataFrame:
    params = pd.read_csv(PARAM_CSV)
    selected = params[params["parameter"].isin(set(PREDICTOR_BY_OUTCOME.values()))].copy()
    predictors = selected.pivot_table(
        index=["prolific_id", "round_name", "timepoint"],
        columns="parameter",
        values="estimate",
        aggfunc="first",
    ).reset_index()
    predictors.columns.name = None
    predictors["prolific_id"] = predictors["prolific_id"].astype(str)
    predictors["timepoint"] = predictors["timepoint"].str.lower()

    hitop_frames = []
    for timepoint, filename in [("t1", "hitop_scales_T1.csv"), ("t2", "hitop_scales_T2.csv")]:
        frame = pd.read_csv(HITOP_DIR / filename)[["prolific_id"] + OUTCOMES].copy()
        frame["prolific_id"] = frame["prolific_id"].astype(str)
        frame["timepoint"] = timepoint
        hitop_frames.append(frame)
    return predictors.merge(pd.concat(hitop_frames, ignore_index=True), on=["prolific_id", "timepoint"], how="inner")


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


def run_models(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome, predictor in PREDICTOR_BY_OUTCOME.items():
        work = data[["prolific_id", "timepoint", predictor, outcome]].dropna().copy()
        work[f"{predictor}_z"] = zscore(work[predictor])
        work[f"{outcome}_z"] = zscore(work[outcome])
        pooled = smf.ols(
            f"{outcome}_z ~ {predictor}_z + C(timepoint)", data=work
        ).fit(cov_type="cluster", cov_kwds={"groups": work["prolific_id"]})
        rows.append(model_row(pooled, outcome, predictor, "pooled_clustered"))

        for timepoint in ["t1", "t2"]:
            session = work[work["timepoint"] == timepoint].copy()
            session[f"{predictor}_z"] = zscore(session[predictor])
            session[f"{outcome}_z"] = zscore(session[outcome])
            model = smf.ols(f"{outcome}_z ~ {predictor}_z", data=session).fit()
            rows.append(model_row(model, outcome, predictor, timepoint))

    result = pd.DataFrame(rows)
    result["fdr_q_within_scope"] = np.nan
    for _, index in result.groupby("analysis_scope").groups.items():
        result.loc[index, "fdr_q_within_scope"] = multipletests(
            result.loc[index, "p_value"], method="fdr_bh"
        )[1]
    return result


def stars(q_value: float) -> str:
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def make_figure(results: pd.DataFrame) -> None:
    pooled = (
        results[results["analysis_scope"] == "pooled_clustered"]
        .set_index("outcome")
        .loc[OUTCOMES]
        .reset_index()
    )
    y = np.arange(len(pooled))
    colors = ["#3b66a0" if beta < 0 else "#b75a49" for beta in pooled["beta"]]
    labels = [
        f"{OUTCOME_LABELS[row.outcome]}\n({PREDICTOR_LABELS[row.predictor]})"
        for row in pooled.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(9.3, 6.5))
    ax.barh(y, pooled["beta"], color=colors, height=0.70)
    ax.errorbar(
        pooled["beta"], y, xerr=1.96 * pooled["se"], fmt="none",
        ecolor="#222222", elinewidth=1.2, capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#666666", linewidth=1)
    ax.set_xlim(-0.32, 0.32)
    ax.set_xlabel("Standardized beta")
    ax.set_title(
        "HiTOP domain analyses: selected parameter and noise effects",
        loc="left", fontsize=16, fontweight="bold", pad=12,
    )
    for index, row in pooled.iterrows():
        x = row["beta"] - 0.012 if row["beta"] < 0 else row["beta"] + 0.012
        ax.text(
            x, index,
            f"β={row['beta']:.2f}{stars(row['fdr_q_within_scope'])}\nq={row['fdr_q_within_scope']:.3f}, R²={row['r_squared']:.3f}",
            ha="right" if row["beta"] < 0 else "left", va="center", fontsize=9,
        )
    ax.grid(axis="x", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(0.5, 0.015, "* q < .05, ** q < .01, *** q < .001; error bars show 95% CIs.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_DIR / "figure_selected_parameter_noise_hitop_domains.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure_selected_parameter_noise_hitop_domains.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    results = run_models(data)
    results.to_csv(OUT_DIR / "selected_parameter_noise_hitop_domain_models.csv", index=False)
    data.to_csv(OUT_DIR / "selected_parameter_noise_hitop_merged.csv", index=False)
    make_figure(results)
    print(OUT_DIR / "figure_selected_parameter_noise_hitop_domains.png")


if __name__ == "__main__":
    main()
