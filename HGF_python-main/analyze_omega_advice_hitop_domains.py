from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
PARAM_CSV = BASE / "hgf_baselinefixed_full" / "parameter_estimates_long_t1_t2.csv"
HITOP_DIR = BASE / "hitop" / "processed_data"
OUT_DIR = BASE / "hitop" / "omega_advice_domain_analysis"

PREDICTOR = "om_a"
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


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=0)
    return (series - series.mean()) / sd if np.isfinite(sd) and sd > 0 else pd.Series(0.0, index=series.index)


def load_data() -> pd.DataFrame:
    params = pd.read_csv(PARAM_CSV)
    omega = params[
        (params["parameter"] == PREDICTOR) & (params["group"] == "prc")
    ][["prolific_id", "round_name", "timepoint", "estimate", "is_fixed"]].copy()
    omega = omega.rename(columns={"estimate": PREDICTOR})
    omega["prolific_id"] = omega["prolific_id"].astype(str)
    omega["timepoint"] = omega["timepoint"].str.lower()

    hitop_frames = []
    for timepoint, filename in [("t1", "hitop_scales_T1.csv"), ("t2", "hitop_scales_T2.csv")]:
        frame = pd.read_csv(HITOP_DIR / filename)[["prolific_id"] + OUTCOMES].copy()
        frame["prolific_id"] = frame["prolific_id"].astype(str)
        frame["timepoint"] = timepoint
        hitop_frames.append(frame)
    hitop = pd.concat(hitop_frames, ignore_index=True)
    return omega.merge(hitop, on=["prolific_id", "timepoint"], how="inner")


def extract_model_row(model, outcome: str, scope: str) -> dict:
    term = f"{PREDICTOR}_z"
    ci = model.conf_int().loc[term]
    return {
        "outcome": outcome,
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
    for outcome in OUTCOMES:
        work = data[["prolific_id", "timepoint", PREDICTOR, outcome]].dropna().copy()
        work[f"{PREDICTOR}_z"] = zscore(work[PREDICTOR])
        work[f"{outcome}_z"] = zscore(work[outcome])
        pooled = smf.ols(
            f"{outcome}_z ~ {PREDICTOR}_z + C(timepoint)", data=work
        ).fit(cov_type="cluster", cov_kwds={"groups": work["prolific_id"]})
        rows.append(extract_model_row(pooled, outcome, "pooled_clustered"))

        for timepoint in ["t1", "t2"]:
            session = work[work["timepoint"] == timepoint].copy()
            session[f"{PREDICTOR}_z"] = zscore(session[PREDICTOR])
            session[f"{outcome}_z"] = zscore(session[outcome])
            model = smf.ols(f"{outcome}_z ~ {PREDICTOR}_z", data=session).fit()
            rows.append(extract_model_row(model, outcome, timepoint))

    result = pd.DataFrame(rows)
    result["fdr_q_within_scope"] = np.nan
    for scope, index in result.groupby("analysis_scope").groups.items():
        result.loc[index, "fdr_q_within_scope"] = multipletests(
            result.loc[index, "p_value"], method="fdr_bh"
        )[1]
    return result


def reliability_summary(data: pd.DataFrame) -> dict:
    wide = data.pivot_table(index="prolific_id", columns="timepoint", values=PREDICTOR, aggfunc="first")
    wide = wide.dropna(subset=["t1", "t2"])
    r, p = pearsonr(wide["t1"], wide["t2"])
    return {"n": int(len(wide)), "pearson_r": float(r), "pearson_p": float(p)}


def stars(q: float) -> str:
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def make_figure(results: pd.DataFrame) -> None:
    order = OUTCOMES
    styles = {
        "t1": {"label": "T1", "color": "#557fa8", "offset": -0.18},
        "t2": {"label": "T2", "color": "#d47b28", "offset": 0.18},
        "pooled_clustered": {"label": "Pooled", "color": "#2f765f", "offset": 0.0},
    }
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    y_base = np.arange(len(order))

    for scope in ["t1", "pooled_clustered", "t2"]:
        sub = results[results["analysis_scope"] == scope].set_index("outcome").loc[order]
        y = y_base + styles[scope]["offset"]
        ax.errorbar(
            sub["beta"],
            y,
            xerr=1.96 * sub["se"],
            fmt="o",
            markersize=7,
            color=styles[scope]["color"],
            ecolor=styles[scope]["color"],
            capsize=3,
            linewidth=1.4,
            label=styles[scope]["label"],
        )
        for yy, (_, row) in zip(y, sub.iterrows()):
            if row["fdr_q_within_scope"] < 0.05:
                ax.text(
                    row["beta"] - 0.012 if row["beta"] < 0 else row["beta"] + 0.012,
                    yy,
                    stars(row["fdr_q_within_scope"]),
                    ha="right" if row["beta"] < 0 else "left",
                    va="center",
                    fontsize=10,
                    color=styles[scope]["color"],
                )

    ax.set_yticks(y_base)
    ax.set_yticklabels([OUTCOME_LABELS[outcome] for outcome in order])
    ax.invert_yaxis()
    ax.axvline(0, color="#666666", linewidth=1)
    ax.set_xlim(-0.32, 0.18)
    ax.set_xlabel("Standardized beta for Omega-Advice")
    ax.set_title("Omega-Advice associations with HiTOP outcomes", loc="left", fontsize=16, fontweight="bold", pad=12)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.grid(axis="x", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(0.5, 0.015, "Error bars show 95% CIs; stars denote FDR-adjusted significance within analysis scope.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_DIR / "figure_omega_advice_hitop_domains.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure_omega_advice_hitop_domains.svg", bbox_inches="tight")
    plt.close(fig)


def make_pooled_bar_figure(results: pd.DataFrame) -> None:
    pooled = (
        results[results["analysis_scope"] == "pooled_clustered"]
        .set_index("outcome")
        .loc[OUTCOMES]
        .reset_index()
    )
    y = np.arange(len(pooled))
    colors = ["#3b66a0" if beta < 0 else "#b75a49" for beta in pooled["beta"]]

    fig, ax = plt.subplots(figsize=(8.8, 6.3))
    ax.barh(y, pooled["beta"], color=colors, height=0.72)
    ax.errorbar(
        pooled["beta"],
        y,
        xerr=1.96 * pooled["se"],
        fmt="none",
        ecolor="#222222",
        elinewidth=1.2,
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([OUTCOME_LABELS[outcome] for outcome in pooled["outcome"]])
    ax.invert_yaxis()
    ax.axvline(0, color="#666666", linewidth=1)
    ax.set_xlim(-0.32, 0.13)
    ax.set_xlabel("Standardized beta for Omega-Advice")
    ax.set_title(
        "HiTOP domain analyses: pooled Omega-Advice effects",
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )
    for index, row in pooled.iterrows():
        x = row["beta"] - 0.012 if row["beta"] < 0 else row["beta"] + 0.012
        ax.text(
            x,
            index,
            f"β={row['beta']:.2f}{stars(row['fdr_q_within_scope'])}\nq={row['fdr_q_within_scope']:.3f}, R²={row['r_squared']:.3f}",
            ha="right" if row["beta"] < 0 else "left",
            va="center",
            fontsize=9,
        )
    ax.grid(axis="x", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(0.5, 0.015, "* q < .05, ** q < .01, *** q < .001; error bars show 95% CIs.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_DIR / "figure_omega_advice_hitop_domains_pooled.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure_omega_advice_hitop_domains_pooled.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    results = run_models(data)
    reliability = reliability_summary(data)
    results.to_csv(OUT_DIR / "omega_advice_hitop_domain_models.csv", index=False)
    data.to_csv(OUT_DIR / "omega_advice_hitop_merged.csv", index=False)
    payload = {
        "predictor": PREDICTOR,
        "parameter_is_fixed_in_fit": bool(data["is_fixed"].all()),
        "n_session_rows": int(len(data)),
        "test_retest": reliability,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2))
    make_figure(results)
    make_pooled_bar_figure(results)
    print(json.dumps(payload, indent=2))
    print(OUT_DIR / "figure_omega_advice_hitop_domains.png")


if __name__ == "__main__":
    main()
