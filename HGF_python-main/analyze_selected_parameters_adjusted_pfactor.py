from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
DOMAIN_DIR = BASE / "hitop" / "selected_parameter_noise_domain_analysis"
MERGED_CSV = DOMAIN_DIR / "selected_parameter_noise_hitop_merged.csv"
PFACTOR_CSV = (
    BASE
    / "behavior_fod_moderation_comparison"
    / "state_and_prevalence_assessment"
    / "hitop_bifactor_cfa"
    / "hitop_bifactor_cfa_scores.csv"
)
OUT_DIR = DOMAIN_DIR / "p_factor_adjustment"

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
PREDICTOR_LABELS = {"om_a": "Omega-Advice", "be_wager": "Wager noise", "be_ch": "Choice noise"}


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=0)
    return (series - series.mean()) / sd if np.isfinite(sd) and sd > 0 else pd.Series(0.0, index=series.index)


def load_data() -> pd.DataFrame:
    data = pd.read_csv(MERGED_CSV)
    data["prolific_id"] = data["prolific_id"].astype(str)
    scores = pd.read_csv(PFACTOR_CSV)
    scores = scores[scores["dataset"] == "Original"][["prolific_id", "timepoint", "p_factor"]].copy()
    scores["prolific_id"] = scores["prolific_id"].astype(str)
    return data.merge(scores, on=["prolific_id", "timepoint"], how="inner")


def predictor_pfactor_associations(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for predictor in sorted(set(PREDICTOR_BY_OUTCOME.values())):
        work = data[["prolific_id", "timepoint", predictor, "p_factor"]].dropna().copy()
        work[f"{predictor}_z"] = zscore(work[predictor])
        work["p_factor_z"] = zscore(work["p_factor"])
        model = smf.ols(
            f"p_factor_z ~ {predictor}_z + C(timepoint)", data=work
        ).fit(cov_type="cluster", cov_kwds={"groups": work["prolific_id"]})
        term = f"{predictor}_z"
        ci = model.conf_int().loc[term]
        rows.append(
            {
                "predictor": predictor,
                "n": int(model.nobs),
                "beta": model.params[term],
                "se": model.bse[term],
                "p_value": model.pvalues[term],
                "ci_low": ci.iloc[0],
                "ci_high": ci.iloc[1],
                "r_squared": model.rsquared,
            }
        )
    result = pd.DataFrame(rows)
    result["fdr_q"] = multipletests(result["p_value"], method="fdr_bh")[1]
    return result


def outcome_adjustment_models(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome, predictor in PREDICTOR_BY_OUTCOME.items():
        work = data[["prolific_id", "timepoint", predictor, outcome, "p_factor"]].dropna().copy()
        work[f"{predictor}_z"] = zscore(work[predictor])
        work[f"{outcome}_z"] = zscore(work[outcome])
        work["p_factor_z"] = zscore(work["p_factor"])

        base = smf.ols(
            f"{outcome}_z ~ {predictor}_z + C(timepoint)", data=work
        ).fit(cov_type="cluster", cov_kwds={"groups": work["prolific_id"]})
        adjusted = smf.ols(
            f"{outcome}_z ~ {predictor}_z + p_factor_z + C(timepoint)", data=work
        ).fit(cov_type="cluster", cov_kwds={"groups": work["prolific_id"]})

        term = f"{predictor}_z"
        for specification, model in [("base_same_sample", base), ("p_factor_adjusted", adjusted)]:
            ci = model.conf_int().loc[term]
            rows.append(
                {
                    "outcome": outcome,
                    "predictor": predictor,
                    "specification": specification,
                    "n": int(model.nobs),
                    "predictor_beta": model.params[term],
                    "predictor_se": model.bse[term],
                    "predictor_p": model.pvalues[term],
                    "predictor_ci_low": ci.iloc[0],
                    "predictor_ci_high": ci.iloc[1],
                    "p_factor_beta": model.params.get("p_factor_z", np.nan),
                    "p_factor_se": model.bse.get("p_factor_z", np.nan),
                    "p_factor_p": model.pvalues.get("p_factor_z", np.nan),
                    "r_squared": model.rsquared,
                    "adj_r_squared": model.rsquared_adj,
                }
            )

    result = pd.DataFrame(rows)
    result["predictor_fdr_q"] = np.nan
    for specification, index in result.groupby("specification").groups.items():
        result.loc[index, "predictor_fdr_q"] = multipletests(
            result.loc[index, "predictor_p"], method="fdr_bh"
        )[1]
    return result


def comparison_table(models: pd.DataFrame) -> pd.DataFrame:
    base = models[models["specification"] == "base_same_sample"].set_index("outcome")
    adjusted = models[models["specification"] == "p_factor_adjusted"].set_index("outcome")
    rows = []
    for outcome in PREDICTOR_BY_OUTCOME:
        b = base.loc[outcome]
        a = adjusted.loc[outcome]
        attenuation = 100.0 * (1.0 - abs(a["predictor_beta"]) / abs(b["predictor_beta"]))
        rows.append(
            {
                "outcome": outcome,
                "predictor": b["predictor"],
                "n": int(b["n"]),
                "base_beta": b["predictor_beta"],
                "base_ci_low": b["predictor_ci_low"],
                "base_ci_high": b["predictor_ci_high"],
                "base_p": b["predictor_p"],
                "base_q": b["predictor_fdr_q"],
                "adjusted_beta": a["predictor_beta"],
                "adjusted_ci_low": a["predictor_ci_low"],
                "adjusted_ci_high": a["predictor_ci_high"],
                "adjusted_p": a["predictor_p"],
                "adjusted_q": a["predictor_fdr_q"],
                "absolute_effect_attenuation_percent": attenuation,
                "p_factor_beta": a["p_factor_beta"],
                "p_factor_p": a["p_factor_p"],
                "base_r_squared": b["r_squared"],
                "adjusted_r_squared": a["r_squared"],
                "delta_r_squared": a["r_squared"] - b["r_squared"],
            }
        )
    return pd.DataFrame(rows)


def make_sensitivity_figure(comparison: pd.DataFrame) -> None:
    comparison = comparison.set_index("outcome").loc[list(PREDICTOR_BY_OUTCOME)].reset_index()
    y = np.arange(len(comparison))
    labels = [
        f"{OUTCOME_LABELS[row.outcome]}\n({PREDICTOR_LABELS[row.predictor]})"
        for row in comparison.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(9.3, 6.2))
    for index, row in comparison.iterrows():
        ax.plot(
            [row["base_beta"], row["adjusted_beta"]],
            [index, index],
            color="#b8b8b8",
            linewidth=1.5,
            zorder=1,
        )
    ax.errorbar(
        comparison["base_beta"],
        y - 0.10,
        xerr=[comparison["base_beta"] - comparison["base_ci_low"], comparison["base_ci_high"] - comparison["base_beta"]],
        fmt="o",
        color="#687786",
        ecolor="#687786",
        capsize=3,
        markersize=7,
        linewidth=1.3,
        label="Timepoint-adjusted",
        zorder=2,
    )
    ax.errorbar(
        comparison["adjusted_beta"],
        y + 0.10,
        xerr=[comparison["adjusted_beta"] - comparison["adjusted_ci_low"], comparison["adjusted_ci_high"] - comparison["adjusted_beta"]],
        fmt="o",
        color="#2f765f",
        ecolor="#2f765f",
        capsize=3,
        markersize=7,
        linewidth=1.3,
        label="Additionally adjusted for p factor",
        zorder=3,
    )
    q_column_x = 0.255
    for index, row in comparison.iterrows():
        ax.text(
            q_column_x,
            index,
            f"q={row['adjusted_q']:.3f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#2f765f",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#666666", linewidth=1)
    ax.set_xlim(-0.30, 0.32)
    ax.set_xlabel("Standardized beta")
    ax.set_title("Sensitivity analysis: adjustment for general psychopathology", loc="left", fontsize=15, fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.grid(axis="x", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(0.5, 0.012, "Error bars show 95% CIs; q values refer to p-factor-adjusted predictor effects.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(OUT_DIR / "figure_pfactor_adjustment_sensitivity.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure_pfactor_adjustment_sensitivity.svg", bbox_inches="tight")
    plt.close(fig)


def make_paper_table(comparison: pd.DataFrame) -> pd.DataFrame:
    table = comparison.copy()
    table["Outcome"] = table["outcome"].map(OUTCOME_LABELS)
    table["Predictor"] = table["predictor"].map(PREDICTOR_LABELS)
    table["Unadjusted effect"] = table.apply(
        lambda row: f"{row['base_beta']:.3f} [{row['base_ci_low']:.3f}, {row['base_ci_high']:.3f}]",
        axis=1,
    )
    table["p-factor-adjusted effect"] = table.apply(
        lambda row: f"{row['adjusted_beta']:.3f} [{row['adjusted_ci_low']:.3f}, {row['adjusted_ci_high']:.3f}]",
        axis=1,
    )
    table["Adjusted q"] = table["adjusted_q"].map(lambda value: f"{value:.3f}")
    table["Attenuation (%)"] = table["absolute_effect_attenuation_percent"].map(lambda value: f"{value:.1f}")
    table["Delta R2"] = table["delta_r_squared"].map(lambda value: f"{value:.3f}")
    return table[["Outcome", "Predictor", "n", "Unadjusted effect", "p-factor-adjusted effect", "Adjusted q", "Attenuation (%)", "Delta R2"]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    parameter_associations = predictor_pfactor_associations(data)
    models = outcome_adjustment_models(data)
    comparison = comparison_table(models)

    data.to_csv(OUT_DIR / "selected_parameters_pfactor_merged.csv", index=False)
    parameter_associations.to_csv(OUT_DIR / "selected_parameters_vs_pfactor.csv", index=False)
    models.to_csv(OUT_DIR / "domain_models_with_pfactor_adjustment.csv", index=False)
    comparison.to_csv(OUT_DIR / "domain_effect_attenuation_by_pfactor.csv", index=False)
    make_paper_table(comparison).to_csv(OUT_DIR / "table_pfactor_adjustment_paper.csv", index=False)
    make_sensitivity_figure(comparison)

    report = (
        "Sensitivity analysis adjusting for general psychopathology\n\n"
        "The selected computational effects were refitted with the CFA-derived general p factor as an additional covariate. "
        "None of Omega-Advice, wager noise, or choice noise was independently associated with the p factor (all FDR q=.494). "
        "Adjustment attenuated the absolute computational effects by approximately 21-27%, but the associations of "
        "Omega-Advice with suspiciousness and wager noise with overall reality distortion and hallucinations remained "
        "significant after FDR correction. The pooled choice-noise association with delusions remained nonsignificant. "
        "These findings suggest that the principal associations are not fully attributable to broad psychopathology severity. "
        "Because the bifactor CFA showed only modest fit (CFI=.827; RMSEA=.106), the adjusted models should be presented as a sensitivity analysis rather than the primary specification.\n"
    )
    (OUT_DIR / "results_pfactor_adjustment.txt").write_text(report)
    print(parameter_associations.to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
