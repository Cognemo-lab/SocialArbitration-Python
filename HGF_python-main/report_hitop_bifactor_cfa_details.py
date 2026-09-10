from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import ncx2
from semopy import Model, calc_stats
from sklearn.impute import SimpleImputer

from fit_hitop_bifactor_cfa_original_vs_redeployment import (
    ALL_INDICATORS,
    DISTRESS_INDICATORS,
    ANTAGONISM_INDICATORS,
    ORIG_PROC_T1,
    ORIG_PROC_T2,
    build_model_syntax,
)


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
OUT_DIR = (
    BASE
    / "behavior_fod_moderation_comparison"
    / "state_and_prevalence_assessment"
    / "hitop_bifactor_cfa"
)


def rmsea_ci(chi2: float, df: int, n: int, confidence: float = 0.90) -> tuple[float, float]:
    alpha = 1.0 - confidence

    def solve(target: float) -> float:
        f = lambda ncp: ncx2.cdf(chi2, df, ncp) - target
        if f(0.0) <= 0:
            return 0.0
        upper = max(chi2, 1.0)
        while f(upper) > 0:
            upper *= 2.0
        return float(brentq(f, 0.0, upper))

    lower_ncp = solve(1.0 - alpha / 2.0)
    upper_ncp = solve(alpha / 2.0)
    return (
        float(np.sqrt(max(lower_ncp, 0.0) / (df * (n - 1)))),
        float(np.sqrt(max(upper_ncp, 0.0) / (df * (n - 1)))),
    )


def main() -> None:
    t1 = pd.read_csv(ORIG_PROC_T1)[ALL_INDICATORS].apply(pd.to_numeric, errors="coerce")
    t2 = pd.read_csv(ORIG_PROC_T2)[ALL_INDICATORS].apply(pd.to_numeric, errors="coerce")
    raw = pd.concat([t1, t2], ignore_index=True)
    imputer = SimpleImputer(strategy="median")
    fit_x = pd.DataFrame(imputer.fit_transform(raw), columns=ALL_INDICATORS)

    model = Model(build_model_syntax())
    model.fit(fit_x, obj="MLW")
    fit = calc_stats(model).loc["Value"].to_dict()

    observed_order = list(model.vars["observed"])
    sample_cov = np.cov(fit_x[observed_order].to_numpy(), rowvar=False, ddof=0)
    implied_cov = model.calc_sigma()[0]
    scale = np.sqrt(np.outer(np.diag(sample_cov), np.diag(sample_cov)))
    standardized_residual = (sample_cov - implied_cov) / scale
    lower = np.tril_indices_from(standardized_residual)
    srmr = float(np.sqrt(np.mean(standardized_residual[lower] ** 2)))

    n = len(fit_x)
    df = int(fit["DoF"])
    rmsea_low, rmsea_high = rmsea_ci(float(fit["chi2"]), df, n)
    missing_by_indicator = raw.isna().sum()

    report = {
        "software": "semopy 2.3.11",
        "estimator": "MLW (Wishart maximum-likelihood objective), SLSQP optimizer",
        "fit_sample": {
            "source": "Original deployment T1 and T2 pooled as rows",
            "n_t1": int(len(t1)),
            "n_t2": int(len(t2)),
            "n_total": n,
            "rows_with_any_missing": int(raw.isna().any(axis=1).sum()),
            "missing_cells": int(raw.isna().sum().sum()),
            "missing_data_handling": "Indicator-wise median imputation before CFA; not FIML",
        },
        "indicators": {
            "general_factor": ALL_INDICATORS,
            "distress_specific": DISTRESS_INDICATORS,
            "antagonism_specific": ANTAGONISM_INDICATORS,
            "missing_counts": {k: int(v) for k, v in missing_by_indicator.items() if v > 0},
        },
        "identification": {
            "marker_loadings_fixed_to_one": {
                "p_factor": "hitop_appetite_loss",
                "distress_specific": "hitop_appetite_loss",
                "antagonism_specific": "hitop_antisocial_behaviour",
            },
            "factor_covariances": "All three latent-factor covariances fixed to zero",
            "factor_variances": "Freely estimated",
            "indicator_residual_covariances": "Fixed to zero by omission",
            "means_intercepts": "No mean structure; latent means implicitly zero",
            "longitudinal_constraints": "None; repeated T1/T2 rows treated as independent in CFA estimation",
        },
        "factor_scores": {
            "method": "semopy predict_factors fast MAP scores",
            "centering": "Scores mean-centered by semopy",
            "required_input": "Complete indicator matrix after median imputation",
            "redeployment_note": "Redeployment indicators were first reconstructed from common items using cross-validated ridge models; the CFA loadings were estimated only in the Original sample.",
        },
        "fit_statistics": {
            **{k: float(v) for k, v in fit.items()},
            "SRMR_manual": srmr,
            "SRMR_definition": "RMS of lower-triangular standardized covariance residuals, including variances",
            "RMSEA_90CI_low": rmsea_low,
            "RMSEA_90CI_high": rmsea_high,
            "n_free_parameters": int(len(ALL_INDICATORS) * (len(ALL_INDICATORS) + 1) / 2 - df),
        },
    }
    (OUT_DIR / "hitop_bifactor_cfa_complete_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
