from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
ORIG_DIR = BASE / "hgf_baselinefixed_full"
BATCH2_DIR = BASE / "hgf_batch2_t1_baselinefixed_full"
OUT_DIR = BASE / "hgf_batch2_t1_baselinefixed_full" / "comparison_to_original"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    orig_recovery = load_json(ORIG_DIR / "aggregate_metrics.json")
    batch2_recovery = load_json(BATCH2_DIR / "aggregate_metrics.json")
    orig_ppc = load_json(ORIG_DIR / "posterior_predictive_checks" / "aggregate_metrics.json")
    batch2_ppc = load_json(BATCH2_DIR / "posterior_predictive_checks" / "aggregate_metrics.json")

    recovery_keys = [
        "n_successful_fits",
        "n_unique_subjects_any",
        "mean_prc_recovery_corr",
        "mean_obs_recovery_corr",
        "mean_all_recovery_corr",
        "mean_prc_recovery_rmse",
        "mean_obs_recovery_rmse",
        "mean_all_recovery_rmse",
    ]
    ppc_keys = [
        "n_sessions",
        "n_trials",
        "choice_auc",
        "choice_brier",
        "choice_neg_log_lik",
        "wager_rmse",
        "wager_corr",
        "wager_r2",
        "session_choice_mean_corr_obs_vs_pred",
        "session_wager_mean_corr_obs_vs_pred",
        "choice_mean_ppc_coverage",
        "wager_mean_ppc_coverage",
        "mean_abs_choice_calibration_error",
    ]

    rows = []
    for domain, src, new, keys in [
        ("recovery", orig_recovery, batch2_recovery, recovery_keys),
        ("ppc", orig_ppc, batch2_ppc, ppc_keys),
    ]:
        for key in keys:
            old_val = src.get(key)
            new_val = new.get(key)
            delta = None
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                delta = new_val - old_val
            rows.append(
                {
                    "domain": domain,
                    "metric": key,
                    "original_dataset": old_val,
                    "batch2_dataset": new_val,
                    "delta_batch2_minus_original": delta,
                }
            )

    comp = pd.DataFrame(rows)
    comp.to_csv(OUT_DIR / "aggregate_metric_comparison.csv", index=False)

    orig_obs = pd.read_csv(ORIG_DIR / "parameter_recovery_report.csv")
    batch2_obs = pd.read_csv(BATCH2_DIR / "parameter_recovery_report.csv")
    orig_obs = orig_obs[orig_obs["group"] == "obs"][["parameter", "pearson_r", "icc3_1", "rmse"]].copy()
    batch2_obs = batch2_obs[batch2_obs["group"] == "obs"][["parameter", "pearson_r", "icc3_1", "rmse"]].copy()
    merged_obs = orig_obs.merge(batch2_obs, on="parameter", suffixes=("_original", "_batch2"))
    merged_obs["delta_pearson_r"] = merged_obs["pearson_r_batch2"] - merged_obs["pearson_r_original"]
    merged_obs["delta_icc3_1"] = merged_obs["icc3_1_batch2"] - merged_obs["icc3_1_original"]
    merged_obs["delta_rmse"] = merged_obs["rmse_batch2"] - merged_obs["rmse_original"]
    merged_obs.to_csv(OUT_DIR / "observation_parameter_recovery_comparison.csv", index=False)

    summary = {
        "original_dir": str(ORIG_DIR),
        "batch2_dir": str(BATCH2_DIR),
        "n_rows_aggregate_comparison": int(len(comp)),
        "n_observation_parameters_compared": int(len(merged_obs)),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
