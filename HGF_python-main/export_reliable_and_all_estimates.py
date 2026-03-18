from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
MODEL_DIR = BASE / "hgf_baselinefixed_full"
STATE_DIR = MODEL_DIR / "state_reliability"
OUT_DIR = MODEL_DIR / "exported_estimates"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    params_long = pd.read_csv(MODEL_DIR / "parameter_estimates_long_t1_t2.csv")
    params_wide = pd.read_csv(MODEL_DIR / "parameter_estimates_wide_t1_t2.csv")
    param_rel = pd.read_csv(MODEL_DIR / "test_retest_reliability_by_parameter.csv")

    states_long = pd.read_csv(STATE_DIR / "state_estimates_long_t1_t2.csv")
    states_wide = pd.read_csv(STATE_DIR / "state_estimates_wide_t1_t2.csv")
    state_rel = pd.read_csv(STATE_DIR / "state_test_retest_reliability.csv")

    reliable_params = param_rel.loc[
        (~param_rel["is_fixed"])
        & (param_rel["pearson_r_t1_t2"] >= 0.4)
        & (param_rel["icc3_1_t1_t2"] >= 0.4),
        "parameter",
    ].drop_duplicates().tolist()

    reliable_states = state_rel.loc[
        (state_rel["pearson_r_t1_t2"] >= 0.4)
        & (state_rel["icc3_1_t1_t2"] >= 0.4),
        "state_metric",
    ].drop_duplicates().tolist()

    all_free_params_long = params_long.loc[~params_long["is_fixed"]].copy()
    all_free_params_long.to_csv(OUT_DIR / "all_free_parameter_estimates_long.csv", index=False)
    params_wide.to_csv(OUT_DIR / "all_parameter_estimates_wide.csv", index=False)

    reliable_params_long = params_long.loc[params_long["parameter"].isin(reliable_params)].copy()
    reliable_params_long.to_csv(OUT_DIR / "reliable_parameter_estimates_long.csv", index=False)

    reliable_params_wide = (
        reliable_params_long.pivot_table(
            index=["prolific_id", "round_name", "timepoint"],
            columns="parameter",
            values="estimate",
            aggfunc="first",
        )
        .reset_index()
    )
    reliable_params_wide.columns.name = None
    reliable_params_wide.to_csv(OUT_DIR / "reliable_parameter_estimates_wide.csv", index=False)

    reliable_states_long = states_long.loc[states_long["state_metric"].isin(reliable_states)].copy()
    reliable_states_long.to_csv(OUT_DIR / "reliable_state_estimates_long.csv", index=False)

    reliable_states_wide = states_wide.loc[states_wide["state_metric"].isin(reliable_states)].copy()
    reliable_states_wide.to_csv(OUT_DIR / "reliable_state_estimates_wide.csv", index=False)

    combined_long = pd.concat(
        [
            reliable_params_long.rename(
                columns={"parameter": "estimate_name", "group": "estimate_group"}
            )[
                ["prolific_id", "round_name", "timepoint", "estimate_group", "estimate_name", "estimate"]
            ],
            reliable_states_long.rename(
                columns={"state_metric": "estimate_name"}
            ).assign(estimate_group="state")[
                ["prolific_id", "round_name", "timepoint", "estimate_group", "estimate_name", "estimate"]
            ],
        ],
        ignore_index=True,
    ).sort_values(["estimate_group", "estimate_name", "prolific_id", "timepoint"])
    combined_long.to_csv(OUT_DIR / "reliable_parameters_and_states_long.csv", index=False)

    summary = {
        "n_all_parameters_total": int(params_long["parameter"].nunique()),
        "n_all_parameters_free": int(all_free_params_long["parameter"].nunique()),
        "n_reliable_parameters": int(len(reliable_params)),
        "n_reliable_states": int(len(reliable_states)),
        "reliable_parameters": reliable_params,
        "reliable_states": reliable_states,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
