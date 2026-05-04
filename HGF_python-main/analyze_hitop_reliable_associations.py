from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
MODEL_DIR = BASE / "hgf_baselinefixed_full"
STATE_DIR = MODEL_DIR / "state_reliability"
HITOP_DIR = BASE / "hitop" / "processed_data"
OUT_DIR = BASE / "hitop" / "model_associations_reliable"

HITOP_TARGETS = [
    "hitop_mistrust_suspiciousness",
    "hitop_reality_distortion",
    "hitop_reality_distortion_delusions",
    "hitop_reality_distortion_hallucinations",
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


def corr_stats(x: pd.Series, y: pd.Series) -> dict[str, float]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(df)
    if n < 3:
        return {
            "n": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }
    x_unique = df["x"].nunique()
    y_unique = df["y"].nunique()
    if x_unique < 2 or y_unique < 2:
        return {
            "n": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }
    pearson_r, pearson_p = pearsonr(df["x"], df["y"])
    spearman_rho, spearman_p = spearmanr(df["x"], df["y"])
    return {
        "n": n,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
    }


def load_reliable_quantities() -> tuple[pd.DataFrame, pd.DataFrame]:
    param_rel = pd.read_csv(MODEL_DIR / "test_retest_reliability_by_parameter.csv")
    state_rel = pd.read_csv(STATE_DIR / "state_test_retest_reliability.csv")

    reliable_params = param_rel.loc[
        (~param_rel["is_fixed"])
        & (param_rel["parameter"] != "m_r")
        & (param_rel["pearson_r_t1_t2"] >= 0.4)
        & (param_rel["icc3_1_t1_t2"] >= 0.4)
    ].copy()

    reliable_states = state_rel.loc[
        (state_rel["pearson_r_t1_t2"] >= 0.4)
        & (state_rel["icc3_1_t1_t2"] >= 0.4)
    ].copy()

    return reliable_params, reliable_states


def load_model_values(
    reliable_params: pd.DataFrame, reliable_states: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params_long = pd.read_csv(MODEL_DIR / "parameter_estimates_long_t1_t2.csv")
    params_long = params_long.loc[
        params_long["parameter"].isin(reliable_params["parameter"])
    ].copy()
    params_long["quantity_kind"] = "parameter"
    params_long["quantity_name"] = params_long["parameter"]
    params_long = params_long.rename(columns={"estimate": "value"})
    params_long = params_long[
        ["prolific_id", "timepoint", "quantity_kind", "quantity_name", "value"]
    ]

    states_wide = pd.read_csv(STATE_DIR / "state_estimates_wide_t1_t2.csv")
    states_wide = states_wide.loc[
        states_wide["state_metric"].isin(reliable_states["state_metric"])
    ].copy()
    states_long = states_wide.melt(
        id_vars=["prolific_id", "state_metric"],
        value_vars=["t1", "t2"],
        var_name="timepoint",
        value_name="value",
    )
    states_long["quantity_kind"] = "state"
    states_long["quantity_name"] = states_long["state_metric"]
    states_long = states_long[
        ["prolific_id", "timepoint", "quantity_kind", "quantity_name", "value"]
    ]

    all_long = pd.concat([params_long, states_long], ignore_index=True)
    all_wide = all_long.pivot_table(
        index=["prolific_id", "timepoint"],
        columns="quantity_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    all_wide.columns.name = None
    return all_long, all_wide


def load_hitop() -> pd.DataFrame:
    frames = []
    for timepoint, filename in [("t1", "hitop_scales_T1.csv"), ("t2", "hitop_scales_T2.csv")]:
        df = pd.read_csv(HITOP_DIR / filename)
        keep = ["prolific_id"] + HITOP_TARGETS
        df = df[keep].copy()
        df["timepoint"] = timepoint
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def association_table(
    merged: pd.DataFrame, quantities: pd.DataFrame, scope_name: str
) -> pd.DataFrame:
    rows = []
    quantity_names = quantities["quantity_name"].tolist()
    kind_map = dict(zip(quantities["quantity_name"], quantities["quantity_kind"]))
    for hitop_var in HITOP_TARGETS:
        for quantity in quantity_names:
            stats = corr_stats(merged[hitop_var], merged[quantity])
            rows.append(
                {
                    "analysis_scope": scope_name,
                    "hitop_measure": hitop_var,
                    "quantity_name": quantity,
                    "quantity_kind": kind_map[quantity],
                    **stats,
                }
            )
    out = pd.DataFrame(rows)
    out["fdr_q_pearson_within_scope"] = (
        out.groupby("analysis_scope")["pearson_p"].transform(bh_fdr)
    )
    out["fdr_q_spearman_within_scope"] = (
        out.groupby("analysis_scope")["spearman_p"].transform(bh_fdr)
    )
    return out.sort_values(["hitop_measure", "quantity_kind", "quantity_name"]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reliable_params, reliable_states = load_reliable_quantities()
    reliable_params = reliable_params.copy()
    reliable_params["quantity_name"] = reliable_params["parameter"]
    reliable_params["quantity_kind"] = "parameter"
    reliable_states = reliable_states.copy()
    reliable_states["quantity_name"] = reliable_states["state_metric"]
    reliable_states["quantity_kind"] = "state"

    selected = pd.concat(
        [
            reliable_params[
                [
                    "quantity_name",
                    "quantity_kind",
                    "pearson_r_t1_t2",
                    "icc3_1_t1_t2",
                    "mean_abs_delta",
                ]
            ],
            reliable_states[
                [
                    "quantity_name",
                    "quantity_kind",
                    "pearson_r_t1_t2",
                    "icc3_1_t1_t2",
                    "mean_abs_delta",
                ]
            ],
        ],
        ignore_index=True,
    ).sort_values(["quantity_kind", "quantity_name"])
    selected.to_csv(OUT_DIR / "selected_reliable_quantities.csv", index=False)

    _, model_wide = load_model_values(reliable_params, reliable_states)
    hitop_long = load_hitop()
    merged_long = hitop_long.merge(model_wide, on=["prolific_id", "timepoint"], how="inner")
    merged_long.to_csv(OUT_DIR / "hitop_model_merged_long.csv", index=False)

    quantity_info = selected[["quantity_name", "quantity_kind"]].drop_duplicates()

    results = []
    for timepoint in ["t1", "t2"]:
        tp = merged_long.loc[merged_long["timepoint"] == timepoint].copy()
        results.append(association_table(tp, quantity_info, timepoint))

    pooled = merged_long.copy()
    results.append(association_table(pooled, quantity_info, "pooled_t1_t2"))

    numeric_cols = HITOP_TARGETS + quantity_info["quantity_name"].tolist()
    averaged = (
        merged_long.groupby("prolific_id")[numeric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    averaged.to_csv(OUT_DIR / "hitop_model_merged_subject_average.csv", index=False)
    results.append(association_table(averaged, quantity_info, "subject_mean_t1_t2"))

    assoc = pd.concat(results, ignore_index=True)
    assoc.to_csv(OUT_DIR / "hitop_reliable_quantity_associations.csv", index=False)

    top = assoc.loc[assoc["analysis_scope"] == "subject_mean_t1_t2"].copy()
    top = top.reindex(
        top["pearson_r"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
    top.to_csv(OUT_DIR / "hitop_reliable_quantity_associations_subject_mean_ranked.csv", index=False)

    summary = {
        "n_hitop_t1": int((hitop_long["timepoint"] == "t1").sum()),
        "n_hitop_t2": int((hitop_long["timepoint"] == "t2").sum()),
        "n_merged_rows": int(len(merged_long)),
        "n_unique_subjects_merged": int(merged_long["prolific_id"].nunique()),
        "n_selected_parameters": int((selected["quantity_kind"] == "parameter").sum()),
        "n_selected_states": int((selected["quantity_kind"] == "state").sum()),
        "selected_parameters": reliable_params["parameter"].tolist(),
        "selected_states": reliable_states["state_metric"].tolist(),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
