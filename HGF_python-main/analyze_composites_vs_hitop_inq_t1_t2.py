from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
OUT_DIR = BASE / "composite_vs_hitop_inq_t1_t2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORIG_HITOP_T1 = BASE / "hitop" / "processed_data" / "hitop_scales_T1.csv"
ORIG_HITOP_T2 = BASE / "hitop" / "processed_data" / "hitop_scales_T2.csv"
ORIG_INQ_T1 = BASE / "hitop" / "processed_data" / "additional_suicidality_scales_T1.csv"
ORIG_INQ_T2 = BASE / "hitop" / "processed_data" / "additional_suicidality_scales_T2.csv"

RED_PSYCH_T1 = BASE / "hitop_batch2" / "processed_data" / "psych_scales_T1_extended.csv"
RED_PSYCH_T2 = BASE / "hitop_batch2" / "processed_data" / "psych_scales_T2_extended.csv"
RED_RAW_T1 = BASE / "MindMetrics- Batch2_T1" / "values_HITOP+++MindMetrics+Batch+2+-+Time+1_April+1,+2026_11.46.csv"
RED_RAW_T2 = BASE / "MindMetrics- Batch2_T2" / "HITOP+++MindMetrics+Batch+2+-+Time+2_May+4,+2026_15.15_values.csv"
ORIG_RAW_T1 = BASE / "hitop" / "raw_data" / "HITOP+++MindMetrics+-+Time+1_September+2,+2024_20.40.csv"

BEHAV_COMP = BASE / "behavior_test_retest_comparison" / "two_axis_behavior_model" / "behavior_two_axis_scores.csv"
PARAM_COMP = BASE / "hgf_parameter_two_axis_model" / "parameter_two_axis_scores.csv"

ANTAGONISM_FACETS = [
    "hitop_callousness",
    "hitop_dishonesty",
    "hitop_domineering",
    "hitop_entitlement",
    "hitop_exhibitionism",
    "hitop_grandiosity",
    "hitop_oppositionality",
    "hitop_social_aggression",
    "hitop_antisocial_behaviour",
]

COMPOSITES = [
    ("policy_axis", "Task-performance / policy axis"),
    ("wager_axis", "Wager / confidence axis"),
    ("advice_axis", "Advice-learning / uncertainty axis"),
    ("response_axis", "Observation / response axis"),
]

TARGETS = [
    ("hitop_distress_dysphoria", "Internalizing distress"),
    ("hitop_antagonistic_externalizing", "Antagonistic externalizing"),
    ("hitop_mistrust_suspiciousness", "Suspiciousness"),
    ("hitop_reality_distortion", "Thought disorder / reality distortion"),
    ("hitop_reality_distortion_delusions", "Delusions"),
    ("hitop_reality_distortion_hallucinations", "Hallucinations"),
    ("hitop_suicidality", "Suicidality"),
    ("inq_grand_sum", "INQ total"),
    ("inq_thwarted_belongigness", "INQ thwarted belongingness"),
    ("inq_perceived_burdensomness", "INQ perceived burdensomeness"),
]


def numify(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*$", str(value))
    if not match:
        return np.nan
    out = float(match.group(1))
    if out in {888.0, 999.0}:
        return np.nan
    return out


def load_raw_hitop(path: Path) -> pd.DataFrame:
    if "values_HITOP" in path.name:
        df = pd.read_csv(path, dtype=str)
        if "StartDate" in df.columns and len(df) > 2 and str(df.iloc[0].get("StartDate", "")).startswith("{"):
            df = pd.read_csv(path, dtype=str, skiprows=[1, 2])
    else:
        df = pd.read_csv(path, dtype=str)
    pid_col = "Q325" if "Q325" in df.columns else ("PROLIFIC_PID" if "PROLIFIC_PID" in df.columns else "ExternalReference")
    df = df[df[pid_col].notna()].copy()
    df = df[~df[pid_col].astype(str).str.contains("ImportId|PROLIFIC_PID|External Data Reference|Q325", na=False)].copy()
    if "Finished" in df.columns:
        finished = df["Finished"].astype(str).str.strip().str.lower()
        df = df[finished.isin(["true", "1"])].copy()
    item_cols = [c for c in df.columns if c.startswith("HiTOP_") or c.startswith("Ext_")]
    for col in item_cols:
        df[col] = df[col].map(numify)
    df = df.rename(columns={pid_col: "prolific_id"})
    df["prolific_id"] = df["prolific_id"].astype(str).str.strip()
    return df.reset_index(drop=True)


def build_antagonism_direct(df: pd.DataFrame, means: pd.Series | None = None, sds: pd.Series | None = None) -> pd.Series:
    facets = df[ANTAGONISM_FACETS].apply(pd.to_numeric, errors="coerce")
    if means is None:
        means = facets.mean()
    if sds is None:
        sds = facets.std(ddof=0).replace(0, np.nan)
    z = (facets - means) / sds
    return z.mean(axis=1)


def fit_reconstruction_model(X: pd.DataFrame, y: pd.Series):
    valid = y.notna()
    Xv = X.loc[valid].copy()
    yv = y.loc[valid].copy()
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-2, 3, 18))),
        ]
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    yhat = cross_val_predict(model, Xv, yv, cv=cv)
    corr = float(pd.Series(yv).corr(pd.Series(yhat)))
    r2 = float(1 - np.sum((yv - yhat) ** 2) / np.sum((yv - yv.mean()) ** 2))
    model.fit(Xv, yv)
    return model, {"n": int(len(yv)), "cv_pearson_r": corr, "cv_r2": r2}


def build_redeployed_additional_domains() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    orig_raw = load_raw_hitop(ORIG_RAW_T1)
    red_raw_t1 = load_raw_hitop(RED_RAW_T1)
    red_raw_t2 = load_raw_hitop(RED_RAW_T2)
    orig_proc_t1 = pd.read_csv(ORIG_HITOP_T1)
    orig_proc_t2 = pd.read_csv(ORIG_HITOP_T2)

    common_items = sorted(
        set(c for c in orig_raw.columns if c.startswith(("HiTOP_", "Ext_")))
        & set(c for c in red_raw_t1.columns if c.startswith(("HiTOP_", "Ext_")))
        & set(c for c in red_raw_t2.columns if c.startswith(("HiTOP_", "Ext_")))
    )

    orig_train = orig_proc_t1.merge(orig_raw[["prolific_id", *common_items]], on="prolific_id", how="inner")
    orig_train["hitop_antagonistic_externalizing"] = build_antagonism_direct(orig_train)

    X_train = orig_train[common_items]
    dist_model, dist_stats = fit_reconstruction_model(X_train, orig_train["hitop_distress_dysphoria"])
    ant_model, ant_stats = fit_reconstruction_model(X_train, orig_train["hitop_antagonistic_externalizing"])

    red_t1 = pd.DataFrame({"prolific_id": red_raw_t1["prolific_id"]})
    red_t1["hitop_distress_dysphoria"] = dist_model.predict(red_raw_t1[common_items])
    red_t1["hitop_antagonistic_externalizing"] = ant_model.predict(red_raw_t1[common_items])

    red_t2 = pd.DataFrame({"prolific_id": red_raw_t2["prolific_id"]})
    red_t2["hitop_distress_dysphoria"] = dist_model.predict(red_raw_t2[common_items])
    red_t2["hitop_antagonistic_externalizing"] = ant_model.predict(red_raw_t2[common_items])

    return red_t1, red_t2, {
        "hitop_distress_dysphoria": dist_stats,
        "hitop_antagonistic_externalizing": ant_stats,
    }


def load_composites_long() -> pd.DataFrame:
    beh = pd.read_csv(BEHAV_COMP)
    param = pd.read_csv(PARAM_COMP)
    rows = []
    for _, row in beh.iterrows():
        for tp in ["t1", "t2"]:
            rows.append({"prolific_id": row["prolific_id"], "dataset": row["dataset"], "timepoint": tp, "measure": "policy_axis", "value": row[f"policy_axis_{tp}"]})
            rows.append({"prolific_id": row["prolific_id"], "dataset": row["dataset"], "timepoint": tp, "measure": "wager_axis", "value": row[f"wager_axis_{tp}"]})
    for _, row in param.iterrows():
        for tp in ["t1", "t2"]:
            rows.append({"prolific_id": row["prolific_id"], "dataset": row["dataset"], "timepoint": tp, "measure": "advice_axis", "value": row[f"advice_axis_{tp}"]})
            rows.append({"prolific_id": row["prolific_id"], "dataset": row["dataset"], "timepoint": tp, "measure": "response_axis", "value": row[f"response_axis_{tp}"]})
    return pd.DataFrame(rows)


def load_psych_long() -> tuple[pd.DataFrame, dict]:
    # Original
    orig_t1 = pd.read_csv(ORIG_HITOP_T1)[["prolific_id", "hitop_distress_dysphoria", "hitop_mistrust_suspiciousness", "hitop_reality_distortion", "hitop_reality_distortion_delusions", "hitop_reality_distortion_hallucinations", "hitop_suicidality"]]
    orig_t1 = orig_t1.merge(pd.read_csv(ORIG_INQ_T1), on="prolific_id", how="left")
    orig_t1["hitop_antagonistic_externalizing"] = build_antagonism_direct(pd.read_csv(ORIG_HITOP_T1))
    orig_t1["dataset"] = "Original"
    orig_t1["timepoint"] = "t1"

    orig_t2_base = pd.read_csv(ORIG_HITOP_T2)
    facet_means = pd.read_csv(ORIG_HITOP_T1)[ANTAGONISM_FACETS].mean()
    facet_sds = pd.read_csv(ORIG_HITOP_T1)[ANTAGONISM_FACETS].std(ddof=0).replace(0, np.nan)
    orig_t2 = orig_t2_base[["prolific_id", "hitop_distress_dysphoria", "hitop_mistrust_suspiciousness", "hitop_reality_distortion", "hitop_reality_distortion_delusions", "hitop_reality_distortion_hallucinations", "hitop_suicidality"]]
    orig_t2 = orig_t2.merge(pd.read_csv(ORIG_INQ_T2), on="prolific_id", how="left")
    orig_t2["hitop_antagonistic_externalizing"] = build_antagonism_direct(orig_t2_base, means=facet_means, sds=facet_sds)
    orig_t2["dataset"] = "Original"
    orig_t2["timepoint"] = "t2"

    # Redeployed
    red_extra_t1, red_extra_t2, recon = build_redeployed_additional_domains()
    red_t1 = pd.read_csv(RED_PSYCH_T1).merge(red_extra_t1, on="prolific_id", how="left")
    red_t1["dataset"] = "Redeployed"
    red_t1["timepoint"] = "t1"
    red_t2 = pd.read_csv(RED_PSYCH_T2).merge(red_extra_t2, on="prolific_id", how="left")
    red_t2["dataset"] = "Redeployed"
    red_t2["timepoint"] = "t2"

    # Bifactor CFA
    bif = pd.read_csv(BASE / "behavior_fod_moderation_comparison" / "state_and_prevalence_assessment" / "hitop_bifactor_cfa" / "hitop_bifactor_cfa_scores.csv")

    frames = []
    for df in [orig_t1, orig_t2, red_t1, red_t2]:
        merged = df.merge(
            bif[["prolific_id", "dataset", "timepoint", "p_factor", "distress_specific", "antagonism_specific"]],
            on=["prolific_id", "dataset", "timepoint"],
            how="left",
        )
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True)
    return out, recon


def p_to_star(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    p = pvals.to_numpy(dtype=float)
    out = np.full(len(p), np.nan)
    mask = np.isfinite(p)
    pv = p[mask]
    if len(pv) == 0:
        return pd.Series(out, index=pvals.index)
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out_idx = np.where(mask)[0][order]
    out[out_idx] = q
    return pd.Series(out, index=pvals.index)


def compute_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in ["Original", "Redeployed"]:
        for tp in ["t1", "t2"]:
            sub = merged.loc[(merged["dataset"] == dataset) & (merged["timepoint"] == tp)].copy()
            for comp_key, comp_label in COMPOSITES:
                for target_key, target_label in TARGETS + [
                    ("p_factor", "General psychopathology"),
                    ("distress_specific", "Distress-specific"),
                    ("antagonism_specific", "Antagonism-specific"),
                ]:
                    x = pd.to_numeric(sub[comp_key], errors="coerce")
                    y = pd.to_numeric(sub[target_key], errors="coerce")
                    valid = x.notna() & y.notna()
                    n = int(valid.sum())
                    if n >= 3 and x[valid].std(ddof=0) > 0 and y[valid].std(ddof=0) > 0:
                        r, p = pearsonr(x[valid], y[valid])
                        r = float(r)
                        p = float(p)
                    else:
                        r = np.nan
                        p = np.nan
                    rows.append(
                        {
                            "dataset": dataset,
                            "timepoint": tp,
                            "composite": comp_key,
                            "composite_label": comp_label,
                            "target": target_key,
                            "target_label": target_label,
                            "n": n,
                            "pearson_r": r,
                            "p_value": p,
                        }
                    )
    out = pd.DataFrame(rows)
    out["q_value"] = (
        out.groupby(["dataset", "timepoint"], group_keys=False)["p_value"].apply(benjamini_hochberg)
    )
    out["sig"] = out["p_value"].map(p_to_star)
    return out


def make_figure(corr_df: pd.DataFrame) -> None:
    target_order = [label for _, label in TARGETS] + [
        "General psychopathology",
        "Distress-specific",
        "Antagonism-specific",
    ]
    comp_order = [label for _, label in COMPOSITES]
    fig, axes = plt.subplots(2, 2, figsize=(18, 9.5), sharex=True, sharey=True)
    panels = [("Original", "t1"), ("Original", "t2"), ("Redeployed", "t1"), ("Redeployed", "t2")]
    vmin, vmax = -0.4, 0.4

    for ax, (dataset, tp) in zip(axes.flat, panels):
        sub = corr_df.loc[(corr_df["dataset"] == dataset) & (corr_df["timepoint"] == tp)].copy()
        mat = sub.pivot(index="composite_label", columns="target_label", values="pearson_r").reindex(index=comp_order, columns=target_order)
        im = ax.imshow(mat.to_numpy(dtype=float), cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
        for i, comp in enumerate(comp_order):
            for j, target in enumerate(target_order):
                row = sub.loc[(sub["composite_label"] == comp) & (sub["target_label"] == target)]
                if len(row):
                    r = row["pearson_r"].iloc[0]
                    sig = row["sig"].iloc[0]
                    if np.isfinite(r):
                        ax.text(j, i, f"{r:.2f}{sig}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{dataset} {tp.upper()}")
        ax.set_xticks(np.arange(len(target_order)))
        ax.set_xticklabels(target_order, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(comp_order)))
        ax.set_yticklabels(comp_order)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Pearson r")
    fig.suptitle("Composite behavioral/model axes vs HiTOP spectra/composites and INQ", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "figure_composites_vs_hitop_inq_t1_t2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    comps = load_composites_long()
    psych, recon = load_psych_long()
    wide = comps.pivot_table(index=["prolific_id", "dataset", "timepoint"], columns="measure", values="value").reset_index()
    merged = wide.merge(psych, on=["prolific_id", "dataset", "timepoint"], how="inner")
    corr_df = compute_correlations(merged)

    merged.to_csv(OUT_DIR / "composites_psych_merged_long.csv", index=False)
    corr_df.to_csv(OUT_DIR / "composites_vs_hitop_inq_correlations.csv", index=False)
    make_figure(corr_df)

    summary = (
        corr_df.sort_values(["dataset", "timepoint", "q_value", "p_value"], na_position="last")
        .groupby(["dataset", "timepoint"])
        .head(8)
    )
    summary.to_csv(OUT_DIR / "top_composites_vs_hitop_inq_hits.csv", index=False)

    with open(OUT_DIR / "reconstruction_validation.json", "w") as f:
        json.dump(recon, f, indent=2)


if __name__ == "__main__":
    main()
