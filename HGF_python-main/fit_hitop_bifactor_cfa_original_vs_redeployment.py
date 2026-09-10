from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from semopy import Model, calc_stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
OUT_DIR = BASE / "behavior_fod_moderation_comparison" / "state_and_prevalence_assessment" / "hitop_bifactor_cfa"

ORIG_RAW_T1 = BASE / "hitop" / "raw_data" / "HITOP+++MindMetrics+-+Time+1_September+2,+2024_20.40.csv"
ORIG_PROC_T1 = BASE / "hitop" / "processed_data" / "hitop_scales_T1.csv"
ORIG_PROC_T2 = BASE / "hitop" / "processed_data" / "hitop_scales_T2.csv"
RED_RAW_T1 = BASE / "MindMetrics- Batch2_T1" / "values_HITOP+++MindMetrics+Batch+2+-+Time+1_April+1,+2026_11.46.csv"
RED_RAW_T2 = BASE / "MindMetrics- Batch2_T2" / "HITOP+++MindMetrics+Batch+2+-+Time+2_May+4,+2026_15.15_values.csv"

OUT_SCORES = OUT_DIR / "hitop_bifactor_cfa_scores.csv"
OUT_SUMMARY = OUT_DIR / "hitop_bifactor_cfa_summary.csv"
OUT_LOADINGS = OUT_DIR / "hitop_bifactor_cfa_parameters.csv"
OUT_FIT = OUT_DIR / "hitop_bifactor_cfa_fit_stats.json"
OUT_RECON = OUT_DIR / "hitop_bifactor_cfa_reconstruction_validation.json"
OUT_TESTS = OUT_DIR / "hitop_bifactor_cfa_distribution_tests.json"
OUT_FIG = OUT_DIR / "figure_hitop_bifactor_cfa_original_vs_redeployment.png"
OUT_MODEL = OUT_DIR / "hitop_bifactor_cfa_model.txt"

DISTRESS_INDICATORS = [
    "hitop_appetite_loss",
    "hitop_dissociation",
    "hitop_distress_dysphoria",
    "hitop_excoriation",
    "hitop_hypervigilance",
    "hitop_insomnia",
    "hitop_low_sexual_arousal",
    "hitop_low_sexual_interest",
    "hitop_nightmares",
    "hitop_nssi",
    "hitop_suicidality",
    "hitop_trauma_reactions",
]

ANTAGONISM_INDICATORS = [
    "hitop_antisocial_behaviour",
    "hitop_callousness",
    "hitop_dishonesty",
    "hitop_domineering",
    "hitop_entitlement",
    "hitop_exhibitionism",
    "hitop_grandiosity",
    "hitop_oppositionality",
    "hitop_social_aggression",
]

ALL_INDICATORS = DISTRESS_INDICATORS + ANTAGONISM_INDICATORS

GROUPS = [
    ("Original", "t1", "Original T1", "#4C78A8"),
    ("Original", "t2", "Original T2", "#8FB9E0"),
    ("Redeployed", "t1", "Redeployed T1", "#E76F51"),
    ("Redeployed", "t2", "Redeployed T2", "#F2A07E"),
]

SCORES = [
    ("p_factor", "General factor"),
    ("distress_specific", "Distress-specific"),
    ("antagonism_specific", "Antagonism-specific"),
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
    df = pd.read_csv(path, dtype=str)
    if "StartDate" in df.columns and len(df) > 2 and str(df.iloc[0].get("StartDate", "")).startswith("{"):
        df = pd.read_csv(path, dtype=str, skiprows=[1, 2])
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


def fit_reconstruction_model(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict[str, float]]:
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
    mae = float(np.mean(np.abs(yv - yhat)))
    r2 = float(1 - np.sum((yv - yhat) ** 2) / np.sum((yv - yv.mean()) ** 2))
    model.fit(Xv, yv)
    alpha = float(model.named_steps["ridge"].alpha_)
    return model, {"n": int(len(yv)), "cv_pearson_r": corr, "cv_mae": mae, "cv_r2": r2, "ridge_alpha": alpha}


def reconstruct_redeployed_indicators() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    orig_raw = load_raw_hitop(ORIG_RAW_T1)
    red_raw_t1 = load_raw_hitop(RED_RAW_T1)
    red_raw_t2 = load_raw_hitop(RED_RAW_T2)
    orig_proc_t1 = pd.read_csv(ORIG_PROC_T1)

    common_items = sorted(
        set(c for c in orig_raw.columns if c.startswith(("HiTOP_", "Ext_")))
        & set(c for c in red_raw_t1.columns if c.startswith(("HiTOP_", "Ext_")))
        & set(c for c in red_raw_t2.columns if c.startswith(("HiTOP_", "Ext_")))
    )

    train = orig_proc_t1[["prolific_id", *ALL_INDICATORS]].merge(
        orig_raw[["prolific_id", *common_items]], on="prolific_id", how="inner"
    )

    models: dict[str, Pipeline] = {}
    stats_out: dict[str, dict[str, float]] = {}
    for indicator in ALL_INDICATORS:
        model, stats_dict = fit_reconstruction_model(train[common_items], train[indicator])
        models[indicator] = model
        stats_out[indicator] = stats_dict

    red_t1 = pd.DataFrame({"prolific_id": red_raw_t1["prolific_id"].astype(str)})
    red_t2 = pd.DataFrame({"prolific_id": red_raw_t2["prolific_id"].astype(str)})
    for indicator, model in models.items():
        red_t1[indicator] = model.predict(red_raw_t1[common_items])
        red_t2[indicator] = model.predict(red_raw_t2[common_items])

    return red_t1, red_t2, stats_out


def build_model_syntax() -> str:
    g = "p_factor =~ " + " + ".join(ALL_INDICATORS)
    d = "distress_specific =~ " + " + ".join(DISTRESS_INDICATORS)
    a = "antagonism_specific =~ " + " + ".join(ANTAGONISM_INDICATORS)
    orth = [
        "p_factor ~~ 0*distress_specific",
        "p_factor ~~ 0*antagonism_specific",
        "distress_specific ~~ 0*antagonism_specific",
    ]
    return "\n".join([g, d, a, *orth])


def fit_bifactor() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    orig_t1 = pd.read_csv(ORIG_PROC_T1)[["prolific_id", *ALL_INDICATORS]].copy()
    orig_t1["dataset"] = "Original"
    orig_t1["timepoint"] = "t1"
    orig_t2 = pd.read_csv(ORIG_PROC_T2)[["prolific_id", *ALL_INDICATORS]].copy()
    orig_t2["dataset"] = "Original"
    orig_t2["timepoint"] = "t2"
    red_t1, red_t2, recon = reconstruct_redeployed_indicators()
    red_t1["dataset"] = "Redeployed"
    red_t1["timepoint"] = "t1"
    red_t2["dataset"] = "Redeployed"
    red_t2["timepoint"] = "t2"

    pooled = pd.concat([orig_t1, orig_t2, red_t1, red_t2], ignore_index=True)
    pooled["prolific_id"] = pooled["prolific_id"].astype(str)

    fit_df = pd.concat([orig_t1, orig_t2], ignore_index=True)
    fit_X = fit_df[ALL_INDICATORS].apply(pd.to_numeric, errors="coerce")
    fit_X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(fit_X), columns=ALL_INDICATORS)

    syntax = build_model_syntax()
    OUT_MODEL.write_text(syntax)

    model = Model(syntax)
    model.fit(fit_X, obj="MLW")
    stats_df = calc_stats(model)
    fit_stats = {k: float(stats_df.loc["Value", k]) for k in stats_df.columns}

    param_table = model.inspect()

    all_X = pooled[ALL_INDICATORS].apply(pd.to_numeric, errors="coerce")
    all_X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(all_X), columns=ALL_INDICATORS)
    factor_scores = model.predict_factors(all_X)
    factor_scores.columns = [str(c) for c in factor_scores.columns]

    out = pooled[["prolific_id", "dataset", "timepoint"]].copy()
    out["p_factor"] = factor_scores["p_factor"].to_numpy()
    out["distress_specific"] = factor_scores["distress_specific"].to_numpy()
    out["antagonism_specific"] = factor_scores["antagonism_specific"].to_numpy()

    meta = {"fit_stats": fit_stats, "reconstruction_validation": recon}
    return out, meta, param_table


def summarize_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, label in SCORES:
        for dataset, timepoint, group_label, _ in GROUPS:
            s = pd.to_numeric(
                df.loc[(df["dataset"] == dataset) & (df["timepoint"] == timepoint), metric],
                errors="coerce",
            ).dropna()
            if s.empty:
                continue
            rows.append(
                {
                    "metric": metric,
                    "label": label,
                    "dataset": dataset,
                    "timepoint": timepoint,
                    "group_label": group_label,
                    "n": int(len(s)),
                    "mean": float(s.mean()),
                    "sd": float(s.std(ddof=0)),
                    "median": float(s.median()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "q25": float(s.quantile(0.25)),
                    "q75": float(s.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def distribution_tests(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for metric, _label in SCORES:
        out[metric] = {}
        for tp in ["t1", "t2"]:
            a = df.loc[(df["dataset"] == "Original") & (df["timepoint"] == tp), metric].dropna()
            b = df.loc[(df["dataset"] == "Redeployed") & (df["timepoint"] == tp), metric].dropna()
            ks = stats.ks_2samp(a, b)
            out[metric][tp] = {
                "n_original": int(len(a)),
                "n_redeployed": int(len(b)),
                "mean_original": float(a.mean()),
                "mean_redeployed": float(b.mean()),
                "cohens_d": float((a.mean() - b.mean()) / np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2)),
                "ks_stat": float(ks.statistic),
                "ks_p": float(ks.pvalue),
            }
    return out


def plot_scores(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(SCORES), 1, figsize=(12, 12))
    rng = np.random.default_rng(42)
    for ax, (metric, label) in zip(axes, SCORES):
        positions = np.arange(1, len(GROUPS) + 1)
        values = []
        colors = []
        ns = []
        for dataset, timepoint, _group_label, color in GROUPS:
            s = pd.to_numeric(
                df.loc[(df["dataset"] == dataset) & (df["timepoint"] == timepoint), metric],
                errors="coerce",
            ).dropna()
            values.append(s.to_numpy())
            colors.append(color)
            ns.append(len(s))

        parts = ax.violinplot(values, positions=positions, showmeans=False, showmedians=False, showextrema=False)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.35)

        box = ax.boxplot(
            values,
            positions=positions,
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.3},
            whiskerprops={"color": "#555", "linewidth": 1.0},
            capprops={"color": "#555", "linewidth": 1.0},
            boxprops={"linewidth": 1.0, "color": "#555"},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        for x, vals, color in zip(positions, values, colors):
            if len(vals) == 0:
                continue
            jitter = rng.normal(0, 0.04, size=len(vals))
            sample_idx = np.arange(len(vals))
            if len(vals) > 250:
                sample_idx = rng.choice(len(vals), size=250, replace=False)
            ax.scatter(
                np.full(len(sample_idx), x) + jitter[sample_idx],
                vals[sample_idx],
                s=9,
                color=color,
                alpha=0.18,
                edgecolors="none",
                zorder=3,
            )

        for x, n in zip(positions, ns):
            ax.text(x, 0.98, f"n={n}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9, color="#444")

        ax.axhline(0, color="#666", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(label, fontsize=13, pad=8)
        ax.set_xticks(positions)
        ax.set_xticklabels([g[2] for g in GROUPS], fontsize=10)
        ax.grid(axis="y", alpha=0.18)
        ax.set_ylabel("Factor score")

    fig.suptitle(
        "CFA bifactor HiTOP latent distributions across original and redeployed samples",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965], h_pad=1.7)
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores, meta, param_table = fit_bifactor()
    summary = summarize_scores(scores)
    tests = distribution_tests(scores)

    scores.to_csv(OUT_SCORES, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    param_table.to_csv(OUT_LOADINGS, index=False)
    OUT_FIT.write_text(json.dumps(meta["fit_stats"], indent=2))
    OUT_RECON.write_text(json.dumps(meta["reconstruction_validation"], indent=2))
    OUT_TESTS.write_text(json.dumps(tests, indent=2))
    plot_scores(scores)

    print(OUT_FIG)
    print(OUT_SCORES)
    print(OUT_SUMMARY)
    print(OUT_LOADINGS)
    print(OUT_FIT)
    print(OUT_RECON)
    print(OUT_TESTS)


if __name__ == "__main__":
    main()
