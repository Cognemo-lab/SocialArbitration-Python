from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
OUT_DIR = BASE / "behavior_fod_moderation_comparison" / "state_and_prevalence_assessment"
OUT_FIG = OUT_DIR / "figure_hitop_bifactor_original_vs_redeployment.png"
OUT_SCORES = OUT_DIR / "hitop_bifactor_scores_original_vs_redeployment.csv"
OUT_SUMMARY = OUT_DIR / "hitop_bifactor_summary_original_vs_redeployment.csv"
OUT_VALIDATION = OUT_DIR / "hitop_bifactor_reconstruction_validation.json"
OUT_TESTS = OUT_DIR / "hitop_bifactor_distribution_tests.json"

ORIG_RAW_T1 = BASE / "hitop" / "raw_data" / "HITOP+++MindMetrics+-+Time+1_September+2,+2024_20.40.csv"
ORIG_PROC_T1 = BASE / "hitop" / "processed_data" / "hitop_scales_T1.csv"
ORIG_PROC_T2 = BASE / "hitop" / "processed_data" / "hitop_scales_T2.csv"
RED_RAW_T1 = BASE / "MindMetrics- Batch2_T1" / "values_HITOP+++MindMetrics+Batch+2+-+Time+1_April+1,+2026_11.46.csv"
RED_RAW_T2 = BASE / "MindMetrics- Batch2_T2" / "HITOP+++MindMetrics+Batch+2+-+Time+2_May+4,+2026_15.15_values.csv"

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
    ("general_factor", "General factor"),
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


def reconstruct_redeployed_scales() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
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


def orient_component(scores: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, float]:
    corr = np.corrcoef(scores, anchor)[0, 1]
    if np.isnan(corr):
        corr = 0.0
    sign = 1.0 if corr >= 0 else -1.0
    return scores * sign, sign


def build_bifactor_scores() -> tuple[pd.DataFrame, dict[str, dict]]:
    orig_t1 = pd.read_csv(ORIG_PROC_T1)[["prolific_id", *ALL_INDICATORS]].copy()
    orig_t1["dataset"] = "Original"
    orig_t1["timepoint"] = "t1"

    orig_t2 = pd.read_csv(ORIG_PROC_T2)[["prolific_id", *ALL_INDICATORS]].copy()
    orig_t2["dataset"] = "Original"
    orig_t2["timepoint"] = "t2"

    red_t1, red_t2, recon_stats = reconstruct_redeployed_scales()
    red_t1["dataset"] = "Redeployed"
    red_t1["timepoint"] = "t1"
    red_t2["dataset"] = "Redeployed"
    red_t2["timepoint"] = "t2"

    all_df = pd.concat([orig_t1, orig_t2, red_t1, red_t2], ignore_index=True)
    all_df["prolific_id"] = all_df["prolific_id"].astype(str)

    fit_df = pd.concat([orig_t1, orig_t2], ignore_index=True)
    fit_X = fit_df[ALL_INDICATORS].apply(pd.to_numeric, errors="coerce")
    orig_mean = fit_X.mean()
    orig_sd = fit_X.std(ddof=0).replace(0, np.nan)

    Z = (all_df[ALL_INDICATORS].apply(pd.to_numeric, errors="coerce") - orig_mean) / orig_sd
    Z = Z.replace([np.inf, -np.inf], np.nan)
    Z = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(Z), columns=ALL_INDICATORS, index=all_df.index)

    pca_g = PCA(n_components=1)
    g_scores_orig = pca_g.fit_transform(((fit_X - orig_mean) / orig_sd).fillna(0.0))[:, 0]
    g_scores_all = pca_g.transform(Z)[:, 0]
    domain_anchor_orig = ((fit_X[DISTRESS_INDICATORS].mean(axis=1) + fit_X[ANTAGONISM_INDICATORS].mean(axis=1)) / 2).to_numpy()
    g_scores_orig, g_sign = orient_component(g_scores_orig, domain_anchor_orig)
    g_scores_all = g_scores_all * g_sign
    g_loadings = pd.Series(pca_g.components_[0] * g_sign, index=ALL_INDICATORS)

    distress_resid_orig = ((fit_X[DISTRESS_INDICATORS] - orig_mean[DISTRESS_INDICATORS]) / orig_sd[DISTRESS_INDICATORS]).fillna(0.0)
    for col in DISTRESS_INDICATORS:
        distress_resid_orig[col] = distress_resid_orig[col] - g_scores_orig * g_loadings[col]
    distress_pca = PCA(n_components=1)
    d_scores_orig = distress_pca.fit_transform(distress_resid_orig)[:, 0]
    distress_anchor_orig = fit_X[DISTRESS_INDICATORS].mean(axis=1).to_numpy()
    d_scores_orig, d_sign = orient_component(d_scores_orig, distress_anchor_orig)
    d_loadings = pd.Series(distress_pca.components_[0] * d_sign, index=DISTRESS_INDICATORS)

    distress_resid_all = Z[DISTRESS_INDICATORS].copy()
    for col in DISTRESS_INDICATORS:
        distress_resid_all[col] = distress_resid_all[col] - g_scores_all * g_loadings[col]
    d_scores_all = distress_pca.transform(distress_resid_all)[:, 0] * d_sign

    antagonism_resid_orig = ((fit_X[ANTAGONISM_INDICATORS] - orig_mean[ANTAGONISM_INDICATORS]) / orig_sd[ANTAGONISM_INDICATORS]).fillna(0.0)
    for col in ANTAGONISM_INDICATORS:
        antagonism_resid_orig[col] = antagonism_resid_orig[col] - g_scores_orig * g_loadings[col]
    antagonism_pca = PCA(n_components=1)
    a_scores_orig = antagonism_pca.fit_transform(antagonism_resid_orig)[:, 0]
    antagonism_anchor_orig = fit_X[ANTAGONISM_INDICATORS].mean(axis=1).to_numpy()
    a_scores_orig, a_sign = orient_component(a_scores_orig, antagonism_anchor_orig)
    a_loadings = pd.Series(antagonism_pca.components_[0] * a_sign, index=ANTAGONISM_INDICATORS)

    antagonism_resid_all = Z[ANTAGONISM_INDICATORS].copy()
    for col in ANTAGONISM_INDICATORS:
        antagonism_resid_all[col] = antagonism_resid_all[col] - g_scores_all * g_loadings[col]
    a_scores_all = antagonism_pca.transform(antagonism_resid_all)[:, 0] * a_sign

    score_df = all_df[["prolific_id", "dataset", "timepoint"]].copy()
    score_df["general_factor"] = g_scores_all
    score_df["distress_specific"] = d_scores_all
    score_df["antagonism_specific"] = a_scores_all

    meta = {
        "reconstruction_validation": recon_stats,
        "general_factor": {
            "loadings": g_loadings.sort_values(ascending=False).round(4).to_dict(),
            "explained_variance_ratio": float(pca_g.explained_variance_ratio_[0]),
        },
        "distress_specific": {
            "loadings": d_loadings.sort_values(ascending=False).round(4).to_dict(),
            "explained_variance_ratio": float(distress_pca.explained_variance_ratio_[0]),
        },
        "antagonism_specific": {
            "loadings": a_loadings.sort_values(ascending=False).round(4).to_dict(),
            "explained_variance_ratio": float(antagonism_pca.explained_variance_ratio_[0]),
        },
    }
    return score_df, meta


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
        o1 = df.loc[(df["dataset"] == "Original") & (df["timepoint"] == "t1"), metric].dropna()
        r1 = df.loc[(df["dataset"] == "Redeployed") & (df["timepoint"] == "t1"), metric].dropna()
        o2 = df.loc[(df["dataset"] == "Original") & (df["timepoint"] == "t2"), metric].dropna()
        r2 = df.loc[(df["dataset"] == "Redeployed") & (df["timepoint"] == "t2"), metric].dropna()
        tests = {}
        for name, a, b in [("t1", o1, r1), ("t2", o2, r2)]:
            if len(a) > 3 and len(b) > 3:
                ks = stats.ks_2samp(a, b)
                tests[name] = {
                    "n_original": int(len(a)),
                    "n_redeployed": int(len(b)),
                    "mean_original": float(a.mean()),
                    "mean_redeployed": float(b.mean()),
                    "cohens_d": float((a.mean() - b.mean()) / np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2)),
                    "ks_stat": float(ks.statistic),
                    "ks_p": float(ks.pvalue),
                }
        out[metric] = tests
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
        "Exploratory bifactor-style HiTOP decomposition across original and redeployed samples\nGeneral psychopathology plus orthogonal distress and antagonism residual factors",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965], h_pad=1.7)
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores, meta = build_bifactor_scores()
    summary = summarize_scores(scores)
    tests = distribution_tests(scores)

    scores.to_csv(OUT_SCORES, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    OUT_VALIDATION.write_text(json.dumps(meta, indent=2))
    OUT_TESTS.write_text(json.dumps(tests, indent=2))
    plot_scores(scores)

    print(OUT_FIG)
    print(OUT_SCORES)
    print(OUT_SUMMARY)
    print(OUT_VALIDATION)
    print(OUT_TESTS)


if __name__ == "__main__":
    main()
