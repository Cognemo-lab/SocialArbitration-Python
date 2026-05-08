from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
OUT_DIR = BASE / "behavior_fod_moderation_comparison" / "state_and_prevalence_assessment"
OUT_FIG = OUT_DIR / "figure_hitop_additional_domain_distributions_original_vs_redeployment_t1_t2.png"
OUT_CSV = OUT_DIR / "hitop_additional_domain_distributions_original_vs_redeployment_t1_t2_summary.csv"
OUT_VALIDATION = OUT_DIR / "hitop_additional_domain_reconstruction_validation.json"

ORIG_RAW_T1 = BASE / "hitop" / "raw_data" / "HITOP+++MindMetrics+-+Time+1_September+2,+2024_20.40.csv"
ORIG_PROC_T1 = BASE / "hitop" / "processed_data" / "hitop_scales_T1.csv"
ORIG_PROC_T2 = BASE / "hitop" / "processed_data" / "hitop_scales_T2.csv"
RED_RAW_T1 = BASE / "MindMetrics- Batch2_T1" / "values_HITOP+++MindMetrics+Batch+2+-+Time+1_April+1,+2026_11.46.csv"
RED_RAW_T2 = BASE / "MindMetrics- Batch2_T2" / "HITOP+++MindMetrics+Batch+2+-+Time+2_May+4,+2026_15.15_values.csv"

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

METRICS = [
    ("hitop_distress_dysphoria", "Internalizing - Distress"),
    ("hitop_antagonistic_externalizing", "Antagonistic - Externalizing"),
]

GROUPS = [
    ("Original", "t1", "Original T1", "#4C78A8"),
    ("Original", "t2", "Original T2", "#8FB9E0"),
    ("Redeployed", "t1", "Redeployed T1", "#E76F51"),
    ("Redeployed", "t2", "Redeployed T2", "#F2A07E"),
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


def build_long_df() -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    orig_raw = load_raw_hitop(ORIG_RAW_T1)
    red_raw_t1 = load_raw_hitop(RED_RAW_T1)
    red_raw_t2 = load_raw_hitop(RED_RAW_T2)
    proc_t1 = pd.read_csv(ORIG_PROC_T1)
    proc_t2 = pd.read_csv(ORIG_PROC_T2)

    common_items = sorted(
        set(c for c in orig_raw.columns if c.startswith(("HiTOP_", "Ext_")))
        & set(c for c in red_raw_t1.columns if c.startswith(("HiTOP_", "Ext_")))
        & set(c for c in red_raw_t2.columns if c.startswith(("HiTOP_", "Ext_")))
    )

    orig_train = proc_t1.merge(orig_raw[["prolific_id", *common_items]], on="prolific_id", how="inner")
    orig_train["hitop_antagonistic_externalizing"] = build_antagonism_direct(orig_train)

    facet_means = proc_t1[ANTAGONISM_FACETS].mean()
    facet_sds = proc_t1[ANTAGONISM_FACETS].std(ddof=0).replace(0, np.nan)
    orig_t2 = proc_t2.copy()
    orig_t2["hitop_antagonistic_externalizing"] = build_antagonism_direct(orig_t2, means=facet_means, sds=facet_sds)

    X_train = orig_train[common_items]
    dist_model, dist_stats = fit_reconstruction_model(X_train, orig_train["hitop_distress_dysphoria"])
    ant_model, ant_stats = fit_reconstruction_model(X_train, orig_train["hitop_antagonistic_externalizing"])

    red_t1 = pd.DataFrame({"prolific_id": red_raw_t1["prolific_id"].astype(str).str.strip()})
    red_t1["hitop_distress_dysphoria"] = dist_model.predict(red_raw_t1[common_items])
    red_t1["hitop_antagonistic_externalizing"] = ant_model.predict(red_raw_t1[common_items])
    red_t1["dataset"] = "Redeployed"
    red_t1["timepoint"] = "t1"

    red_t2 = pd.DataFrame({"prolific_id": red_raw_t2["prolific_id"].astype(str).str.strip()})
    red_t2["hitop_distress_dysphoria"] = dist_model.predict(red_raw_t2[common_items])
    red_t2["hitop_antagonistic_externalizing"] = ant_model.predict(red_raw_t2[common_items])
    red_t2["dataset"] = "Redeployed"
    red_t2["timepoint"] = "t2"

    orig_t1 = orig_train[["prolific_id", "hitop_distress_dysphoria", "hitop_antagonistic_externalizing"]].copy()
    orig_t1["dataset"] = "Original"
    orig_t1["timepoint"] = "t1"

    orig_t2 = orig_t2[["prolific_id", "hitop_distress_dysphoria", "hitop_antagonistic_externalizing"]].copy()
    orig_t2["dataset"] = "Original"
    orig_t2["timepoint"] = "t2"

    long_df = pd.concat([orig_t1, orig_t2, red_t1, red_t2], ignore_index=True)
    validation = {"hitop_distress_dysphoria": dist_stats, "hitop_antagonistic_externalizing": ant_stats}
    return long_df, validation


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, label in METRICS:
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
                    "prop_floor": float((s == s.min()).mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, validation = build_long_df()
    summary = summarize(df)
    summary.to_csv(OUT_CSV, index=False)
    OUT_VALIDATION.write_text(json.dumps(validation, indent=2))

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(12, 8.5))
    if len(METRICS) == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    for ax, (metric, label) in zip(axes, METRICS):
        values = []
        colors = []
        ns = []
        positions = np.arange(1, len(GROUPS) + 1)
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

        ax.set_title(label, fontsize=13, pad=8)
        ax.set_xticks(positions)
        ax.set_xticklabels([g[2] for g in GROUPS], fontsize=10)
        ax.grid(axis="y", alpha=0.18)
        ax.set_ylabel("Score")

    fig.suptitle(
        "HiTOP domain distributions across original and redeployed samples\nRedeployed domains are calibrated reconstructions from raw HiTOP items",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=1.6)
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    print(OUT_FIG)
    print(OUT_CSV)
    print(OUT_VALIDATION)


if __name__ == "__main__":
    main()
