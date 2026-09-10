from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
PARAM_LONG = BASE / "hgf_baselinefixed_full" / "parameter_estimates_long_t1_t2.csv"
STATE_WIDE = BASE / "hgf_baselinefixed_full" / "state_reliability" / "state_estimates_wide_t1_t2.csv"
HITOP_DIR = BASE / "hitop" / "processed_data"
OUT_DIR = BASE / "hitop" / "consistent_domain_analysis"

PREDICTORS = [
    "inferv_a_mean",
    "om_a",
    "abs_eps2_a_mean",
    "eps3_a_mean",
    "ka_a",
]

LABELS = {
    "inferv_a_mean": "Inferential variance",
    "om_a": "Omega-advice",
    "abs_eps2_a_mean": "|epsilon2|",
    "eps3_a_mean": "epsilon3",
    "ka_a": "Kappa-advice",
}


def load_predictors_long() -> pd.DataFrame:
    params = pd.read_csv(PARAM_LONG)
    params = params[["prolific_id", "round_name", "timepoint", "parameter", "estimate"]].copy()
    params["quantity_kind"] = "parameter"
    params = params.rename(columns={"parameter": "quantity_name"})

    states = pd.read_csv(STATE_WIDE)
    states = states.melt(
        id_vars=["prolific_id", "state_metric"],
        value_vars=["t1", "t2"],
        var_name="timepoint",
        value_name="estimate",
    )
    states["round_name"] = states["timepoint"].map({"t1": "round1", "t2": "round2"})
    states["quantity_kind"] = "state"
    states = states.rename(columns={"state_metric": "quantity_name"})
    states = states[["prolific_id", "round_name", "timepoint", "quantity_name", "estimate", "quantity_kind"]]

    return pd.concat([params, states], ignore_index=True)


def load_hitop_long() -> pd.DataFrame:
    frames = []
    for timepoint, hitop_file in [("t1", "hitop_scales_T1.csv"), ("t2", "hitop_scales_T2.csv")]:
        hitop = pd.read_csv(HITOP_DIR / hitop_file)[["prolific_id", "hitop_mistrust_suspiciousness"]].copy()
        hitop["timepoint"] = timepoint
        frames.append(hitop)
    return pd.concat(frames, ignore_index=True)


def zscore_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        sd = out[col].std(ddof=0)
        out[f"{col}_z"] = 0.0 if sd == 0 or pd.isna(sd) else (out[col] - out[col].mean()) / sd
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_long = load_predictors_long()
    hitop_long = load_hitop_long()

    wide = (
        pred_long[pred_long["quantity_name"].isin(PREDICTORS)]
        .pivot_table(
            index=["prolific_id", "round_name", "timepoint"],
            columns="quantity_name",
            values="estimate",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None

    merged = wide.merge(hitop_long, on=["prolific_id", "timepoint"], how="inner").dropna().copy()
    merged = zscore_frame(merged, PREDICTORS)
    zcols = [f"{p}_z" for p in PREDICTORS]

    x = merged[zcols].to_numpy()
    pca = PCA(n_components=1)
    pca.fit(x)
    explained = float(pca.explained_variance_ratio_[0])
    loadings = pd.Series(pca.components_[0], index=PREDICTORS)

    cov = merged[zcols].cov()
    cov.index = [LABELS[p[:-2]] for p in cov.index]
    cov.columns = [LABELS[p[:-2]] for p in cov.columns]
    cov.to_csv(OUT_DIR / "suspiciousness_composite_standardized_covariance_matrix.csv")

    fig = plt.figure(figsize=(13.8, 8.8))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1.18], wspace=0.42, hspace=0.42)

    hist_axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(len(PREDICTORS))]
    for ax, predictor in zip(hist_axes, PREDICTORS):
        vals = merged[predictor].to_numpy()
        ax.hist(vals, bins=28, color="#3f6aa8", alpha=0.88, edgecolor="white")
        ax.axvline(np.mean(vals), color="#1f1f1f", lw=1.2)
        title = LABELS[predictor]
        loading = loadings[predictor]
        ax.set_title(f"{title}\nPC1 loading = {loading:.3f}", fontsize=11, pad=8)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.18)

    axh = fig.add_subplot(gs[:, 2])
    im = axh.imshow(cov.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    axh.set_xticks(np.arange(len(cov.columns)))
    axh.set_xticklabels(cov.columns, rotation=45, ha="right", fontsize=10)
    axh.set_yticks(np.arange(len(cov.index)))
    axh.set_yticklabels(cov.index, fontsize=10)
    axh.set_title(
        "Standardized covariance matrix\n(z-scored predictors)",
        fontsize=12,
        pad=10,
    )
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            val = cov.iloc[i, j]
            axh.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(val) > 0.5 else "#222222",
            )
    cbar = fig.colorbar(im, ax=axh, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Pooled suspiciousness computational composite: predictor distributions and covariance",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.06,
        0.93,
        f"Pooled merged sample n = {len(merged)}; pooled suspiciousness PC1 variance explained = {explained:.3f}",
        fontsize=10,
        color="#444444",
    )
    fig.text(
        0.06,
        0.02,
        "Covariance matrix is shown after z-scoring predictors, so values are directly comparable across variables.",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    out_png = OUT_DIR / "figure_suspiciousness_composite_distributions_covariance.png"
    out_svg = OUT_DIR / "figure_suspiciousness_composite_distributions_covariance.svg"
    fig.savefig(out_png, dpi=260)
    fig.savefig(out_svg)
    plt.close(fig)

    merged[
        ["prolific_id", "round_name", "timepoint", "hitop_mistrust_suspiciousness"] + PREDICTORS + zcols
    ].to_csv(OUT_DIR / "suspiciousness_composite_pooled_data.csv", index=False)

    print(out_png)
    print(out_svg)


if __name__ == "__main__":
    main()
