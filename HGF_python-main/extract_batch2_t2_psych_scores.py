from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
T1_VALUES = BASE / "MindMetrics- Batch2_T1" / "values_HITOP+++MindMetrics+Batch+2+-+Time+1_April+1,+2026_11.46.csv"
T2_VALUES = BASE / "MindMetrics- Batch2_T2" / "HITOP+++MindMetrics+Batch+2+-+Time+2_May+4,+2026_15.15_values.csv"
OUT_DIR = BASE / "hitop_batch2" / "processed_data"

SUSPICIOUSNESS_ITEMS = ["HiTOP_662", "Ext_58", "HiTOP_54", "Ext_279"]
DELUSIONS_ITEMS = ["HiTOP_531", "HiTOP_532", "HiTOP_533", "HiTOP_662"]
HALLUCINATIONS_ITEMS = ["HiTOP_606", "HiTOP_594", "HiTOP_608", "HiTOP_595", "HiTOP_601", "HiTOP_469"]
SUICIDALITY_PROXY_ITEMS = ["Ext_468", "HiTOP_361", "Ext_266"]


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
    df = pd.read_csv(path, dtype=str, skiprows=[1, 2])
    pid_col = "Q325" if "Q325" in df.columns else "ExternalReference"
    df = df[df[pid_col].notna()].copy()
    if "Finished" in df.columns:
        finished = df["Finished"].astype(str).str.strip().str.lower()
        df = df[finished.isin(["true", "1"])].copy()
    item_cols = [
        c
        for c in df.columns
        if c.startswith("HiTOP_") or c.startswith("Ext_") or c.startswith("inq_") or c.startswith("gcsq_")
    ]
    for col in item_cols:
        df[col] = df[col].map(numify)
    df = df.rename(columns={pid_col: "prolific_id", "SC29": "inq_grand_sum", "SC31": "inq_thwarted_belongigness", "SC32": "inq_perceived_burdensomness", "SC37": "gcsq_perceived_capability", "SC38": "gcsq_pain_tolerance", "SC39": "gcsq_fearlesness_of_death"})
    df["prolific_id"] = df["prolific_id"].astype(str).str.strip()
    return df.reset_index(drop=True)


def build_scales(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"prolific_id": df["prolific_id"].astype(str)})
    out["hitop_mistrust_suspiciousness"] = df[SUSPICIOUSNESS_ITEMS].sum(axis=1, min_count=len(SUSPICIOUSNESS_ITEMS))
    out["hitop_reality_distortion_delusions"] = df[DELUSIONS_ITEMS].sum(axis=1, min_count=len(DELUSIONS_ITEMS))
    out["hitop_reality_distortion_hallucinations"] = df[HALLUCINATIONS_ITEMS].sum(axis=1, min_count=len(HALLUCINATIONS_ITEMS))
    out["hitop_reality_distortion"] = (
        out["hitop_reality_distortion_delusions"] + out["hitop_reality_distortion_hallucinations"]
    )
    out["hitop_suicidality_proxy"] = df[SUICIDALITY_PROXY_ITEMS].sum(axis=1, min_count=len(SUICIDALITY_PROXY_ITEMS))
    # Original processed HiTOP suicidality is on a 4-13 scale; with these three
    # raw 1-4 items, the aligned redeployment score is the item sum plus one.
    out["hitop_suicidality"] = out["hitop_suicidality_proxy"] + 1.0
    for col in [
        "inq_grand_sum",
        "inq_thwarted_belongigness",
        "inq_perceived_burdensomness",
        "gcsq_perceived_capability",
        "gcsq_pain_tolerance",
        "gcsq_fearlesness_of_death",
    ]:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    return out.dropna(subset=["prolific_id"]).copy()


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    pair = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(pair) < 3:
        return np.nan
    if pair["a"].std(ddof=0) == 0 or pair["b"].std(ddof=0) == 0:
        return np.nan
    return float(pair["a"].corr(pair["b"]))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t1 = build_scales(load_raw_hitop(T1_VALUES))
    t2 = build_scales(load_raw_hitop(T2_VALUES))

    t1.to_csv(OUT_DIR / "psych_scales_T1_extended.csv", index=False)
    t2.to_csv(OUT_DIR / "psych_scales_T2_extended.csv", index=False)

    merged = t1.merge(t2, on="prolific_id", suffixes=("_t1", "_t2"))
    merged.to_csv(OUT_DIR / "psych_scales_t1_t2_wide.csv", index=False)

    metrics = [
        "hitop_mistrust_suspiciousness",
        "hitop_reality_distortion",
        "hitop_reality_distortion_delusions",
        "hitop_reality_distortion_hallucinations",
        "hitop_suicidality",
        "hitop_suicidality_proxy",
        "inq_grand_sum",
        "inq_thwarted_belongigness",
        "inq_perceived_burdensomness",
        "gcsq_perceived_capability",
        "gcsq_pain_tolerance",
        "gcsq_fearlesness_of_death",
    ]
    rows = []
    for metric in metrics:
        rows.append(
            {
                "metric": metric,
                "n_pairs": int(merged[[f"{metric}_t1", f"{metric}_t2"]].dropna().shape[0]),
                "pearson_r_t1_t2": _safe_corr(merged[f"{metric}_t1"], merged[f"{metric}_t2"]),
                "mean_t1": float(pd.to_numeric(merged[f"{metric}_t1"], errors="coerce").mean()),
                "mean_t2": float(pd.to_numeric(merged[f"{metric}_t2"], errors="coerce").mean()),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "psych_test_retest_summary.csv", index=False)
    print(f"Saved: {OUT_DIR / 'psych_scales_T2_extended.csv'}")
    print(f"Saved: {OUT_DIR / 'psych_scales_t1_t2_wide.csv'}")
    print(f"Saved: {OUT_DIR / 'psych_test_retest_summary.csv'}")


if __name__ == "__main__":
    main()
