from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
BATCH2_VALUES = (
    Path("/Users/drea/Documents/CAMH/Projects/MindMetrics_Prolific/MindMetrics- Batch2_T1")
    / "values_HITOP+++MindMetrics+Batch+2+-+Time+1_April+1,+2026_11.46.csv"
)
ORIG_RAW = BASE / "hitop" / "raw_data" / "HITOP+++MindMetrics+-+Time+1_September+2,+2024_20.40.csv"
ORIG_PROC = BASE / "hitop" / "processed_data" / "hitop_scales_T1.csv"
OUT_DIR = BASE / "hitop_batch2" / "processed_data"

SUSPICIOUSNESS_ITEMS = ["HiTOP_662", "Ext_58", "HiTOP_54", "Ext_279"]
DELUSIONS_ITEMS = ["HiTOP_531", "HiTOP_532", "HiTOP_533", "HiTOP_662"]
HALLUCINATIONS_ITEMS = ["HiTOP_606", "HiTOP_594", "HiTOP_608", "HiTOP_595", "HiTOP_601", "HiTOP_469"]


def numify(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*$", str(value))
    return float(match.group(1)) if match else np.nan


def load_raw_hitop(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    if "Q325" in df.columns:
        pid_col = "Q325"
    elif "PROLIFIC_PID" in df.columns and df["PROLIFIC_PID"].notna().sum() > 2:
        pid_col = "PROLIFIC_PID"
    else:
        pid_col = "ExternalReference"
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


def build_scales(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"prolific_id": df["prolific_id"].astype(str)})
    out["hitop_mistrust_suspiciousness"] = df[SUSPICIOUSNESS_ITEMS].sum(axis=1, min_count=len(SUSPICIOUSNESS_ITEMS))
    out["hitop_reality_distortion_delusions"] = df[DELUSIONS_ITEMS].sum(axis=1, min_count=len(DELUSIONS_ITEMS))
    out["hitop_reality_distortion_hallucinations"] = df[HALLUCINATIONS_ITEMS].sum(
        axis=1, min_count=len(HALLUCINATIONS_ITEMS)
    )
    out["hitop_reality_distortion"] = (
        out["hitop_reality_distortion_delusions"] + out["hitop_reality_distortion_hallucinations"]
    )
    return out


def corr_and_mae(a: pd.Series, b: pd.Series) -> dict[str, float]:
    pair = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(pair) < 3:
        return {"n": int(len(pair)), "pearson_r": np.nan, "mae": np.nan}
    return {
        "n": int(len(pair)),
        "pearson_r": float(pair["a"].corr(pair["b"])),
        "mae": float((pair["a"] - pair["b"]).abs().mean()),
    }


def validate_against_original() -> tuple[pd.DataFrame, dict]:
    raw = load_raw_hitop(ORIG_RAW)
    recon = build_scales(raw)
    proc = pd.read_csv(ORIG_PROC)
    merged = proc.merge(recon, on="prolific_id", suffixes=("_orig", "_recon"))
    targets = [
        "hitop_mistrust_suspiciousness",
        "hitop_reality_distortion_delusions",
        "hitop_reality_distortion_hallucinations",
        "hitop_reality_distortion",
    ]
    rows = []
    summary = {}
    for target in targets:
        stats = corr_and_mae(merged[f"{target}_orig"], merged[f"{target}_recon"])
        rows.append({"scale": target, **stats})
        summary[target] = stats
    return merged, {"validation": summary}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    batch2_raw = load_raw_hitop(BATCH2_VALUES)
    batch2_scales = build_scales(batch2_raw).dropna().copy()
    batch2_scales.to_csv(OUT_DIR / "hitop_scales_T1.csv", index=False)

    validation_df, summary = validate_against_original()
    validation_df.to_csv(OUT_DIR / "original_scale_reconstruction_validation.csv", index=False)
    summary["n_batch2_rows"] = int(len(batch2_scales))
    summary["suspiciousness_items"] = SUSPICIOUSNESS_ITEMS
    summary["delusions_items"] = DELUSIONS_ITEMS
    summary["hallucinations_items"] = HALLUCINATIONS_ITEMS
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
