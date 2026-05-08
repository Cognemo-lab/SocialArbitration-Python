from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")

ORIGINAL_TRIALS = BASE / "hgf_baselinefixed_full" / "extracted_model_trials.csv"
BATCH2_T1_RAW = BASE / "MindMetrics- Batch2_T1" / "Jungle_quest_T1_251.csv"
BATCH2_T2_RAW = BASE / "MindMetrics- Batch2_T2" / "Jungle_quest_T2_218.csv"

OUT_DIR = BASE / "behavior_test_retest_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MEASURE_LABELS = {
    "accuracy_rate": "Accuracy",
    "accuracy_first30": "Accuracy (first 30)",
    "accuracy_last20": "Accuracy (last 20)",
    "advice_taking_rate": "Advice-taking",
    "win_stay_rate": "Win-stay",
    "lose_switch_rate": "Lose-switch",
    "mean_wager": "Mean wager",
}
MEASURE_ORDER = list(MEASURE_LABELS.keys())


def _to_side(v):
    if pd.isna(v):
        return None
    if isinstance(v, str):
        vv = v.strip().lower()
        if vv == "l":
            return 0
        if vv == "r":
            return 1
        return None
    try:
        iv = int(float(v))
        if iv in (0, 1):
            return iv
    except Exception:
        pass
    return None


def _is_practice_true(v):
    if pd.isna(v):
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes"}
    try:
        return int(v) == 1
    except Exception:
        return False


def convert_payload_rows(raw_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv)
    rows: List[dict] = []
    for _, row in raw.iterrows():
        value = row.get("value")
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            obj = json.loads(value)
        except Exception:
            continue
        for trial in obj.get("trials", []):
            tt = trial.get("trial_type")
            if tt == "choice":
                rows.append(
                    {
                        "PROLIFIC_PID": trial.get("PROLIFIC_PID"),
                        "trial_type": "html-button-response",
                        "trial_index": trial.get("trial_index"),
                        "time_elapsed": trial.get("time_elapsed"),
                        "advice": trial.get("advice"),
                        "outcome": trial.get("outcome"),
                        "response": trial.get("response"),
                        "practice": trial.get("practice", False),
                    }
                )
            elif tt == "wager":
                rows.append(
                    {
                        "PROLIFIC_PID": trial.get("PROLIFIC_PID"),
                        "trial_type": "html-slider-response",
                        "trial_index": trial.get("trial_index"),
                        "time_elapsed": trial.get("time_elapsed"),
                        "response": trial.get("response"),
                        "practice": trial.get("practice", False),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["PROLIFIC_PID", "trial_index", "time_elapsed"]).reset_index(drop=True)


def extract_trials_from_raw(raw_df: pd.DataFrame, round_name: str) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df = raw_df.sort_values(["PROLIFIC_PID", "trial_index", "time_elapsed"]).reset_index(drop=True)
    out = []
    for pid, g in raw_df.groupby("PROLIFIC_PID", sort=False):
        pending = None
        for _, row in g.iterrows():
            tt = row.get("trial_type", None)
            practice = _is_practice_true(row.get("practice", np.nan))
            advice_side = _to_side(row.get("advice", np.nan))
            outcome_side = _to_side(row.get("outcome", np.nan))

            if tt == "html-button-response" and advice_side is not None and outcome_side is not None and not practice:
                response_side = _to_side(row.get("response", np.nan))
                if response_side is None:
                    pending = None
                    continue
                advice_correctness = 1.0 if advice_side == outcome_side else 0.0
                advice_taken = 1.0 if response_side == advice_side else 0.0
                pending = {
                    "prolific_id": str(pid),
                    "round_name": round_name,
                    "input_advice": advice_correctness,
                    "input_reward": float(outcome_side),
                    "advice_card_space": float(advice_side),
                    "choice_advice_taken": advice_taken,
                    "choice_side": float(response_side),
                    "trial_index_choice": row.get("trial_index", np.nan),
                }
                continue

            if pending is not None and tt == "html-slider-response":
                try:
                    wager = float(row.get("response", np.nan))
                except Exception:
                    wager = np.nan
                if np.isfinite(wager):
                    rec = dict(pending)
                    rec["wager"] = wager
                    rec["trial_index_wager"] = row.get("trial_index", np.nan)
                    out.append(rec)
                pending = None

    out_df = pd.DataFrame(out)
    if out_df.empty:
        return out_df
    return out_df.sort_values(["prolific_id", "trial_index_choice"]).reset_index(drop=True)


def behavior_summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, group in trials.groupby("prolific_id", sort=False):
        g = group.sort_values("trial_index_choice").reset_index(drop=True).copy()
        correct = (g["choice_side"] == g["input_reward"]).astype(float)
        prev_correct = correct.shift(1)
        prev_choice = g["choice_side"].shift(1)
        stay = (g["choice_side"] == prev_choice).astype(float)
        switch = 1.0 - stay
        win_mask = prev_correct == 1
        lose_mask = prev_correct == 0
        first30 = correct.iloc[:30]
        last20 = correct.iloc[-20:]
        rows.append(
            {
                "prolific_id": pid,
                "accuracy_rate": float(correct.mean()),
                "accuracy_first30": float(first30.mean()) if len(first30) else np.nan,
                "accuracy_last20": float(last20.mean()) if len(last20) else np.nan,
                "advice_taking_rate": float(g["choice_advice_taken"].mean()),
                "win_stay_rate": float(stay.loc[win_mask].mean()) if win_mask.any() else np.nan,
                "lose_switch_rate": float(switch.loc[lose_mask].mean()) if lose_mask.any() else np.nan,
                "mean_wager": float(g["wager"].mean()),
            }
        )
    return pd.DataFrame(rows)


def icc3_1(two_col: np.ndarray) -> float:
    x = np.asarray(two_col, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    n, k = x.shape
    if n < 3 or k != 2:
        return np.nan
    grand = np.mean(x)
    mean_row = np.mean(x, axis=1, keepdims=True)
    mean_col = np.mean(x, axis=0, keepdims=True)
    ss_row = k * np.sum((mean_row - grand) ** 2)
    ss_col = n * np.sum((mean_col - grand) ** 2)
    ss_tot = np.sum((x - grand) ** 2)
    ss_err = ss_tot - ss_row - ss_col
    ms_row = ss_row / (n - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_row + (k - 1) * ms_err
    if denom == 0:
        return np.nan
    return float((ms_row - ms_err) / denom)


def compute_reliability(wide_df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    rows = []
    for measure in MEASURE_ORDER:
        t1_col = f"{measure}_t1"
        t2_col = f"{measure}_t2"
        sub = wide_df[[t1_col, t2_col]].dropna().copy()
        x = sub[t1_col].to_numpy(dtype=float)
        y = sub[t2_col].to_numpy(dtype=float)
        pearson = np.nan
        if len(sub) >= 3 and np.std(x) > 0 and np.std(y) > 0:
            pearson = float(np.corrcoef(x, y)[0, 1])
        rows.append(
            {
                "dataset": dataset_label,
                "measure": measure,
                "label": MEASURE_LABELS[measure],
                "n_pairs": int(len(sub)),
                "pearson_r": pearson,
                "icc3_1": icc3_1(sub.to_numpy(dtype=float)),
                "mean_abs_delta": float(np.mean(np.abs(y - x))) if len(sub) else np.nan,
                "t1_mean": float(np.mean(x)) if len(sub) else np.nan,
                "t2_mean": float(np.mean(y)) if len(sub) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_reliability(comparison: pd.DataFrame, out_path: Path) -> None:
    colors = {"Original": "#4C78A8", "Redeployment": "#F58518"}
    y = np.arange(len(MEASURE_ORDER))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.5), sharey=True)
    metrics = [("pearson_r", "Behavioral Pearson r"), ("icc3_1", "Behavioral ICC(3,1)")]

    for ax, (metric, title) in zip(axes, metrics):
        ax.axvline(0.4, color="#999999", linestyle="--", linewidth=1)
        ax.axvline(0.0, color="#DDDDDD", linestyle="-", linewidth=0.8)
        for i, measure in enumerate(MEASURE_ORDER):
            yy = y[i]
            sub = comparison.loc[comparison["measure"] == measure].copy()
            orig = sub.loc[sub["dataset"] == "Original", metric].iloc[0]
            redep = sub.loc[sub["dataset"] == "Redeployment", metric].iloc[0]
            ax.plot([orig, redep], [yy, yy], color="#BBBBBB", linewidth=1.2, zorder=1)
            ax.scatter(orig, yy, s=70, color=colors["Original"], edgecolor="white", linewidth=0.8, zorder=3)
            ax.scatter(redep, yy, s=70, color=colors["Redeployment"], edgecolor="white", linewidth=0.8, zorder=3)

        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlim(-0.1, 1.0)
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel("Reliability", fontsize=12)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([MEASURE_LABELS[m] for m in MEASURE_ORDER], fontsize=11)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["Original"], markeredgecolor="white", markersize=9, label="Original"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["Redeployment"], markeredgecolor="white", markersize=9, label="Redeployment"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Behavioral Test-Retest Reliability: Original vs Redeployment", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), w_pad=1.0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    original_trials = pd.read_csv(ORIGINAL_TRIALS)
    original_trials = original_trials.loc[original_trials["round_name"].isin(["round1", "round2"])].copy()

    original_t1 = behavior_summary(original_trials.loc[original_trials["round_name"] == "round1"].copy())
    original_t2 = behavior_summary(original_trials.loc[original_trials["round_name"] == "round2"].copy())
    original_wide = original_t1.merge(original_t2, on="prolific_id", suffixes=("_t1", "_t2"))

    batch2_t1_events = convert_payload_rows(BATCH2_T1_RAW)
    batch2_t2_events = convert_payload_rows(BATCH2_T2_RAW)
    batch2_t1_trials = extract_trials_from_raw(batch2_t1_events, "round1")
    batch2_t2_trials = extract_trials_from_raw(batch2_t2_events, "round2")
    batch2_t1 = behavior_summary(batch2_t1_trials)
    batch2_t2 = behavior_summary(batch2_t2_trials)
    batch2_wide = batch2_t1.merge(batch2_t2, on="prolific_id", suffixes=("_t1", "_t2"))

    original_reliability = compute_reliability(original_wide, "Original")
    batch2_reliability = compute_reliability(batch2_wide, "Redeployment")
    comparison = pd.concat([original_reliability, batch2_reliability], ignore_index=True)

    comparison.to_csv(OUT_DIR / "behavior_test_retest_reliability_comparison.csv", index=False)
    original_wide.to_csv(OUT_DIR / "original_behavior_t1_t2_wide.csv", index=False)
    batch2_wide.to_csv(OUT_DIR / "redeployment_behavior_t1_t2_wide.csv", index=False)
    batch2_t1_trials.to_csv(OUT_DIR / "redeployment_t1_extracted_trials.csv", index=False)
    batch2_t2_trials.to_csv(OUT_DIR / "redeployment_t2_extracted_trials.csv", index=False)

    plot_reliability(comparison, OUT_DIR / "figure_behavior_test_retest_reliability_comparison.png")
    print("saved", OUT_DIR / "behavior_test_retest_reliability_comparison.csv")
    print("saved", OUT_DIR / "figure_behavior_test_retest_reliability_comparison.png")


if __name__ == "__main__":
    main()
