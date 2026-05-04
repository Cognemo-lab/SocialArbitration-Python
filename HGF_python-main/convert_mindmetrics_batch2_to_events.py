from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def convert_payload(value: str) -> list[dict]:
    obj = json.loads(value)
    trials = obj.get("trials", [])
    out = []
    for row in trials:
        tt = row.get("trial_type")
        if tt == "choice":
            out.append(
                {
                    "PROLIFIC_PID": row.get("PROLIFIC_PID"),
                    "trial_type": "html-button-response",
                    "trial_index": row.get("trial_index"),
                    "time_elapsed": row.get("time_elapsed"),
                    "advice": row.get("advice"),
                    "outcome": row.get("outcome"),
                    "response": row.get("response"),
                    "practice": row.get("practice", False),
                }
            )
        elif tt == "wager":
            out.append(
                {
                    "PROLIFIC_PID": row.get("PROLIFIC_PID"),
                    "trial_type": "html-slider-response",
                    "trial_index": row.get("trial_index"),
                    "time_elapsed": row.get("time_elapsed"),
                    "response": row.get("response"),
                    "practice": row.get("practice", False),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MindMetrics Batch 2 task payloads to event-level CSV.")
    parser.add_argument("--in-file", required=True)
    parser.add_argument("--out-file", required=True)
    args = parser.parse_args()

    raw = pd.read_csv(args.in_file)
    all_rows: list[dict] = []
    for _, row in raw.iterrows():
        value = row.get("value")
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            all_rows.extend(convert_payload(value))
        except Exception:
            continue

    out = pd.DataFrame(all_rows)
    out = out.sort_values(["PROLIFIC_PID", "trial_index", "time_elapsed"]).reset_index(drop=True)
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_file, index=False)

    n_subj = out["PROLIFIC_PID"].nunique() if not out.empty else 0
    n_choice = int((out["trial_type"] == "html-button-response").sum()) if not out.empty else 0
    n_wager = int((out["trial_type"] == "html-slider-response").sum()) if not out.empty else 0
    print(f"saved={args.out_file}")
    print(f"n_rows={len(out)} n_subjects={n_subj} n_choice={n_choice} n_wager={n_wager}")


if __name__ == "__main__":
    main()
