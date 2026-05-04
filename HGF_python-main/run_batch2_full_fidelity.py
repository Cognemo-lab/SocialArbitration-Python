from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
PROJECT = BASE / "HGF_python-main"
RAW_FILE = BASE / "raw_data_batch2" / "jungle-task.round1.batch2_t1_events.csv"
OUT_DIR = BASE / "hgf_batch2_t1_baselinefixed_fullfidelity"

PRC_CONFIG = (
    "HGF.code_model_wagad.prc_model.hgf_binary3l_freekappa_reward_social_baselinefixed_config."
    "hgf_binary3l_freekappa_reward_social_baselinefixed_config"
)
OBS_CONFIG = (
    "HGF.code_model_wagad.obs_model.linear_volatilitydecnoise_1stlevelprecision_reward_social_baselinefixed_config."
    "linear_volatilitydecnoise_1stlevelprecision_reward_social_baselinefixed_config"
)


def run_step(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT}:{PROJECT / 'python'}"
    print("\n[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(BASE), env=env, check=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_step(
        [
            sys.executable,
            str(PROJECT / "fit_recover_reliability_raw.py"),
            "--raw-files",
            str(RAW_FILE),
            "--out-dir",
            str(OUT_DIR),
            "--max-iter",
            "5",
            "--n-jobs",
            "8",
            "--prc-config",
            PRC_CONFIG,
            "--obs-config",
            OBS_CONFIG,
        ]
    )

    run_step(
        [
            sys.executable,
            str(PROJECT / "posterior_predictive_checks.py"),
            "--fit-dir",
            str(OUT_DIR / "fit_jsons"),
            "--trials-csv",
            str(OUT_DIR / "extracted_model_trials.csv"),
            "--out-dir",
            str(OUT_DIR / "posterior_predictive_checks"),
        ]
    )

    run_step([sys.executable, str(PROJECT / "compare_original_vs_batch2_modeling.py")])

    print("\n[DONE] Batch 2 full-fidelity run completed.", flush=True)


if __name__ == "__main__":
    main()
