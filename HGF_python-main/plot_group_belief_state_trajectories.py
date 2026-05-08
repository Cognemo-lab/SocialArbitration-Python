from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR / "python") not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR / "python"))

from HGF.code_model_wagad.prc_model.hgf_binary3l_freekappa_reward_social import (
    hgf_binary3l_freekappa_reward_social,
)


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
OUT_DIR = BASE / "belief_state_group_trajectories"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRC_ORDER = [
    "mu2r_0",
    "sa2r_0",
    "mu3r_0",
    "sa3r_0",
    "ka_r",
    "om_r",
    "th_r",
    "mu2a_0",
    "sa2a_0",
    "mu3a_0",
    "sa3a_0",
    "ka_a",
    "om_a",
    "th_a",
    "phi_r",
    "m_r",
    "phi_a",
    "m_a",
]
OBS_ORDER = ["be0", "be1", "be2", "be3", "be4", "be5", "be6", "ze", "be_ch", "be_wager"]

GROUPS = [
    {
        "name": "Original T1",
        "dataset": "Original",
        "timepoint": "t1",
        "params": BASE / "hgf_baselinefixed_full" / "parameter_estimates_long_t1_t2.csv",
        "trials": BASE / "hgf_baselinefixed_full" / "extracted_model_trials.csv",
        "round_name": "round1",
        "color": "#4C78A8",
    },
    {
        "name": "Original T2",
        "dataset": "Original",
        "timepoint": "t2",
        "params": BASE / "hgf_baselinefixed_full" / "parameter_estimates_long_t1_t2.csv",
        "trials": BASE / "hgf_baselinefixed_full" / "extracted_model_trials.csv",
        "round_name": "round2",
        "color": "#8FB9E0",
    },
    {
        "name": "Redeployed T1",
        "dataset": "Redeployed",
        "timepoint": "t1",
        "params": BASE / "hgf_batch2_t1_baselinefixed_fullfidelity" / "parameter_estimates_long_t1_t2.csv",
        "trials": BASE / "hgf_batch2_t1_baselinefixed_fullfidelity" / "extracted_model_trials.csv",
        "round_name": "round1",
        "color": "#E76F51",
    },
    {
        "name": "Redeployed T2",
        "dataset": "Redeployed",
        "timepoint": "t2",
        "params": BASE / "hgf_batch2_t2_parameter_reliability" / "parameter_estimates_long_t1_t2.csv",
        "trials": BASE / "hgf_batch2_t2_parameter_reliability" / "extracted_model_trials_t2.csv",
        "round_name": "round2",
        "color": "#F2A07E",
    },
]
MAX_TRIAL = 100

ADVICE_METRICS = {
    "mu1hat": ("muhat_a", 0, False),
    "mu2hat": ("muhat_a", 1, False),
    "mu3hat": ("muhat_a", 2, False),
    "sigma1hat": ("sahat_a", 0, False),
    "sigma2hat": ("sahat_a", 1, False),
    "sigma3hat": ("sahat_a", 2, False),
    "abs_epsilon2": ("da_a", 1, True),
    "epsilon3": ("eps3_a", None, False),
}

REWARD_METRICS = {
    "mu1hat": ("muhat_r", 0, False),
    "mu2hat": ("muhat_r", 1, False),
    "mu3hat": ("muhat_r", 2, False),
    "sigma1hat": ("sahat_r", 0, False),
    "sigma2hat": ("sahat_r", 1, False),
    "sigma3hat": ("sahat_r", 2, False),
    "abs_epsilon2": ("da_r", 1, True),
    "epsilon3": ("eps3_r", None, False),
}

PANEL_LABELS = {
    "mu1hat": r"$\hat{\mu}_1$",
    "mu2hat": r"$\hat{\mu}_2$",
    "mu3hat": r"$\hat{\mu}_3$",
    "sigma1hat": r"$\hat{\sigma}_1$",
    "sigma2hat": r"$\hat{\sigma}_2$",
    "sigma3hat": r"$\hat{\sigma}_3$",
    "abs_epsilon2": r"$|\epsilon_2|$",
    "epsilon3": r"$\epsilon_3$",
}


def load_param_pivot(path: Path) -> pd.DataFrame:
    params = pd.read_csv(path)
    pivot = (
        params.pivot_table(
            index=["prolific_id", "round_name", "timepoint"],
            columns=["group", "parameter"],
            values="estimate",
            aggfunc="first",
        )
        .reset_index()
    )
    if isinstance(pivot.columns, pd.MultiIndex):
        pivot.columns = [
            left if right == "" else (left, right) for left, right in pivot.columns.to_flat_index()
        ]
    return pivot


def compute_trialwise_states(group_cfg: dict) -> Dict[str, np.ndarray]:
    params = load_param_pivot(group_cfg["params"])
    trials = pd.read_csv(group_cfg["trials"])

    rows = params[
        (params["timepoint"].astype(str).str.lower() == group_cfg["timepoint"])
        & (params["round_name"].astype(str) == group_cfg["round_name"])
    ].copy()

    metrics_store = {**{f"advice_{k}": [] for k in ADVICE_METRICS}, **{f"reward_{k}": [] for k in REWARD_METRICS}}
    lengths = []

    for _, row in rows.iterrows():
        pid = str(row["prolific_id"])
        g = trials[
            (trials["prolific_id"].astype(str) == pid)
            & (trials["round_name"].astype(str) == group_cfg["round_name"])
        ].copy()
        if g.empty:
            continue
        g = g.sort_values("trial_index_choice").reset_index(drop=True)
        u = g[["input_advice", "input_reward", "advice_card_space"]].to_numpy(dtype=float)
        if len(u) < 5:
            continue

        p_prc = {name: float(row[("prc", name)]) for name in PRC_ORDER}
        prc_vec = np.array([p_prc[name] for name in PRC_ORDER], dtype=float)
        traj, _ = hgf_binary3l_freekappa_reward_social({"u": u, "ign": []}, prc_vec)

        eps3_r = 0.5 * p_prc["ka_r"] * traj["w_r"] * traj["da_r"][:, 1] / traj["sa_r"][:, 2]
        eps3_a = 0.5 * p_prc["ka_a"] * traj["w_a"] * traj["da_a"][:, 1] / traj["sa_a"][:, 2]
        derived = {"eps3_r": eps3_r, "eps3_a": eps3_a}

        for key, (src, idx, take_abs) in ADVICE_METRICS.items():
            arr = derived[src] if src in derived else traj[src][:, idx]
            arr = np.abs(arr) if take_abs else arr
            metrics_store[f"advice_{key}"].append(np.asarray(arr, dtype=float))
        for key, (src, idx, take_abs) in REWARD_METRICS.items():
            arr = derived[src] if src in derived else traj[src][:, idx]
            arr = np.abs(arr) if take_abs else arr
            metrics_store[f"reward_{key}"].append(np.asarray(arr, dtype=float))
        lengths.append(len(u))

    out = {}
    for metric, series_list in metrics_store.items():
        max_len = max(len(x) for x in series_list)
        arr = np.full((len(series_list), max_len), np.nan, dtype=float)
        for i, seq in enumerate(series_list):
            arr[i, : len(seq)] = seq
        out[metric] = arr
    out["n_sessions"] = np.array([len(lengths)])
    out["trial_count_summary"] = np.array(lengths)
    return out


def summarise_group(group_cfg: dict, states: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for branch, metrics in [("advice", ADVICE_METRICS), ("reward", REWARD_METRICS)]:
        for metric in metrics:
            arr = states[f"{branch}_{metric}"]
            mean = np.nanmean(arr, axis=0)
            n = np.sum(np.isfinite(arr), axis=0)
            sd = np.nanstd(arr, axis=0, ddof=1)
            sem = sd / np.sqrt(np.maximum(n, 1))
            for t_idx, (m, s, nn) in enumerate(zip(mean, sem, n), start=1):
                if t_idx > MAX_TRIAL:
                    break
                if nn == 0 or not np.isfinite(m):
                    continue
                rows.append(
                    {
                        "group": group_cfg["name"],
                        "dataset": group_cfg["dataset"],
                        "timepoint": group_cfg["timepoint"],
                        "branch": branch,
                        "metric": metric,
                        "trial": t_idx,
                        "mean": m,
                        "sem": s,
                        "n": int(nn),
                    }
                )
    return pd.DataFrame(rows)


def plot_branch(df: pd.DataFrame, branch: str, outpath: Path) -> None:
    sub = df[df["branch"] == branch].copy()
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    order = list(ADVICE_METRICS.keys())

    color_map = {g["name"]: g["color"] for g in GROUPS}
    for ax, metric in zip(axes.flat, order):
        d = sub[sub["metric"] == metric].copy()
        for grp in [g["name"] for g in GROUPS]:
            gd = d[d["group"] == grp].sort_values("trial")
            x = gd["trial"].to_numpy(dtype=float)
            y = gd["mean"].to_numpy(dtype=float)
            sem = gd["sem"].to_numpy(dtype=float)
            c = color_map[grp]
            ax.plot(x, y, color=c, linewidth=1.8, label=grp)
            ax.fill_between(x, y - sem, y + sem, color=c, alpha=0.18)
        ax.set_title(PANEL_LABELS[metric])
        ax.grid(alpha=0.2)
        ax.set_xlim(left=1)
    for ax in axes[1, :]:
        ax.set_xlabel("Trial")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean ± SEM")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.01))
    title = "Advice branch belief states" if branch == "advice" else "Reward branch belief states"
    fig.suptitle(title, fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    all_summaries = []
    group_meta = []
    for cfg in GROUPS:
        states = compute_trialwise_states(cfg)
        summary = summarise_group(cfg, states)
        all_summaries.append(summary)
        group_meta.append(
            {
                "group": cfg["name"],
                "dataset": cfg["dataset"],
                "timepoint": cfg["timepoint"],
                "n_sessions": int(states["n_sessions"][0]),
                "mean_n_trials": float(np.mean(states["trial_count_summary"])),
                "min_n_trials": int(np.min(states["trial_count_summary"])),
                "max_n_trials": int(np.max(states["trial_count_summary"])),
                "plot_capped_at_trial": MAX_TRIAL,
            }
        )

    out = pd.concat(all_summaries, ignore_index=True)
    out.to_csv(OUT_DIR / "belief_state_group_trajectories_long.csv", index=False)
    pd.DataFrame(group_meta).to_csv(OUT_DIR / "belief_state_group_metadata.csv", index=False)

    plot_branch(out, "advice", OUT_DIR / "figure_belief_states_advice_branch.png")
    plot_branch(out, "reward", OUT_DIR / "figure_belief_states_reward_branch.png")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(group_meta, f, indent=2)


if __name__ == "__main__":
    main()
