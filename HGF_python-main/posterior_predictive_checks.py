from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from HGF.code_inversion.tapas_sgm import tapas_sgm
from HGF.code_model_wagad.obs_model.linear_volatilitydecnoise_1stlevelprecision_reward_social_sim import (
    linear_volatilitydecnoise_1stlevelprecision_reward_social_sim,
)
from HGF.code_model_wagad.prc_model.hgf_binary3l_freekappa_reward_social import (
    hgf_binary3l_freekappa_reward_social,
)


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


def load_fit_json(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    payload = json.loads(path.read_text())
    p_prc = np.array([payload["p_prc"][k] for k in PRC_ORDER], dtype=float)
    p_obs = np.array([payload["p_obs"][k] for k in OBS_ORDER], dtype=float)
    meta = {
        "prolific_id": payload["prolific_id"],
        "round_name": payload["round_name"],
        "timepoint": payload["timepoint"],
    }
    return p_prc, p_obs, meta


def compute_model_quantities(
    u: np.ndarray, prc_vec: np.ndarray, obs_vec: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    r = {"u": u, "ign": []}
    traj, inf_states = hgf_binary3l_freekappa_reward_social(r, prc_vec)

    be0, be1, be2, be3, be4, be5, be6, ze, be_ch, be_wager = obs_vec

    reward_location = u[:, 0]
    advice_helpfulness = u[:, 1]
    advice_in_loc_coords = 1.0 - np.abs(reward_location - advice_helpfulness)

    mu1hat_a = inf_states[:, 0, 2]
    mu1hat_r = inf_states[:, 0, 0]
    mu2hat_a = inf_states[:, 1, 2]
    mu2hat_r = inf_states[:, 1, 0]
    sa2hat_r = inf_states[:, 1, 1]
    sa2hat_a = inf_states[:, 1, 3]
    mu3hat_r = inf_states[:, 2, 0]
    mu3hat_a = inf_states[:, 2, 2]

    transformed_mu1hat_r = mu1hat_r ** advice_in_loc_coords * (1.0 - mu1hat_r) ** (1.0 - advice_in_loc_coords)
    px = 1.0 / (mu1hat_a * (1.0 - mu1hat_a))
    pc = 1.0 / (mu1hat_r * (1.0 - mu1hat_r))
    wx = ze * px / (ze * px + pc)
    wc = pc / (ze * px + pc)
    b = np.clip(wx * mu1hat_a + wc * transformed_mu1hat_r, 1e-8, 1.0 - 1e-8)
    decision_noise = np.exp(-mu3hat_r - mu3hat_a) / be_ch
    choice_prob = b ** decision_noise / (b ** decision_noise + (1.0 - b) ** decision_noise)

    inferv_a = tapas_sgm(mu2hat_a, 1.0) * (1.0 - tapas_sgm(mu2hat_a, 1.0)) * sa2hat_a
    inferv_r = tapas_sgm(mu2hat_r, 1.0) * (1.0 - tapas_sgm(mu2hat_r, 1.0)) * sa2hat_r
    pv_a = tapas_sgm(mu2hat_a, 1.0) * (1.0 - tapas_sgm(mu2hat_a, 1.0)) * np.exp(mu3hat_a)
    pv_r = tapas_sgm(mu2hat_r, 1.0) * (1.0 - tapas_sgm(mu2hat_r, 1.0)) * np.exp(mu3hat_r)
    surprise = -np.log2(b)
    wager_pred = be0 + be1 * surprise + be2 * wx + be3 * inferv_a + be4 * inferv_r + be5 * pv_a + be6 * pv_r

    return (
        {
            "choice_prob": choice_prob,
            "wager_pred": wager_pred,
            "arbitration": wx,
            "belief": b,
        },
        inf_states,
    )


def stay_rate(x: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    return float(np.mean(x[1:] == x[:-1]))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def neg_log_lik_bernoulli(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def calibration_table(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    bins = pd.qcut(df["choice_prob"], q=n_bins, duplicates="drop")
    out = (
        df.assign(bin=bins)
        .groupby("bin", observed=False)
        .agg(
            n=("choice_obs", "size"),
            mean_pred=("choice_prob", "mean"),
            mean_obs=("choice_obs", "mean"),
        )
        .reset_index(drop=True)
    )
    out["abs_calibration_error"] = np.abs(out["mean_obs"] - out["mean_pred"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Posterior predictive checks for fitted HGF model.")
    parser.add_argument(
        "--fit-dir",
        default="/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full/fit_jsons",
    )
    parser.add_argument(
        "--trials-csv",
        default="/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full/extracted_model_trials.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="/Users/drea/Documents/CAMH/Projects/McGill-Collaboration/hgf_baselinefixed_full/posterior_predictive_checks",
    )
    parser.add_argument("--n-reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260318)
    args = parser.parse_args()

    fit_dir = Path(args.fit_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    trials = pd.read_csv(args.trials_csv)
    rng = np.random.RandomState(args.seed)

    trial_rows = []
    session_rows = []
    rep_rows = []

    fit_files = sorted(fit_dir.glob("*.json"))
    for idx, fit_path in enumerate(fit_files, start=1):
        prc_vec, obs_vec, meta = load_fit_json(fit_path)
        g = (
            trials.loc[
                (trials["prolific_id"] == meta["prolific_id"])
                & (trials["round_name"] == meta["round_name"])
            ]
            .sort_values("trial_index_choice")
            .reset_index(drop=True)
        )
        if g.empty:
            continue

        u = g[["input_advice", "input_reward", "advice_card_space"]].to_numpy(dtype=float)
        y_choice = g["choice_advice_taken"].to_numpy(dtype=float)
        y_wager = g["wager"].to_numpy(dtype=float)
        y_side = g["choice_side"].to_numpy(dtype=float)

        q, inf_states = compute_model_quantities(u, prc_vec, obs_vec)
        choice_prob = q["choice_prob"]
        wager_pred = q["wager_pred"]

        for i in range(len(g)):
            trial_rows.append(
                {
                    **meta,
                    "trial_index": int(i + 1),
                    "choice_obs": float(y_choice[i]),
                    "choice_prob": float(choice_prob[i]),
                    "wager_obs": float(y_wager[i]),
                    "wager_pred": float(wager_pred[i]),
                    "choice_side_obs": float(y_side[i]),
                    "arbitration": float(q["arbitration"][i]),
                    "belief": float(q["belief"][i]),
                }
            )

        obs_summary = {
            "choice_mean_obs": float(np.mean(y_choice)),
            "choice_sd_obs": float(np.std(y_choice, ddof=0)),
            "stay_rate_obs": stay_rate(y_side),
            "wager_mean_obs": float(np.mean(y_wager)),
            "wager_sd_obs": float(np.std(y_wager, ddof=0)),
        }
        pred_summary = {
            "choice_prob_mean_pred": float(np.mean(choice_prob)),
            "wager_mean_pred": float(np.mean(wager_pred)),
            "wager_sd_pred_mean": float(np.sqrt(obs_vec[9])),
            "choice_brier": brier_score(y_choice, choice_prob),
            "choice_nll": neg_log_lik_bernoulli(y_choice, choice_prob),
            "wager_rmse": float(np.sqrt(np.mean((y_wager - wager_pred) ** 2))),
            "wager_corr": safe_corr(y_wager, wager_pred),
        }

        rep_summaries = []
        for rep in range(args.n_reps):
            np.random.seed(rng.randint(0, 2**31 - 1))
            y_sim, _ = linear_volatilitydecnoise_1stlevelprecision_reward_social_sim(
                {"u": u, "ign": []}, inf_states, obs_vec
            )
            rep_summary = {
                "choice_mean": float(np.mean(y_sim[:, 1])),
                "choice_sd": float(np.std(y_sim[:, 1], ddof=0)),
                "stay_rate": stay_rate(y_sim[:, 1]),
                "wager_mean": float(np.mean(y_sim[:, 0])),
                "wager_sd": float(np.std(y_sim[:, 0], ddof=0)),
            }
            rep_summaries.append(rep_summary)
            rep_rows.append({**meta, "replicate": rep + 1, **rep_summary})

        rep_df = pd.DataFrame(rep_summaries)
        session_rows.append(
            {
                **meta,
                "n_trials": int(len(g)),
                **obs_summary,
                **pred_summary,
                "choice_mean_rep_mean": float(rep_df["choice_mean"].mean()),
                "choice_mean_rep_lo": float(rep_df["choice_mean"].quantile(0.025)),
                "choice_mean_rep_hi": float(rep_df["choice_mean"].quantile(0.975)),
                "wager_mean_rep_mean": float(rep_df["wager_mean"].mean()),
                "wager_mean_rep_lo": float(rep_df["wager_mean"].quantile(0.025)),
                "wager_mean_rep_hi": float(rep_df["wager_mean"].quantile(0.975)),
                "wager_sd_rep_mean": float(rep_df["wager_sd"].mean()),
                "wager_sd_rep_lo": float(rep_df["wager_sd"].quantile(0.025)),
                "wager_sd_rep_hi": float(rep_df["wager_sd"].quantile(0.975)),
                "choice_in_ppc_interval": bool(
                    obs_summary["choice_mean_obs"] >= rep_df["choice_mean"].quantile(0.025)
                    and obs_summary["choice_mean_obs"] <= rep_df["choice_mean"].quantile(0.975)
                ),
                "wager_in_ppc_interval": bool(
                    obs_summary["wager_mean_obs"] >= rep_df["wager_mean"].quantile(0.025)
                    and obs_summary["wager_mean_obs"] <= rep_df["wager_mean"].quantile(0.975)
                ),
            }
        )

        if idx % 50 == 0:
            print(f"[{idx}/{len(fit_files)}] processed")

    trial_df = pd.DataFrame(trial_rows)
    session_df = pd.DataFrame(session_rows)
    rep_df = pd.DataFrame(rep_rows)
    trial_df.to_csv(out_dir / "trial_level_predictions.csv", index=False)
    session_df.to_csv(out_dir / "session_level_ppc_summary.csv", index=False)
    rep_df.to_csv(out_dir / "replicate_session_summaries.csv", index=False)

    calibration = calibration_table(trial_df)
    calibration.to_csv(out_dir / "choice_calibration_table.csv", index=False)

    choice_auc = roc_auc_score(trial_df["choice_obs"], trial_df["choice_prob"])
    choice_brier_global = brier_score(
        trial_df["choice_obs"].to_numpy(float), trial_df["choice_prob"].to_numpy(float)
    )
    choice_nll_global = neg_log_lik_bernoulli(
        trial_df["choice_obs"].to_numpy(float), trial_df["choice_prob"].to_numpy(float)
    )
    wager_rmse_global = float(
        np.sqrt(np.mean((trial_df["wager_obs"].to_numpy(float) - trial_df["wager_pred"].to_numpy(float)) ** 2))
    )
    wager_corr_global = safe_corr(
        trial_df["wager_obs"].to_numpy(float), trial_df["wager_pred"].to_numpy(float)
    )
    wager_r2_global = float(
        1.0
        - np.sum((trial_df["wager_obs"] - trial_df["wager_pred"]) ** 2)
        / np.sum((trial_df["wager_obs"] - trial_df["wager_obs"].mean()) ** 2)
    )

    aggregate = {
        "n_sessions": int(len(session_df)),
        "n_trials": int(len(trial_df)),
        "n_replicates_per_session": int(args.n_reps),
        "choice_auc": float(choice_auc),
        "choice_brier": float(choice_brier_global),
        "choice_neg_log_lik": float(choice_nll_global),
        "wager_rmse": float(wager_rmse_global),
        "wager_corr": float(wager_corr_global),
        "wager_r2": float(wager_r2_global),
        "session_choice_mean_corr_obs_vs_pred": float(
            safe_corr(session_df["choice_mean_obs"].to_numpy(float), session_df["choice_prob_mean_pred"].to_numpy(float))
        ),
        "session_wager_mean_corr_obs_vs_pred": float(
            safe_corr(session_df["wager_mean_obs"].to_numpy(float), session_df["wager_mean_pred"].to_numpy(float))
        ),
        "choice_mean_ppc_coverage": float(session_df["choice_in_ppc_interval"].mean()),
        "wager_mean_ppc_coverage": float(session_df["wager_in_ppc_interval"].mean()),
        "mean_abs_choice_calibration_error": float(calibration["abs_calibration_error"].mean()),
    }
    (out_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2))

    plt.figure(figsize=(5.5, 5.2))
    plt.plot([0, 1], [0, 1], linestyle="--", color="0.5")
    plt.scatter(calibration["mean_pred"], calibration["mean_obs"], s=50, color="#264653")
    plt.xlabel("Predicted choice probability")
    plt.ylabel("Observed choice rate")
    plt.title("Choice Calibration")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure_choice_calibration.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5.5, 5.2))
    plt.scatter(session_df["choice_prob_mean_pred"], session_df["choice_mean_obs"], s=18, alpha=0.5, color="#2a9d8f")
    mn = min(session_df["choice_prob_mean_pred"].min(), session_df["choice_mean_obs"].min())
    mx = max(session_df["choice_prob_mean_pred"].max(), session_df["choice_mean_obs"].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", color="0.5")
    plt.xlabel("Predicted session choice mean")
    plt.ylabel("Observed session choice mean")
    plt.title("Choice Mean by Session")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure_session_choice_mean_scatter.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5.5, 5.2))
    plt.scatter(session_df["wager_mean_pred"], session_df["wager_mean_obs"], s=18, alpha=0.5, color="#e76f51")
    mn = min(session_df["wager_mean_pred"].min(), session_df["wager_mean_obs"].min())
    mx = max(session_df["wager_mean_pred"].max(), session_df["wager_mean_obs"].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", color="0.5")
    plt.xlabel("Predicted session wager mean")
    plt.ylabel("Observed session wager mean")
    plt.title("Wager Mean by Session")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure_session_wager_mean_scatter.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5.5, 5.2))
    plt.scatter(trial_df["wager_pred"], trial_df["wager_obs"], s=8, alpha=0.15, color="#bc6c25")
    mn = min(trial_df["wager_pred"].min(), trial_df["wager_obs"].min())
    mx = max(trial_df["wager_pred"].max(), trial_df["wager_obs"].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", color="0.5")
    plt.xlabel("Predicted wager")
    plt.ylabel("Observed wager")
    plt.title("Trial-Level Wager Fit")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure_trial_wager_scatter.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(rep_df["choice_mean"], bins=30, color="#8ecae6", alpha=0.8)
    axes[0].axvline(session_df["choice_mean_obs"].mean(), color="#023047", lw=2)
    axes[0].set_title("Replicated Session Choice Means")
    axes[0].set_xlabel("Choice mean")
    axes[0].set_ylabel("Count")

    axes[1].hist(rep_df["wager_mean"], bins=30, color="#ffb703", alpha=0.8)
    axes[1].axvline(session_df["wager_mean_obs"].mean(), color="#9b2226", lw=2)
    axes[1].set_title("Replicated Session Wager Means")
    axes[1].set_xlabel("Wager mean")
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_ppc_session_mean_histograms.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
