from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
IN_FILE = BASE / "hitop" / "hierarchical_fod_moderation_features" / "hierarchical_moderation_interactions.csv"
OUT_DIR = BASE / "hitop" / "hierarchical_fod_moderation_features"

LABELS = {
    "abs_eps2_a_mean_z:hitop_suicidality_z": "Abs advice epsilon2 x suicidality",
    "inferv_a_mean_z:hitop_suicidality_z": "Advice inferential variance x suicidality",
    "ka_a_z:hitop_suicidality_z": "ka_a x suicidality",
    "om_a_z:hitop_suicidality_z": "om_a x suicidality",
    "eps3_a_mean_z:hitop_suicidality_z": "Advice epsilon3 x suicidality",
}


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def main() -> None:
    df = pd.read_csv(IN_FILE).copy()
    df["predictor"] = df["term"].map(LABELS).fillna(df["term"])
    df["beta_fmt"] = df["coef"].map(lambda x: f"{x:.3f}")
    df["se_fmt"] = df["se"].map(lambda x: f"{x:.3f}")
    df["ci_fmt"] = df.apply(lambda r: f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1)
    df["z_fmt"] = df["z_value"].map(lambda x: f"{x:.3f}")
    df["p_fmt"] = df["p_value"].map(lambda x: f"{x:.6f}" if x < 0.001 else f"{x:.3f}")
    df["sig"] = df["p_value"].map(stars)

    out = df[
        ["predictor", "coef", "se", "ci_low", "ci_high", "z_value", "p_value", "sig", "beta_fmt", "se_fmt", "ci_fmt", "z_fmt", "p_fmt"]
    ].sort_values("coef")

    out.to_csv(OUT_DIR / "table_hierarchical_moderation_interactions.csv", index=False)

    pretty = out[["predictor", "beta_fmt", "se_fmt", "ci_fmt", "z_fmt", "p_fmt", "sig"]].rename(
        columns={
            "predictor": "Predictor x suicidality",
            "beta_fmt": "Beta",
            "se_fmt": "SE",
            "ci_fmt": "95% CI",
            "z_fmt": "z",
            "p_fmt": "p",
            "sig": "Sig",
        }
    )
    pretty.to_csv(OUT_DIR / "table_hierarchical_moderation_interactions_pretty.csv", index=False)


if __name__ == "__main__":
    main()
