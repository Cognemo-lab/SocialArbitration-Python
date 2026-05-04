from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("/Users/drea/Documents/CAMH/Projects/McGill-Collaboration")
IN_FILE = BASE / "hitop" / "hierarchical_fod_moderation_features" / "table_hierarchical_moderation_interactions_pretty.csv"
OUT_DIR = BASE / "hitop" / "hierarchical_fod_moderation_features"


def main() -> None:
    df = pd.read_csv(IN_FILE)
    # Word-friendly outputs: Excel workbook and tab-delimited text.
    df.to_excel(OUT_DIR / "table_hierarchical_moderation_interactions_word.xlsx", index=False)
    df.to_csv(OUT_DIR / "table_hierarchical_moderation_interactions_word.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
