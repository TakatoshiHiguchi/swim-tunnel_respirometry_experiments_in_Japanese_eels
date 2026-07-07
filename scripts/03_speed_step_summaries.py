from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ensure_columns, read_table, to_numeric


REQUIRED = [
    "fish_code", "trial_id", "experimental_condition", "water_medium", "temperature_c",
    "silvering_stage", "swimming_speed_m_s", "mo2_mg_o2_kg_h", "cot_mg_o2_kg_km"
]


def summarize(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    d[value_col] = to_numeric(d[value_col])
    out = (
        d.dropna(subset=[value_col])
        .groupby(group_cols, dropna=False)
        .agg(
            n=(value_col, "count"),
            mean=(value_col, "mean"),
            sd=(value_col, lambda x: x.std(ddof=1)),
            median=(value_col, "median"),
            min=(value_col, "min"),
            max=(value_col, "max"),
        )
        .reset_index()
    )
    out["se"] = out["sd"] / np.sqrt(out["n"])
    out.insert(len(group_cols), "response_variable", value_col)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize speed-step-level MO2 and COT data from Table S3.")
    parser.add_argument("--s3", default="data/processed/speed_step_data.csv")
    parser.add_argument("--out-dir", default="outputs/tables")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(Path(args.s3), ["Table_S3"])
    ensure_columns(df, REQUIRED, "Table S3")
    for c in ["temperature_c", "swimming_speed_m_s", "mo2_mg_o2_kg_h", "cot_mg_o2_kg_km"]:
        df[c] = to_numeric(df[c])

    # User-confirmed speed sequence: 0.1, 0.4, ..., 1.3 m/s.
    expected = [0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    observed = sorted(round(float(x), 1) for x in df["swimming_speed_m_s"].dropna().unique())
    speed_check = pd.DataFrame({
        "check": ["observed_speed_sequence", "expected_speed_sequence", "matches_expected"],
        "value": [str(observed), str(expected), str(observed == expected)],
    })
    speed_check.to_csv(out_dir / "speed_sequence_check.csv", index=False, encoding="utf-8-sig")

    group_cols = ["experimental_condition", "silvering_stage", "swimming_speed_m_s"]
    mo2 = summarize(df, "mo2_mg_o2_kg_h", group_cols)
    cot = summarize(df[df["swimming_speed_m_s"] >= 0.4], "cot_mg_o2_kg_km", group_cols)

    mo2.to_csv(out_dir / "mo2_speed_summary_by_condition.csv", index=False, encoding="utf-8-sig")
    cot.to_csv(out_dir / "cot_speed_summary_by_condition.csv", index=False, encoding="utf-8-sig")

    n_summary = (
        df.groupby(["experimental_condition", "silvering_stage", "swimming_speed_m_s"], dropna=False)
        .agg(
            n_mo2=("mo2_mg_o2_kg_h", lambda x: x.notna().sum()),
            n_cot=("cot_mg_o2_kg_km", lambda x: x.notna().sum()),
            n_trials=("trial_id", "nunique"),
        )
        .reset_index()
    )
    n_summary.to_csv(out_dir / "speed_n_summary.csv", index=False, encoding="utf-8-sig")

    # Also write a compact Excel workbook for convenience.
    with pd.ExcelWriter(out_dir / "speed_step_summaries.xlsx", engine="openpyxl") as writer:
        mo2.to_excel(writer, sheet_name="MO2_speed_summary", index=False)
        cot.to_excel(writer, sheet_name="COT_speed_summary", index=False)
        n_summary.to_excel(writer, sheet_name="Speed_n_summary", index=False)
        speed_check.to_excel(writer, sheet_name="Speed_sequence_check", index=False)

    print(f"Wrote speed-step summaries to {out_dir}")


if __name__ == "__main__":
    main()
