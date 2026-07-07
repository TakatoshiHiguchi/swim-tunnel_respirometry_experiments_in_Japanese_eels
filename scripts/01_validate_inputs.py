from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ensure_columns, read_table


REQUIRED_S2 = [
    "fish_code", "trial_id", "experimental_condition", "water_medium", "temperature_c",
    "silvering_stage", "tl_used_cm", "bw_used_g", "ucrit_m_s", "ucrit_tl_s",
    "uopt_m_s", "uopt_tl_s", "mincot_mg_o2_kg_km", "mo2_at_uopt_mg_o2_kg_h",
    "mo2_at_0_1_m_s_mg_o2_kg_h", "n_active_speed_steps", "ucrit_status",
    "active_metrics_status", "lowflow_mo2_status"
]

REQUIRED_S3 = [
    "fish_code", "trial_id", "experimental_condition", "water_medium", "temperature_c",
    "silvering_stage", "swimming_speed_m_s", "mo2_mg_o2_kg_h", "cot_mg_o2_kg_km"
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Table S2 and S3 analysis inputs.")
    parser.add_argument("--s2", default="data/processed/individual_endpoints.csv")
    parser.add_argument("--s3", default="data/processed/speed_step_data.csv")
    parser.add_argument("--out", default="outputs/tables/input_validation_report.csv")
    args = parser.parse_args()

    s2 = read_table(Path(args.s2), ["Table_S2"])
    s3 = read_table(Path(args.s3), ["Table_S3"])
    ensure_columns(s2, REQUIRED_S2, "Table S2")
    ensure_columns(s3, REQUIRED_S3, "Table S3")

    rows = []
    rows.append({"check": "Table S2 rows", "value": len(s2), "status": "PASS" if len(s2) == 35 else "CHECK"})
    rows.append({"check": "Table S3 rows", "value": len(s3), "status": "INFO"})
    rows.append({"check": "Table S2 unique trial_id", "value": s2["trial_id"].nunique(), "status": "PASS" if s2["trial_id"].nunique() == len(s2) else "FAIL"})
    rows.append({"check": "Table S3 unique trial_id", "value": s3["trial_id"].nunique(), "status": "INFO"})

    for cond in ["FW25", "FW18", "SW18"]:
        rows.append({
            "check": f"Table S2 count {cond}",
            "value": int((s2["experimental_condition"].astype(str) == cond).sum()),
            "status": "INFO",
        })

    # Check user-confirmed missing speed-step trials.
    expected_missing = {"E09_FW25", "E11_SW18"}
    present_trials = set(s3["trial_id"].astype(str))
    for trial in sorted(expected_missing):
        rows.append({
            "check": f"Confirmed missing speed-step data: {trial}",
            "value": "absent from Table S3" if trial not in present_trials else "present",
            "status": "PASS" if trial not in present_trials else "CHECK",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Validation report written to {out}")


if __name__ == "__main__":
    main()
