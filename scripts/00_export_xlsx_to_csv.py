from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import normalize_columns


def pick_sheet(xlsx_path: Path, preferred: list[str]) -> str:
    xls = pd.ExcelFile(xlsx_path)
    lower = {s.lower(): s for s in xls.sheet_names}
    for name in preferred:
        if name.lower() in lower:
            return lower[name.lower()]
    # normalized matching
    preferred_norm = {name.lower().replace("_", "") for name in preferred}
    for s in xls.sheet_names:
        if s.lower().replace("_", "") in preferred_norm:
            return s
    return xls.sheet_names[0]


def export_one(xlsx_path: Path, preferred_sheets: list[str], out_csv: Path) -> None:
    sheet = pick_sheet(xlsx_path, preferred_sheets)
    df = normalize_columns(pd.read_excel(xlsx_path, sheet_name=sheet))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Exported {xlsx_path.name} / sheet '{sheet}' -> {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Table S2 and Table S3 Excel sheets to CSV.")
    parser.add_argument("--s2", default="data/raw/individual_endpoints.xlsx", help="Path to Table S2 Excel file")
    parser.add_argument("--s3", default="data/raw/speed_step_data.xlsx", help="Path to Table S3 Excel file")
    parser.add_argument("--out-dir", default="data/processed", help="Output directory for CSV files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    export_one(Path(args.s2), ["Table_S2", "S2", "Sheet1"], out_dir / "individual_endpoints.csv")
    export_one(Path(args.s3), ["Table_S3", "S3", "Sheet1"], out_dir / "speed_step_data.csv")


if __name__ == "__main__":
    main()
