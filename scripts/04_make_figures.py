from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_columns, read_table, to_numeric


REQUIRED = [
    "experimental_condition", "silvering_stage", "swimming_speed_m_s",
    "mo2_mg_o2_kg_h", "cot_mg_o2_kg_km"
]


def add_mean_sd(ax, df: pd.DataFrame, value_col: str, label: str) -> None:
    g = (
        df.dropna(subset=[value_col])
        .groupby("swimming_speed_m_s")
        .agg(n=(value_col, "count"), mean=(value_col, "mean"), sd=(value_col, lambda x: x.std(ddof=1)))
        .reset_index()
    )
    if g.empty:
        return
    g["se"] = g["sd"] / np.sqrt(g["n"])
    ax.errorbar(g["swimming_speed_m_s"], g["mean"], yerr=g["sd"], marker="o", label=label, capsize=3)


def make_plot(df: pd.DataFrame, value_col: str, y_label: str, out_path: Path, active_only: bool = False) -> None:
    d = df.copy()
    if active_only:
        d = d[d["swimming_speed_m_s"] >= 0.4]

    fig, ax = plt.subplots(figsize=(7, 5))
    for cond in ["FW25", "FW18", "SW18"]:
        sub = d[d["experimental_condition"] == cond]
        if sub.empty:
            continue
        # Individual points, slightly transparent; matplotlib default colors are used.
        ax.scatter(sub["swimming_speed_m_s"], sub[value_col], alpha=0.35, s=20)
        add_mean_sd(ax, sub, value_col, cond)
    ax.set_xlabel("Swimming speed (m s$^{-1}$)")
    ax.set_ylabel(y_label)
    ax.legend(title="Condition")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make preliminary revised figures from Table S3.")
    parser.add_argument("--s3", default="data/processed/speed_step_data.csv")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    df = read_table(Path(args.s3), ["Table_S3"])
    ensure_columns(df, REQUIRED, "Table S3")
    for c in ["swimming_speed_m_s", "mo2_mg_o2_kg_h", "cot_mg_o2_kg_km"]:
        df[c] = to_numeric(df[c])

    make_plot(
        df,
        value_col="mo2_mg_o2_kg_h",
        y_label="MO$_2$ (mg O$_2$ kg$^{-1}$ h$^{-1}$)",
        out_path=out_dir / "fig_mo2_speed_by_condition.png",
        active_only=False,
    )
    make_plot(
        df,
        value_col="cot_mg_o2_kg_km",
        y_label="COT (mg O$_2$ kg$^{-1}$ km$^{-1}$)",
        out_path=out_dir / "fig_cot_speed_by_condition.png",
        active_only=True,
    )
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
