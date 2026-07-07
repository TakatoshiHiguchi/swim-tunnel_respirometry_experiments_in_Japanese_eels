from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import (
    bh_fdr,
    bootstrap_ci_independent,
    bootstrap_ci_paired,
    clean_status,
    clean_text,
    ensure_columns,
    independent_permutation_p,
    mean_sd_median,
    paired_permutation_p,
    read_table,
    to_numeric,
    write_csv_and_xlsx,
)


ENDPOINTS = {
    "ucrit_m_s": "ucrit",
    "ucrit_tl_s": "ucrit",
    "uopt_m_s": "active",
    "uopt_tl_s": "active",
    "mincot_mg_o2_kg_km": "active",
    "mo2_at_uopt_mg_o2_kg_h": "active",
    "mo2_at_0_1_m_s_mg_o2_kg_h": "lowflow",
}

REQUIRED = [
    "fish_code", "trial_id", "experimental_condition", "water_medium", "temperature_c",
    "silvering_stage", "tl_used_cm", "bw_used_g", "ucrit_status",
    "active_metrics_status", "lowflow_mo2_status", *ENDPOINTS.keys()
]


UCRIT_ALLOWED = {"valid", "boundary_ambiguous", "protocol_check"}
ACTIVE_MAIN_ALLOWED = {"valid", "boundary_estimate"}
ACTIVE_VALID_ONLY = {"valid"}
LOWFLOW_ALLOWED = {"valid"}


def prepare_s2(path: Path) -> pd.DataFrame:
    df = read_table(path, ["Table_S2"])
    ensure_columns(df, REQUIRED, "Table S2")
    for c in ENDPOINTS.keys():
        df[c] = to_numeric(df[c])
    for c in ["tl_used_cm", "bw_used_g", "temperature_c"]:
        df[c] = to_numeric(df[c])
    for c in ["fish_code", "trial_id", "experimental_condition", "water_medium", "silvering_stage"]:
        df[c] = df[c].map(clean_text)
    for c in ["ucrit_status", "active_metrics_status", "lowflow_mo2_status"]:
        df[c] = df[c].map(clean_status)
    return df


def analysis_filter(df: pd.DataFrame, response: str, analysis_set: str) -> pd.DataFrame:
    kind = ENDPOINTS[response]
    d = df.copy()
    if kind == "ucrit":
        keep = d["ucrit_status"].isin(UCRIT_ALLOWED)
    elif kind == "active":
        allowed = ACTIVE_VALID_ONLY if analysis_set == "sensitivity_valid_only" else ACTIVE_MAIN_ALLOWED
        keep = d["active_metrics_status"].isin(allowed)
    elif kind == "lowflow":
        keep = d["lowflow_mo2_status"].isin(LOWFLOW_ALLOWED)
    else:
        raise ValueError(f"Unknown endpoint kind: {kind}")
    keep &= np.isfinite(d[response])
    return d.loc[keep].copy()


def count_excluded(df_all: pd.DataFrame, df_used: pd.DataFrame, response: str, mask_population: pd.Series) -> tuple[int, str]:
    pop = df_all.loc[mask_population].copy()
    used_trials = set(df_used["trial_id"].astype(str))
    excluded = pop[~pop["trial_id"].astype(str).isin(used_trials)]
    if excluded.empty:
        return 0, ""
    reasons = []
    kind = ENDPOINTS[response]
    if kind == "ucrit":
        col = "ucrit_status"
    elif kind == "active":
        col = "active_metrics_status"
    else:
        col = "lowflow_mo2_status"
    for status, n in excluded[col].fillna("missing_status").astype(str).value_counts().items():
        reasons.append(f"{status}:{int(n)}")
    missing_value = excluded[response].isna().sum()
    if missing_value:
        reasons.append(f"missing_value:{int(missing_value)}")
    return int(len(excluded)), "; ".join(reasons)


def paired_comparison(df_all: pd.DataFrame, response: str, analysis_set: str, seed: int, n_boot: int, n_perm: int) -> dict:
    """FW25 - FW18 paired comparison in yellow eels."""
    pop_mask = df_all["experimental_condition"].isin(["FW25", "FW18"])
    d = analysis_filter(df_all.loc[pop_mask], response, analysis_set)
    d = d[d["experimental_condition"].isin(["FW25", "FW18"])]
    pivot = d.pivot_table(index="fish_code", columns="experimental_condition", values=response, aggfunc="first")
    pivot = pivot.dropna(subset=["FW25", "FW18"])
    x = pivot["FW25"].to_numpy(float)
    y = pivot["FW18"].to_numpy(float)

    s1 = mean_sd_median(x)
    s2 = mean_sd_median(y)
    diff = x - y
    diff_stats = mean_sd_median(diff)
    ci_lo, ci_hi = bootstrap_ci_paired(x, y, n_boot=n_boot, seed=seed)
    p = paired_permutation_p(x, y, n_perm=n_perm, seed=seed)
    n_excl, reasons = count_excluded(df_all, d, response, pop_mask)

    return {
        "comparison": "FW25_vs_FW18_paired",
        "comparison_description": "Sequential FW25 and FW18 trials in the same yellow eels; estimate = FW25 - FW18.",
        "response_variable": response,
        "analysis_set": analysis_set,
        "group_1": "FW25",
        "group_2": "FW18",
        "estimate_definition": "group_1_minus_group_2",
        "n_group_1": s1["n"],
        "n_group_2": s2["n"],
        "n_pairs": int(len(pivot)),
        "mean_group_1": s1["mean"],
        "sd_group_1": s1["sd"],
        "median_group_1": s1["median"],
        "mean_group_2": s2["mean"],
        "sd_group_2": s2["sd"],
        "median_group_2": s2["median"],
        "estimate_difference": diff_stats["mean"],
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "test_method": "paired sign-flip permutation test; bootstrap percentile CI for mean paired difference",
        "p_value": p,
        "n_excluded_from_population": n_excl,
        "exclusion_reason_summary": reasons,
        "notes": "Fixed trial order; interpret as temperature-associated difference, not isolated temperature effect.",
    }


def independent_comparison(
    df_all: pd.DataFrame,
    response: str,
    analysis_set: str,
    comparison: str,
    group1_mask: pd.Series,
    group2_mask: pd.Series,
    group1_name: str,
    group2_name: str,
    estimate_note: str,
    interpretation_note: str,
    seed: int,
    n_boot: int,
    n_perm: int,
) -> dict:
    pop_mask = group1_mask | group2_mask
    d = analysis_filter(df_all.loc[pop_mask], response, analysis_set)
    x = d.loc[group1_mask.loc[d.index], response].to_numpy(float)
    y = d.loc[group2_mask.loc[d.index], response].to_numpy(float)

    s1 = mean_sd_median(x)
    s2 = mean_sd_median(y)
    ci_lo, ci_hi = bootstrap_ci_independent(x, y, n_boot=n_boot, seed=seed)
    p = independent_permutation_p(x, y, n_perm=n_perm, seed=seed)
    n_excl, reasons = count_excluded(df_all, d, response, pop_mask)

    return {
        "comparison": comparison,
        "comparison_description": estimate_note,
        "response_variable": response,
        "analysis_set": analysis_set,
        "group_1": group1_name,
        "group_2": group2_name,
        "estimate_definition": "group_2_minus_group_1",
        "n_group_1": s1["n"],
        "n_group_2": s2["n"],
        "n_pairs": np.nan,
        "mean_group_1": s1["mean"],
        "sd_group_1": s1["sd"],
        "median_group_1": s1["median"],
        "mean_group_2": s2["mean"],
        "sd_group_2": s2["sd"],
        "median_group_2": s2["median"],
        "estimate_difference": s2["mean"] - s1["mean"] if s1["n"] > 0 and s2["n"] > 0 else np.nan,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "test_method": "two-sample permutation test; bootstrap percentile CI for mean difference",
        "p_value": p,
        "n_excluded_from_population": n_excl,
        "exclusion_reason_summary": reasons,
        "notes": interpretation_note,
    }


def make_s4(df: pd.DataFrame, seed: int, n_boot: int, n_perm: int) -> pd.DataFrame:
    rows = []
    analysis_sets_for_kind = {
        "ucrit": ["main"],
        "lowflow": ["main"],
        "active": ["main_valid_plus_boundary", "sensitivity_valid_only"],
    }

    for response, kind in ENDPOINTS.items():
        for analysis_set in analysis_sets_for_kind[kind]:
            # 1. FW25 vs FW18 paired.
            rows.append(paired_comparison(df, response, analysis_set, seed, n_boot, n_perm))

            # 2. Y2 FW18 vs Y2 SW18; estimate SW18 - FW18.
            g1 = (df["silvering_stage"] == "Y2") & (df["experimental_condition"] == "FW18")
            g2 = (df["silvering_stage"] == "Y2") & (df["experimental_condition"] == "SW18")
            rows.append(independent_comparison(
                df, response, analysis_set,
                comparison="Y2_FW18_vs_Y2_SW18_independent",
                group1_mask=g1, group2_mask=g2,
                group1_name="Y2_FW18", group2_name="Y2_SW18",
                estimate_note="Exploratory Y2 freshwater-seawater comparison at 18 C; estimate = SW18 - FW18.",
                interpretation_note="Exploratory salinity-associated comparison; independent groups with differences in cohort/history and fasting duration.",
                seed=seed, n_boot=n_boot, n_perm=n_perm,
            ))

            # 3. SW18 Y2 vs SW18 S1; estimate S1 - Y2.
            g1 = (df["experimental_condition"] == "SW18") & (df["silvering_stage"] == "Y2")
            g2 = (df["experimental_condition"] == "SW18") & (df["silvering_stage"] == "S1")
            rows.append(independent_comparison(
                df, response, analysis_set,
                comparison="SW18_Y2_vs_SW18_S1_independent",
                group1_mask=g1, group2_mask=g2,
                group1_name="SW18_Y2", group2_name="SW18_S1",
                estimate_note="Stage-associated comparison under SW18; estimate = S1 - Y2.",
                interpretation_note="Stage-associated comparison; interpret cautiously because silvering stage and body size may be confounded.",
                seed=seed, n_boot=n_boot, n_perm=n_perm,
            ))

    out = pd.DataFrame(rows)
    out["p_value_fdr"] = np.nan
    for _, idx in out.groupby(["comparison", "analysis_set"]).groups.items():
        out.loc[idx, "p_value_fdr"] = bh_fdr(out.loc[idx, "p_value"])
    return out


def body_size_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_pop = (df["experimental_condition"] == "SW18") & (df["silvering_stage"].isin(["Y2", "S1"]))

    for response, kind in ENDPOINTS.items():
        analysis_sets = ["main_valid_plus_boundary", "sensitivity_valid_only"] if kind == "active" else ["main"]
        for analysis_set in analysis_sets:
            d0 = analysis_filter(df.loc[base_pop], response, analysis_set).copy()
            d0 = d0[d0["silvering_stage"].isin(["Y2", "S1"])]
            d0["stage_s1"] = (d0["silvering_stage"] == "S1").astype(float)
            for covariate in ["tl_used_cm", "bw_used_g"]:
                d = d0[[response, "stage_s1", "silvering_stage", covariate]].dropna().copy()
                note = ""
                if len(d) < 5 or d["stage_s1"].nunique() < 2:
                    rows.append({
                        "comparison": "SW18_Y2_vs_SW18_S1_body_size_adjusted",
                        "response_variable": response,
                        "analysis_set": analysis_set,
                        "covariate": covariate,
                        "n": len(d),
                        "estimate_stage_S1_vs_Y2": np.nan,
                        "se": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "p_value": np.nan,
                        "r_squared": np.nan,
                        "estimate_covariate": np.nan,
                        "note": "not fitted: insufficient sample size or only one stage represented",
                    })
                    continue
                try:
                    cov = d[covariate].astype(float)
                    X = pd.DataFrame({
                        "intercept": 1.0,
                        "stage_s1": d["stage_s1"].astype(float),
                        covariate: cov - cov.mean(),
                    })
                    model = sm.OLS(d[response].astype(float), X).fit()
                    ci = model.conf_int().loc["stage_s1"]
                    rows.append({
                        "comparison": "SW18_Y2_vs_SW18_S1_body_size_adjusted",
                        "response_variable": response,
                        "analysis_set": analysis_set,
                        "covariate": covariate,
                        "n": int(len(d)),
                        "estimate_stage_S1_vs_Y2": float(model.params["stage_s1"]),
                        "se": float(model.bse["stage_s1"]),
                        "ci_lower": float(ci[0]),
                        "ci_upper": float(ci[1]),
                        "p_value": float(model.pvalues["stage_s1"]),
                        "r_squared": float(model.rsquared),
                        "estimate_covariate": float(model.params[covariate]),
                        "note": "exploratory OLS; covariate centered; TL and BW fitted in separate models",
                    })
                except Exception as exc:
                    rows.append({
                        "comparison": "SW18_Y2_vs_SW18_S1_body_size_adjusted",
                        "response_variable": response,
                        "analysis_set": analysis_set,
                        "covariate": covariate,
                        "n": len(d),
                        "estimate_stage_S1_vs_Y2": np.nan,
                        "se": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "p_value": np.nan,
                        "r_squared": np.nan,
                        "estimate_covariate": np.nan,
                        "note": f"not fitted: {exc}",
                    })
    out = pd.DataFrame(rows)
    out["p_value_fdr"] = np.nan
    for _, idx in out.groupby(["covariate", "analysis_set"]).groups.items():
        out.loc[idx, "p_value_fdr"] = bh_fdr(out.loc[idx, "p_value"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Reanalyse individual endpoint variables from Table S2.")
    parser.add_argument("--s2", default="data/processed/individual_endpoints.csv")
    parser.add_argument("--out-dir", default="outputs/tables")
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--n-perm", type=int, default=10000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    df = prepare_s2(Path(args.s2))

    s4 = make_s4(df, seed=args.seed, n_boot=args.n_boot, n_perm=args.n_perm)
    write_csv_and_xlsx(
        s4,
        out_dir / "Table_S4_statistical_summary.csv",
        out_dir / "Table_S4_statistical_summary.xlsx",
        sheet_name="Table_S4",
    )
    print(f"Wrote {out_dir / 'Table_S4_statistical_summary.csv'}")

    bs = body_size_sensitivity(df)
    write_csv_and_xlsx(
        bs,
        out_dir / "Table_S4_body_size_sensitivity.csv",
        out_dir / "Table_S4_body_size_sensitivity.xlsx",
        sheet_name="BodySizeSensitivity",
    )
    print(f"Wrote {out_dir / 'Table_S4_body_size_sensitivity.csv'}")


if __name__ == "__main__":
    main()
