from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "null", "missing", "-", "—"}


def norm_col(name: object) -> str:
    """Normalize a column name while preserving meaning."""
    s = str(name).strip()
    s = s.replace("\u2212", "-")
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [norm_col(c) for c in out.columns]
    return out


def clean_status(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def clean_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in MISSING_STRINGS:
        return ""
    return s


def read_table(path: Path, sheet_candidates: Iterable[str] | None = None) -> pd.DataFrame:
    """Read CSV or Excel, selecting a named sheet when possible."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".csv":
        return normalize_columns(pd.read_csv(path))

    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        xls = pd.ExcelFile(path)
        candidates = [s.lower() for s in (sheet_candidates or [])]
        chosen = None
        for sheet in xls.sheet_names:
            if sheet.lower() in candidates:
                chosen = sheet
                break
        if chosen is None:
            for sheet in xls.sheet_names:
                ns = norm_col(sheet)
                if ns in {norm_col(c) for c in (sheet_candidates or [])}:
                    chosen = sheet
                    break
        if chosen is None:
            chosen = xls.sheet_names[0]
        return normalize_columns(pd.read_excel(path, sheet_name=chosen))

    raise ValueError(f"Unsupported file type: {path.suffix}")


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def ensure_columns(df: pd.DataFrame, required: Iterable[str], table_name: str = "table") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def mean_sd_median(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def bootstrap_ci_paired(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 20260625) -> tuple[float, float]:
    """Percentile bootstrap CI for mean paired difference x - y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    d = x[mask] - y[mask]
    if d.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_ci_independent(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 20260625) -> tuple[float, float]:
    """Percentile bootstrap CI for mean difference y - x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bx = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    by = y[rng.integers(0, y.size, size=(n_boot, y.size))].mean(axis=1)
    boot = by - bx
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def paired_permutation_p(x: np.ndarray, y: np.ndarray, n_perm: int = 10000, seed: int = 20260625) -> float:
    """Two-sided sign-flip permutation test for mean paired difference x - y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    d = x[mask] - y[mask]
    n = d.size
    if n == 0:
        return np.nan
    obs = abs(np.mean(d))

    # Exact enumeration for small n.
    if n <= 20 and 2**n <= 200000:
        stats = []
        for bits in range(2**n):
            signs = np.array([1 if (bits >> i) & 1 else -1 for i in range(n)], dtype=float)
            stats.append(abs(np.mean(signs * d)))
        stats = np.asarray(stats)
        return float(np.mean(stats >= obs - 1e-15))

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    stats = np.abs((signs * d).mean(axis=1))
    return float((np.sum(stats >= obs - 1e-15) + 1) / (n_perm + 1))


def independent_permutation_p(x: np.ndarray, y: np.ndarray, n_perm: int = 10000, seed: int = 20260625) -> float:
    """Two-sided permutation test for mean difference y - x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    obs = abs(np.mean(y) - np.mean(x))
    pooled = np.concatenate([x, y])
    nx = x.size
    rng = np.random.default_rng(seed)
    stats = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        stats[i] = abs(np.mean(perm[nx:]) - np.mean(perm[:nx]))
    return float((np.sum(stats >= obs - 1e-15) + 1) / (n_perm + 1))


def bh_fdr(pvalues: Iterable[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values; NaNs remain NaN."""
    p = np.asarray(list(pvalues), dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    vals = p[mask]
    m = vals.size
    if m == 0:
        return out.tolist()
    order = np.argsort(vals)
    ranked = vals[order]
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    tmp = np.empty_like(vals)
    tmp[order] = adj
    out[mask] = tmp
    return out.tolist()


def write_csv_and_xlsx(df: pd.DataFrame, csv_path: Path, xlsx_path: Optional[Path] = None, sheet_name: str = "Sheet1") -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if xlsx_path is not None:
        xlsx_path = Path(xlsx_path)
        xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
