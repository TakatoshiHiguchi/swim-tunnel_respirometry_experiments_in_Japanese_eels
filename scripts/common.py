from __future__ import annotations

import math
import re
from itertools import combinations
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


def _extreme_statistic_regions(
    perm_intercept: float,
    perm_slope: float,
    observed_intercept: float,
    tol: float = 1e-12,
) -> list[tuple[float, float]]:
    """Return candidate-effect regions counted as at least as extreme.

    For a candidate effect ``delta``, this solves

        |perm_intercept + perm_slope * delta|
            >= |observed_intercept - delta|.

    The result is represented as one or two closed intervals, which may be
    unbounded. This helper is used to invert the exact permutation tests.
    """
    q2 = perm_slope**2 - 1.0
    q1 = 2.0 * (perm_intercept * perm_slope + observed_intercept)
    q0 = perm_intercept**2 - observed_intercept**2

    if math.isclose(q2, 0.0, rel_tol=tol, abs_tol=tol):
        linear_scale = max(1.0, abs(q1), abs(q0))
        if abs(q1) <= tol * linear_scale:
            return [(-np.inf, np.inf)] if q0 >= -tol * linear_scale else []
        root = -q0 / q1
        return [(root, np.inf)] if q1 > 0 else [(-np.inf, root)]

    discriminant = q1**2 - 4.0 * q2 * q0
    disc_scale = max(1.0, q1**2, abs(4.0 * q2 * q0))
    if discriminant < -tol * disc_scale:
        return []
    discriminant = max(0.0, discriminant)
    sqrt_discriminant = math.sqrt(discriminant)
    root_1 = (-q1 - sqrt_discriminant) / (2.0 * q2)
    root_2 = (-q1 + sqrt_discriminant) / (2.0 * q2)
    lower, upper = sorted((root_1, root_2))

    # In the present sign-flip and label-permutation applications, q2 <= 0.
    # The q2 > 0 branch is retained for numerical completeness.
    if q2 < 0:
        return [(lower, upper)]
    return [(-np.inf, lower), (upper, np.inf)]


def _smallest_enclosing_confidence_interval(
    regions: list[tuple[float, float]],
    n_permutations: int,
    alpha: float,
    tol: float = 1e-11,
) -> tuple[float, float]:
    """Return the smallest interval containing the inverted confidence set.

    A candidate effect is retained when its exact two-sided permutation
    p-value is at least ``alpha``. Exact tests are discrete, so the inverted
    confidence set can occasionally be disconnected; this function returns
    the smallest interval containing that set.
    """
    if n_permutations <= 0:
        return (np.nan, np.nan)

    # p >= alpha, expressed as an integer exceedance-count threshold.
    threshold = max(1, math.ceil(alpha * n_permutations - 1e-12))

    initial_count = 0
    raw_events: list[tuple[float, int, int]] = []
    for lower, upper in regions:
        lower_is_inf = np.isneginf(lower)
        upper_is_inf = np.isposinf(upper)
        if lower_is_inf:
            initial_count += 1
        if not lower_is_inf:
            raw_events.append((float(lower), 1, 0))  # interval starts
        if not upper_is_inf:
            raw_events.append((float(upper), 0, 1))  # interval ends

    raw_events.sort(key=lambda item: item[0])
    events: list[list[float | int]] = []
    for position, n_start, n_end in raw_events:
        if (
            not events
            or not math.isclose(
                position, float(events[-1][0]), rel_tol=tol, abs_tol=tol
            )
        ):
            events.append([position, n_start, n_end])
        else:
            events[-1][1] = int(events[-1][1]) + n_start
            events[-1][2] = int(events[-1][2]) + n_end

    current_count = initial_count  # count on the open interval before event
    confidence_lower: float | None = -np.inf if current_count >= threshold else None
    confidence_upper: float | None = None

    for i, (position_raw, starts_raw, ends_raw) in enumerate(events):
        position = float(position_raw)
        starts = int(starts_raw)
        ends = int(ends_raw)

        # Accepted open interval immediately to the left of this event.
        if current_count >= threshold:
            confidence_upper = position

        # Closed intervals that start or end here are both counted at the point.
        at_event_count = current_count + starts
        if at_event_count >= threshold:
            if confidence_lower is None:
                confidence_lower = position
            confidence_upper = position

        # Count on the open interval immediately to the right of this event.
        after_event_count = current_count + starts - ends
        next_position = (
            float(events[i + 1][0]) if i + 1 < len(events) else np.inf
        )
        if after_event_count >= threshold:
            if confidence_lower is None:
                confidence_lower = position
            confidence_upper = next_position

        current_count = after_event_count

    if confidence_lower is None:
        return (np.nan, np.nan)
    if confidence_upper is None:
        confidence_upper = np.inf
    return (float(confidence_lower), float(confidence_upper))


def paired_signflip_ci(
    x: np.ndarray,
    y: np.ndarray,
    confidence_level: float = 0.95,
    max_exact_permutations: int = 200000,
) -> tuple[float, float, str]:
    """Invert the exact paired sign-flip test for the effect x - y.

    The returned limits are the smallest interval containing the exact
    confidence set. If the exact set is unbounded, infinite limits are
    returned and ``status`` records the type of unboundedness.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    differences = x[mask] - y[mask]
    n = differences.size
    if n == 0:
        return (np.nan, np.nan, "not_estimable_no_pairs")

    n_sign_patterns = 2**n
    if n_sign_patterns > max_exact_permutations:
        return (np.nan, np.nan, "not_computed_exact_space_too_large")

    bit_patterns = np.arange(n_sign_patterns, dtype=np.uint64)[:, None]
    bit_positions = np.arange(n, dtype=np.uint64)[None, :]
    signs = np.where((bit_patterns >> bit_positions) & 1, 1.0, -1.0)

    estimate = float(np.mean(differences))
    perm_intercepts = (signs @ differences) / n
    perm_slopes = -np.mean(signs, axis=1)

    regions: list[tuple[float, float]] = []
    for intercept, slope in zip(perm_intercepts, perm_slopes):
        regions.extend(
            _extreme_statistic_regions(
                float(intercept), float(slope), estimate
            )
        )

    alpha = 1.0 - confidence_level
    lower, upper = _smallest_enclosing_confidence_interval(
        regions, n_sign_patterns, alpha
    )
    if np.isneginf(lower) and np.isposinf(upper):
        status = "unbounded_both_sides"
    elif np.isneginf(lower):
        status = "unbounded_lower"
    elif np.isposinf(upper):
        status = "unbounded_upper"
    elif np.isfinite(lower) and np.isfinite(upper):
        status = "finite"
    else:
        status = "not_estimable"
    return (lower, upper, status)


def independent_permutation_ci(
    x: np.ndarray,
    y: np.ndarray,
    confidence_level: float = 0.95,
    max_exact_permutations: int = 200000,
) -> tuple[float, float, str]:
    """Invert the exact two-sample label-permutation test for y - x.

    This confidence set uses an additive location-shift formulation: under a
    candidate effect ``delta``, ``delta`` is subtracted from y before labels
    are permuted. The returned limits are the smallest interval containing
    the exact confidence set.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    nx = x.size
    ny = y.size
    if nx == 0 or ny == 0:
        return (np.nan, np.nan, "not_estimable_empty_group")

    n_total = nx + ny
    n_allocations = math.comb(n_total, nx)
    if n_allocations > max_exact_permutations:
        return (np.nan, np.nan, "not_computed_exact_space_too_large")

    estimate = float(np.mean(y) - np.mean(x))
    pooled_intercepts = np.concatenate([x, y])
    # Under candidate delta, y is shifted to y - delta.
    pooled_slopes = np.concatenate([np.zeros(nx), -np.ones(ny)])

    regions: list[tuple[float, float]] = []
    for group_x_indices in combinations(range(n_total), nx):
        in_group_x = np.zeros(n_total, dtype=bool)
        in_group_x[list(group_x_indices)] = True
        in_group_y = ~in_group_x

        perm_intercept = float(
            np.mean(pooled_intercepts[in_group_y])
            - np.mean(pooled_intercepts[in_group_x])
        )
        perm_slope = float(
            np.mean(pooled_slopes[in_group_y])
            - np.mean(pooled_slopes[in_group_x])
        )
        regions.extend(
            _extreme_statistic_regions(
                perm_intercept, perm_slope, estimate
            )
        )

    alpha = 1.0 - confidence_level
    lower, upper = _smallest_enclosing_confidence_interval(
        regions, n_allocations, alpha
    )
    if np.isneginf(lower) and np.isposinf(upper):
        status = "unbounded_both_sides"
    elif np.isneginf(lower):
        status = "unbounded_lower"
    elif np.isposinf(upper):
        status = "unbounded_upper"
    elif np.isfinite(lower) and np.isfinite(upper):
        status = "finite"
    else:
        status = "not_estimable"
    return (lower, upper, status)


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
    """Two-sided permutation test for mean difference y - x.

    Exact enumeration is used whenever the number of distinct label
    allocations is at most 200,000; otherwise a Monte Carlo test is used.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan

    obs = abs(np.mean(y) - np.mean(x))
    pooled = np.concatenate([x, y])
    nx = x.size
    n_total = pooled.size
    n_allocations = math.comb(n_total, nx)

    if n_allocations <= 200000:
        stats = np.empty(n_allocations, dtype=float)
        for i, group_x_indices in enumerate(combinations(range(n_total), nx)):
            in_group_x = np.zeros(n_total, dtype=bool)
            in_group_x[list(group_x_indices)] = True
            stats[i] = abs(
                np.mean(pooled[~in_group_x]) - np.mean(pooled[in_group_x])
            )
        return float(np.mean(stats >= obs - 1e-15))

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
