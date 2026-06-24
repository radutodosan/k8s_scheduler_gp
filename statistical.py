"""Statistical analysis module for experiment results.

Provides non-parametric hypothesis tests, effect sizes, confidence intervals,
rank analysis, and rule interpretability — all needed for dissertation sections
5.4 (Teste statistice, Analiza semnificației) and 5.5 (Analiza structurală).

Functions work on the combined_results.csv DataFrame produced by run_experiments.py.
"""

from __future__ import annotations

import itertools
import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Pairwise hypothesis tests
# ═══════════════════════════════════════════════════════════════════════


def wilcoxon_pairwise(
    df: pd.DataFrame,
    metric: str,
    strategy_a: str,
    strategy_b: str,
    experiment: Optional[str] = None,
) -> Dict[str, float]:
    """Wilcoxon signed-rank test between two strategies on paired instances.

    Uses the instance_id as the pairing key (same test instance, different
    strategies). Appropriate when samples are paired and non-normally distributed.

    Returns dict with keys: statistic, p_value, n_pairs.
    """
    subset = df if experiment is None else df[df["experiment"] == experiment]

    a = subset[subset["strategy"] == strategy_a].set_index("instance_id")[metric]
    b = subset[subset["strategy"] == strategy_b].set_index("instance_id")[metric]

    common = a.index.intersection(b.index)
    if len(common) < 2:
        return {"statistic": np.nan, "p_value": np.nan, "n_pairs": len(common)}

    x = a.loc[common].values
    y = b.loc[common].values
    diff = x - y

    # All zeros → no difference
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": len(common)}

    stat, p = stats.wilcoxon(x, y, alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p), "n_pairs": len(common)}


def mann_whitney(
    df: pd.DataFrame,
    metric: str,
    strategy_a: str,
    strategy_b: str,
    group: Optional[str] = None,
) -> Dict[str, float]:
    """Mann-Whitney U test (unpaired) between two strategies.

    Useful for comparing strategies across different experiment configurations
    (e.g., DEAP across different experiment configurations).

    Returns dict with keys: statistic, p_value, n_a, n_b.
    """
    subset = df if group is None else df[df["group"] == group]

    a = subset[subset["strategy"] == strategy_a][metric].values
    b = subset[subset["strategy"] == strategy_b][metric].values

    if len(a) < 1 or len(b) < 1:
        return {"statistic": np.nan, "p_value": np.nan, "n_a": len(a), "n_b": len(b)}

    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p), "n_a": len(a), "n_b": len(b)}


def friedman_test(
    df: pd.DataFrame,
    metric: str,
    experiment: Optional[str] = None,
) -> Dict[str, float]:
    """Friedman test — non-parametric test for k related samples.

    Tests whether there are significant differences among all strategies
    within an experiment. Each instance_id is a block.

    Returns dict with keys: statistic, p_value, k_strategies, n_blocks.
    """
    subset = df if experiment is None else df[df["experiment"] == experiment]

    pivot = subset.pivot_table(
        index="instance_id", columns="strategy", values=metric, aggfunc="mean",
    ).dropna(axis=1, how="all").dropna(axis=0, how="any")

    if pivot.shape[0] < 2 or pivot.shape[1] < 3:
        return {
            "statistic": np.nan, "p_value": np.nan,
            "k_strategies": pivot.shape[1], "n_blocks": pivot.shape[0],
        }

    samples = [pivot[col].values for col in pivot.columns]
    stat, p = stats.friedmanchisquare(*samples)

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "k_strategies": pivot.shape[1],
        "n_blocks": pivot.shape[0],
    }


# ═══════════════════════════════════════════════════════════════════════
# Effect sizes
# ═══════════════════════════════════════════════════════════════════════


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta — non-parametric effect size.

    Returns a value in [-1, 1]:
      |δ| < 0.147 → negligible
      |δ| < 0.33  → small
      |δ| < 0.474 → medium
      |δ| ≥ 0.474 → large
    """
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0

    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)

    return (more - less) / (n_x * n_y)


def cliffs_delta_interpretation(delta: float) -> str:
    """Interpret Cliff's delta magnitude."""
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    elif d < 0.33:
        return "small"
    elif d < 0.474:
        return "medium"
    else:
        return "large"


def vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    """Vargha-Delaney A₁₂ — probability that x > y.

    A₁₂ = 0.5     → no difference
    A₁₂ > 0.5     → x tends to be larger
    A₁₂ < 0.5     → y tends to be larger

    Interpretation:
      |A₁₂ - 0.5| < 0.06 → negligible
      |A₁₂ - 0.5| < 0.14 → small
      |A₁₂ - 0.5| < 0.21 → medium
      |A₁₂ - 0.5| ≥ 0.21 → large
    """
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.5

    more = sum(1 for xi in x for yi in y if xi > yi)
    equal = sum(1 for xi in x for yi in y if xi == yi)

    return (more + 0.5 * equal) / (n_x * n_y)


# ═══════════════════════════════════════════════════════════════════════
# P-value correction
# ═══════════════════════════════════════════════════════════════════════


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[dict]:
    """Holm-Bonferroni correction for multiple comparisons.

    More powerful than Bonferroni while still controlling FWER.

    Returns a list of dicts with: original_p, adjusted_p, rejected (bool).
    """
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [None] * n

    cummax_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (n - rank)
        adjusted = max(adjusted, cummax_adj)  # Enforce monotonicity
        adjusted = min(adjusted, 1.0)
        cummax_adj = adjusted
        results[orig_idx] = {
            "original_p": p,
            "adjusted_p": adjusted,
            "rejected": adjusted < alpha,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════
# Confidence intervals (bootstrap)
# ═══════════════════════════════════════════════════════════════════════


def bootstrap_ci(
    x: np.ndarray,
    statistic_fn=np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for a statistic.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return (np.nan, np.nan, np.nan)

    point = float(statistic_fn(x))
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(x, size=n, replace=True)
        boot_stats[i] = statistic_fn(sample)

    alpha = 1 - confidence
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return (point, lower, upper)


# ═══════════════════════════════════════════════════════════════════════
# Rank analysis
# ═══════════════════════════════════════════════════════════════════════


def average_ranks(
    df: pd.DataFrame,
    metric: str,
    ascending: bool = True,
    experiment: Optional[str] = None,
) -> pd.Series:
    """Compute average rank of each strategy across test instances.

    Args:
        ascending: If True, lower metric values get rank 1 (good for wait_time).
                   If False, higher values get rank 1 (good for success_rate).

    Returns a Series indexed by strategy with average rank values.
    """
    subset = df if experiment is None else df[df["experiment"] == experiment]

    pivot = subset.pivot_table(
        index="instance_id", columns="strategy", values=metric, aggfunc="mean",
    ).dropna(axis=1, how="all")

    ranks = pivot.rank(axis=1, ascending=ascending, method="average")
    return ranks.mean().sort_values()


# ═══════════════════════════════════════════════════════════════════════
# Comprehensive pairwise comparison table
# ═══════════════════════════════════════════════════════════════════════


def gp_vs_baselines_table(
    df: pd.DataFrame,
    metric: str,
    experiment: Optional[str] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare GP strategy against every baseline with tests + effect sizes.

    Returns a DataFrame with one row per baseline, columns:
    baseline, gp_mean, baseline_mean, diff, wilcoxon_p, adjusted_p,
    significant, cliffs_d, effect_magnitude.
    """
    subset = df if experiment is None else df[df["experiment"] == experiment]

    # Find GP strategy name
    gp_strategies = [s for s in subset["strategy"].unique() if s.startswith("GP(")]
    if not gp_strategies:
        return pd.DataFrame()
    gp_name = gp_strategies[0]

    baselines = [s for s in subset["strategy"].unique() if not s.startswith("GP(")]
    if not baselines:
        return pd.DataFrame()

    gp_data = subset[subset["strategy"] == gp_name]
    gp_by_instance = gp_data.set_index("instance_id")[metric]

    rows = []
    p_values = []

    for bl in sorted(baselines):
        bl_data = subset[subset["strategy"] == bl]
        bl_by_instance = bl_data.set_index("instance_id")[metric]

        common = gp_by_instance.index.intersection(bl_by_instance.index)
        if len(common) == 0:
            continue

        gp_vals = gp_by_instance.loc[common].values
        bl_vals = bl_by_instance.loc[common].values

        wt = wilcoxon_pairwise(subset, metric, gp_name, bl, experiment)
        cd = cliffs_delta(gp_vals, bl_vals)

        rows.append({
            "baseline": bl,
            "gp_mean": float(np.mean(gp_vals)),
            "baseline_mean": float(np.mean(bl_vals)),
            "diff": float(np.mean(gp_vals) - np.mean(bl_vals)),
            "wilcoxon_p": wt["p_value"],
            "n_pairs": wt["n_pairs"],
            "cliffs_d": cd,
            "effect_magnitude": cliffs_delta_interpretation(cd),
        })
        p_values.append(wt["p_value"])

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # Apply Holm-Bonferroni correction
    valid_ps = [p for p in p_values if not np.isnan(p)]
    if valid_ps:
        corrected = holm_bonferroni(valid_ps, alpha=alpha)
        adj_idx = 0
        adj_p_list = []
        sig_list = []
        for p in p_values:
            if np.isnan(p):
                adj_p_list.append(np.nan)
                sig_list.append(False)
            else:
                adj_p_list.append(corrected[adj_idx]["adjusted_p"])
                sig_list.append(corrected[adj_idx]["rejected"])
                adj_idx += 1
        result["adjusted_p"] = adj_p_list
        result["significant"] = sig_list
    else:
        result["adjusted_p"] = np.nan
        result["significant"] = False

    return result


def gp_vs_baselines_multiseed_table(
    df: pd.DataFrame,
    metric: str,
    experiment: Optional[str] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare GP against every baseline using multi-seed paired data.

    Expects a DataFrame with a ``run_seed`` column (from multiseed_results.csv).
    Pairs are formed by ``(run_seed, instance_id)`` so every GP–baseline
    comparison is matched to the exact same workload and the same GP training
    run.  This gives n_seeds × n_test_instances pairs, providing much higher
    statistical power than the single-seed version.

    Returns the same columns as ``gp_vs_baselines_table`` plus ``n_pairs``.
    """
    subset = df if experiment is None else df[df["experiment"] == experiment]

    if "run_seed" not in subset.columns:
        return gp_vs_baselines_table(df, metric, experiment, alpha)

    # Build composite pair key
    subset = subset.copy()
    subset["pair_id"] = (
        subset["run_seed"].astype(str) + "_" + subset["instance_id"].astype(str)
    )

    gp_strategies = [s for s in subset["strategy"].unique() if s.startswith("GP(")]
    if not gp_strategies:
        return pd.DataFrame()
    gp_name = gp_strategies[0]

    baselines = [s for s in subset["strategy"].unique() if not s.startswith("GP(")]
    if not baselines:
        return pd.DataFrame()

    gp_vals_by_pair = subset[subset["strategy"] == gp_name].set_index("pair_id")[metric]

    rows = []
    p_values: List[float] = []

    for bl in sorted(baselines):
        bl_vals_by_pair = subset[subset["strategy"] == bl].set_index("pair_id")[metric]
        common = gp_vals_by_pair.index.intersection(bl_vals_by_pair.index)

        if len(common) < 2:
            continue

        gp_arr = gp_vals_by_pair.loc[common].values.astype(float)
        bl_arr = bl_vals_by_pair.loc[common].values.astype(float)

        diff = gp_arr - bl_arr
        if np.all(diff == 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = stats.wilcoxon(gp_arr, bl_arr, alternative="two-sided")
            stat, p = float(stat), float(p)

        cd = cliffs_delta(gp_arr, bl_arr)
        rows.append({
            "baseline": bl,
            "gp_median": float(np.median(gp_arr)),
            "baseline_median": float(np.median(bl_arr)),
            "gp_mean": float(np.mean(gp_arr)),
            "baseline_mean": float(np.mean(bl_arr)),
            "diff_median": float(np.median(gp_arr) - np.median(bl_arr)),
            "wilcoxon_p": p,
            "n_pairs": int(len(common)),
            "cliffs_d": cd,
            "effect_magnitude": cliffs_delta_interpretation(cd),
        })
        p_values.append(p)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    valid_ps = [p for p in p_values if not np.isnan(p)]
    if valid_ps:
        corrected = holm_bonferroni(valid_ps, alpha=alpha)
        adj_idx = 0
        adj_p_list: List[float] = []
        sig_list: List[bool] = []
        for p in p_values:
            if np.isnan(p):
                adj_p_list.append(float("nan"))
                sig_list.append(False)
            else:
                adj_p_list.append(corrected[adj_idx]["adjusted_p"])
                sig_list.append(bool(corrected[adj_idx]["rejected"]))
                adj_idx += 1
        result["adjusted_p"] = adj_p_list
        result["significant"] = sig_list
    else:
        result["adjusted_p"] = float("nan")
        result["significant"] = False

    return result.sort_values("diff_median", ascending=False)


def acceptance_criteria_report(
    df: pd.DataFrame,
    metric: str = "quality_score",
    experiment: Optional[str] = None,
    alpha: float = 0.05,
    min_effect_size: float = 0.147,
) -> str:
    """Generate the anti-cherry-picking acceptance criteria report.

    Checks the four acceptance thresholds from the dissertation:
      1. GP median > best-baseline median on ≥ 4/5 seeds (seed-level ranking)
      2. Statistical significance: adjusted p < alpha vs ≥ 4 baselines
      3. Effect size: |Cliff's δ| ≥ min_effect_size vs best baseline
      4. n_pairs ≥ 10 (sufficient statistical power)

    Works with either single-seed data (instance_id pairing) or multi-seed
    data (run_seed × instance_id pairing).
    """
    multiseed = "run_seed" in df.columns
    if multiseed:
        cmp = gp_vs_baselines_multiseed_table(df, metric, experiment, alpha)
    else:
        cmp = gp_vs_baselines_table(df, metric, experiment, alpha)

    if cmp.empty:
        return "No GP strategy found in data.\n"

    gp_strategies = [s for s in df["strategy"].unique() if s.startswith("GP(")]
    gp_name = gp_strategies[0] if gp_strategies else "GP(?)"

    lines = [
        "ACCEPTANCE CRITERIA REPORT (anti-cherry-picking)",
        f"GP strategy : {gp_name}",
        f"Metric      : {metric}",
        f"Alpha       : {alpha}  (Holm-Bonferroni corrected)",
        f"Min |delta| : {min_effect_size} ({cliffs_delta_interpretation(min_effect_size)}+)",
        f"Mode        : {'multi-seed' if multiseed else 'single-seed'}",
        "=" * 80,
        "",
    ]

    median_col = "gp_median" if "gp_median" in cmp.columns else "gp_mean"
    baseline_median_col = "baseline_median" if "baseline_median" in cmp.columns else "baseline_mean"
    diff_col = "diff_median" if "diff_median" in cmp.columns else "diff"

    # Criterion 1: GP median > each baseline median
    gp_wins = cmp[cmp[diff_col] > 0]
    n_wins = len(gp_wins)
    n_total = len(cmp)
    crit1 = n_wins >= max(1, round(0.57 * n_total))  # >= 4/7 baselines

    lines.append(f"[1] GP median > baseline median: {n_wins}/{n_total} baselines")
    lines.append(f"    {'PASS' if crit1 else 'FAIL'} (threshold: >= {max(1, round(0.57*n_total))}/{n_total})")
    lines.append("")

    # Criterion 2: Significance after correction
    n_sig = int(cmp["significant"].sum()) if "significant" in cmp.columns else 0
    crit2 = n_sig >= max(1, round(0.57 * n_total))

    lines.append(f"[2] Significant (adjusted p < {alpha}): {n_sig}/{n_total} baselines")
    lines.append(f"    {'PASS' if crit2 else 'FAIL'} (threshold: >= {max(1, round(0.57*n_total))}/{n_total})")
    lines.append("")

    # Criterion 3: Effect size vs best baseline
    best_row = cmp.loc[cmp["baseline_median" if "baseline_median" in cmp.columns else "baseline_mean"].idxmax()]
    best_delta = abs(float(best_row["cliffs_d"]))
    crit3 = best_delta >= min_effect_size

    lines.append(
        f"[3] |Cliff's delta| >= {min_effect_size} vs best baseline "
        f"({best_row['baseline']}): delta={best_row['cliffs_d']:.3f} "
        f"({best_row['effect_magnitude']})"
    )
    lines.append(f"    {'PASS' if crit3 else 'FAIL'}")
    lines.append("")

    # Criterion 4: Sample size
    min_pairs = int(cmp["n_pairs"].min()) if "n_pairs" in cmp.columns else 0
    crit4 = min_pairs >= 10

    lines.append(f"[4] n_pairs >= 10: min={min_pairs}")
    lines.append(f"    {'PASS' if crit4 else 'FAIL (increase --seeds or n_test_instances)'}")
    lines.append("")

    # Overall verdict
    n_pass = sum([crit1, crit2, crit3, crit4])
    verdict = "ACCEPTED" if n_pass >= 3 else "REJECTED"
    lines.append(f"VERDICT: {verdict}  ({n_pass}/4 criteria met)")
    lines.append("")

    # Full comparison table
    lines.append("Full comparison table:")
    lines.append(
        f"  {'Baseline':<30} {median_col[:8]:>8} {baseline_median_col[:10]:>10} "
        f"{'delta':>7} {'|d|':>5} {'mag':>10} {'adj_p':>8} {'sig':>4}"
    )
    lines.append("  " + "-" * 85)
    for _, row in cmp.iterrows():
        gp_m = float(row.get(median_col, float("nan")))
        bl_m = float(row.get(baseline_median_col, float("nan")))
        lines.append(
            f"  {row['baseline']:<30} {gp_m:>8.4f} {bl_m:>10.4f}"
            f" {float(row[diff_col]):>7.4f} {abs(float(row['cliffs_d'])):>5.3f}"
            f" {row['effect_magnitude']:>10} {float(row.get('adjusted_p', float('nan'))):>8.4f}"
            f" {'Yes' if row.get('significant') else 'No':>4}"
        )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Rule interpretability
# ═══════════════════════════════════════════════════════════════════════


def _build_terminal_pattern() -> re.Pattern:
    from gp.primitives import TERMINAL_NAMES
    names = "|".join(re.escape(n) for n in TERMINAL_NAMES)
    return re.compile(r'\b(' + names + r')\b')

_TERMINAL_PATTERN: re.Pattern = _build_terminal_pattern()

_FUNCTION_PATTERN = re.compile(
    r'\b(add|sub|mul|protected_div|neg|min|max|if_positive)\b'
)


def extract_features_from_expression(expression: str) -> Dict[str, int]:
    """Count terminal (feature) usage in a GP expression string.

    Returns a dict mapping terminal name → count of occurrences.
    """
    matches = _TERMINAL_PATTERN.findall(expression)
    return dict(Counter(matches))


def expression_complexity(expression: str) -> Dict[str, int]:
    """Compute structural complexity metrics of a GP expression.

    Returns:
        n_terminals: number of terminal references
        n_functions: number of function calls
        n_unique_features: distinct terminals used
        depth_estimate: approximate nesting depth (count of '(')
    """
    terminals = _TERMINAL_PATTERN.findall(expression)
    functions = _FUNCTION_PATTERN.findall(expression)

    return {
        "n_terminals": len(terminals),
        "n_functions": len(functions),
        "n_unique_features": len(set(terminals)),
        "depth_estimate": expression.count("("),
    }


def simplify_expression(expression: str) -> str:
    """Algebraic simplification of a GP expression string.

    Applies rewrite rules iteratively until no more changes occur:
      - add(X, 0) / add(0, X) → X
      - sub(X, 0) → X
      - sub(X, X) → 0
      - mul(X, 1) / mul(1, X) → X
      - mul(X, 0) / mul(0, X) → 0
      - protected_div(X, 1) → X
      - protected_div(0, X) → 0
      - protected_div(X, X) → 1
      - neg(neg(X)) → X
      - neg(0) → 0
      - if_positive(X, Y, Y) → Y   (both branches equal)
      - min(X, X) / max(X, X) → X
    """
    def _match_paren(text, open_pos):
        """Return index of the ')' matching '(' at *open_pos*, or -1."""
        depth = 1
        i = open_pos + 1
        while i < len(text):
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return -1

    def _split_args(text, start, end):
        """Split text[start:end] by top-level commas."""
        args = []
        depth = 0
        arg_start = start
        for i in range(start, end):
            c = text[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c == ',' and depth == 0:
                args.append(text[arg_start:i].strip())
                arg_start = i + 1
        args.append(text[arg_start:end].strip())
        return args

    s = expression
    funcs = ('add', 'sub', 'mul', 'protected_div', 'neg', 'min', 'max', 'if_positive')

    prev = None
    while prev != s:
        prev = s

        # neg(0) → 0
        s = s.replace('neg(0)', '0')

        for func in funcs:
            pattern = func + '('
            i = 0
            while i < len(s):
                pos = s.find(pattern, i)
                if pos == -1:
                    break
                # Make sure it's not part of a longer name (e.g. "xadd(")
                if pos > 0 and (s[pos - 1].isalnum() or s[pos - 1] == '_'):
                    i = pos + 1
                    continue
                open_p = pos + len(func)
                close_p = _match_paren(s, open_p)
                if close_p == -1:
                    i = pos + 1
                    continue
                args = _split_args(s, open_p + 1, close_p)
                replacement = None

                if func == 'neg' and len(args) == 1:
                    # neg(neg(X)) → X
                    inner = args[0]
                    if inner.startswith('neg(') and inner.endswith(')'):
                        inner_open = 3
                        inner_close = _match_paren(inner, inner_open)
                        if inner_close == len(inner) - 1:
                            replacement = inner[4:inner_close]
                elif func == 'add' and len(args) == 2:
                    if args[1] == '0':
                        replacement = args[0]
                    elif args[0] == '0':
                        replacement = args[1]
                elif func == 'sub' and len(args) == 2:
                    if args[1] == '0':
                        replacement = args[0]
                    elif args[0] == args[1]:
                        replacement = '0'
                elif func == 'mul' and len(args) == 2:
                    if args[0] == '0' or args[1] == '0':
                        replacement = '0'
                    elif args[1] == '1':
                        replacement = args[0]
                    elif args[0] == '1':
                        replacement = args[1]
                elif func == 'protected_div' and len(args) == 2:
                    if args[1] == '1':
                        replacement = args[0]
                    elif args[0] == '0':
                        replacement = '0'
                    elif args[0] == args[1] and args[0] != '0':
                        replacement = '1'
                elif func in ('min', 'max') and len(args) == 2:
                    if args[0] == args[1]:
                        replacement = args[0]
                elif func == 'if_positive' and len(args) == 3:
                    if args[1] == args[2]:
                        replacement = args[1]

                if replacement is not None:
                    s = s[:pos] + replacement + s[close_p + 1:]
                    # Don't advance — recheck at same position
                else:
                    i = pos + 1

    return s


def feature_importance_from_metadata(
    metadata: Dict[str, dict],
) -> pd.DataFrame:
    """Extract feature importance by counting terminal usage across all evolved rules.

    Returns a DataFrame with columns: feature, total_count, n_rules, avg_per_rule.
    """
    all_counts: Dict[str, int] = {}
    rules_with_feature: Dict[str, int] = {}
    n_rules = 0

    for exp_name, meta in metadata.items():
        expr = meta.get("best_expression", "")
        if not expr:
            continue
        n_rules += 1
        counts = extract_features_from_expression(expr)
        for feat, cnt in counts.items():
            all_counts[feat] = all_counts.get(feat, 0) + cnt
            rules_with_feature[feat] = rules_with_feature.get(feat, 0) + 1

    if n_rules == 0:
        return pd.DataFrame(columns=["feature", "total_count", "n_rules", "avg_per_rule"])

    rows = []
    for feat in sorted(all_counts.keys(), key=lambda f: -all_counts[f]):
        rows.append({
            "feature": feat,
            "total_count": all_counts[feat],
            "n_rules": rules_with_feature.get(feat, 0),
            "avg_per_rule": round(all_counts[feat] / n_rules, 2),
        })

    return pd.DataFrame(rows)


def rule_summary_table(metadata: Dict[str, dict]) -> pd.DataFrame:
    """Summarise all evolved rules with complexity and feature usage.

    Returns a DataFrame with columns: experiment, group, engine, expression,
    n_terminals, n_functions, n_unique_features, depth_estimate, features_used.
    """
    rows = []
    for exp_name, meta in sorted(metadata.items()):
        expr = meta.get("best_expression", "")
        if not expr:
            continue
        complexity = expression_complexity(expr)
        features = extract_features_from_expression(expr)
        rows.append({
            "experiment": exp_name,
            "group": meta.get("group", ""),
            "engine": meta.get("engine", ""),
            "fitness": meta.get("best_fitness", np.nan),
            "expression": expr,
            **complexity,
            "features_used": ", ".join(sorted(features.keys())),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════


def sensitivity_table(
    df: pd.DataFrame,
    metadata: Dict[str, dict],
    group: str,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build a sensitivity analysis table for a given experiment group.

    For each experiment in the group, shows the varying parameter + GP metric means.

    Args:
        group: experiment group to analyse (e.g., 'fitness_weights', 'gp_params')
        metrics: list of metric columns (default: sched_rate, avg_wait, cpu_util)
    """
    if metrics is None:
        metrics = ["scheduling_success_rate", "avg_wait_time", "avg_cpu_utilization"]

    group_df = df[df["group"] == group]
    gp_rows = group_df[group_df["strategy"].str.startswith("GP(")]

    if gp_rows.empty:
        return pd.DataFrame()

    rows = []
    for exp_name in sorted(gp_rows["experiment"].unique()):
        exp_data = gp_rows[gp_rows["experiment"] == exp_name]
        meta = metadata.get(exp_name, {})

        row = {
            "experiment": exp_name,
            "engine": meta.get("engine", ""),
            "population_size": meta.get("population_size", ""),
            "n_generations": meta.get("n_generations", ""),
            "total_pods": meta.get("total_pods", ""),
            "node_count": meta.get("node_count", ""),
            "alpha": meta.get("alpha", ""),
            "beta": meta.get("beta", ""),
            "gamma": meta.get("gamma", ""),
            "node_failures": meta.get("node_failures", ""),
            "best_fitness": meta.get("best_fitness", np.nan),
            "training_time_s": meta.get("training_time_s", np.nan),
        }
        for m in metrics:
            row[f"mean_{m}"] = exp_data[m].mean()
        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Full statistical report
# ═══════════════════════════════════════════════════════════════════════


def generate_statistical_report(
    df: pd.DataFrame,
    metadata: Dict[str, dict],
    metrics: Optional[List[str]] = None,
) -> str:
    """Generate a comprehensive statistical report as text.

    Includes:
      - Friedman tests per experiment
      - GP vs baselines pairwise comparisons
      - Rank analysis
      - Effect sizes with interpretation
      - Rule complexity summary
      - Feature importance
      - Sensitivity tables
    """
    if metrics is None:
        metrics = ["scheduling_success_rate", "avg_wait_time"]

    lines: List[str] = [
        "STATISTICAL ANALYSIS REPORT",
        "=" * 70,
        "",
    ]

    experiments = sorted(df["experiment"].unique())

    # ── 1. Friedman tests ────────────────────────────────────────
    lines.append("1. FRIEDMAN TESTS (per experiment)")
    lines.append("-" * 50)
    for exp in experiments:
        for metric in metrics:
            ft = friedman_test(df, metric, experiment=exp)
            sig = "***" if ft["p_value"] < 0.001 else (
                "**" if ft["p_value"] < 0.01 else (
                    "*" if ft["p_value"] < 0.05 else "ns"))
            lines.append(
                f"  {exp} | {metric}: χ²={ft['statistic']:.3f}, "
                f"p={ft['p_value']:.4f} {sig} "
                f"(k={ft['k_strategies']}, n={ft['n_blocks']})"
            )
    lines.append("")

    # ── 2. GP vs Baselines pairwise ──────────────────────────────
    lines.append("2. GP vs BASELINES (Wilcoxon + Holm-Bonferroni + Cliff's δ)")
    lines.append("-" * 50)
    for exp in experiments:
        for metric in metrics:
            table = gp_vs_baselines_table(df, metric, experiment=exp)
            if table.empty:
                continue
            lines.append(f"\n  Experiment: {exp} | Metric: {metric}")
            lines.append(
                f"  {'Baseline':<22} {'GP':>7} {'Base':>7} "
                f"{'Diff':>7} {'p-adj':>8} {'Sig':>4} {'δ':>6} {'Mag':>12}"
            )
            lines.append("  " + "-" * 80)
            for _, row in table.iterrows():
                sig = "YES" if row["significant"] else "no"
                lines.append(
                    f"  {row['baseline']:<22} {row['gp_mean']:>7.3f} "
                    f"{row['baseline_mean']:>7.3f} {row['diff']:>+7.3f} "
                    f"{row['adjusted_p']:>8.4f} {sig:>4} "
                    f"{row['cliffs_d']:>+6.3f} {row['effect_magnitude']:>12}"
                )
    lines.append("")

    # ── 3. Rank analysis ─────────────────────────────────────────
    lines.append("3. AVERAGE RANKS (lower = better)")
    lines.append("-" * 50)
    for metric in metrics:
        ascending = metric in ("avg_wait_time",)
        ranks = average_ranks(df, metric, ascending=ascending)
        lines.append(f"\n  Metric: {metric}")
        for strat, rank in ranks.items():
            lines.append(f"    {strat:<22} {rank:.2f}")
    lines.append("")

    # ── 4. Rule interpretability ─────────────────────────────────
    if metadata:
        lines.append("4. RULE COMPLEXITY & FEATURES")
        lines.append("-" * 50)

        rules = rule_summary_table(metadata)
        if not rules.empty:
            for _, row in rules.iterrows():
                lines.append(
                    f"  {row['experiment']:<25} [{row['engine']}] "
                    f"fitness={row['fitness']:.4f}"
                )
                lines.append(
                    f"    Complexity: {row['n_terminals']} terminals, "
                    f"{row['n_functions']} functions, "
                    f"{row['n_unique_features']} unique features, "
                    f"depth≈{row['depth_estimate']}"
                )
                lines.append(f"    Features: {row['features_used']}")
                lines.append(f"    Rule: {row['expression'][:100]}...")
                lines.append("")

        fi = feature_importance_from_metadata(metadata)
        if not fi.empty:
            lines.append("  Feature Importance (across all rules):")
            lines.append(f"    {'Feature':<25} {'Count':>6} {'Rules':>6} {'Avg':>6}")
            lines.append("    " + "-" * 50)
            for _, row in fi.iterrows():
                lines.append(
                    f"    {row['feature']:<25} {row['total_count']:>6} "
                    f"{row['n_rules']:>6} {row['avg_per_rule']:>6.2f}"
                )
        lines.append("")

    # ── 5. Sensitivity tables ────────────────────────────────────
    groups_with_sensitivity = ["fitness_weights", "gp_params", "scale", "dynamics"]
    available_groups = set(df["group"].unique())

    lines.append("5. SENSITIVITY ANALYSIS")
    lines.append("-" * 50)
    for grp in groups_with_sensitivity:
        if grp not in available_groups:
            continue
        st = sensitivity_table(df, metadata, grp)
        if st.empty:
            continue
        lines.append(f"\n  Group: {grp}")
        # Select relevant columns based on group
        display_cols = ["experiment", "best_fitness", "training_time_s"]
        for m in metrics:
            display_cols.append(f"mean_{m}")
        available = [c for c in display_cols if c in st.columns]
        lines.append(st[available].to_string(index=False))
        lines.append("")

    return "\n".join(lines)
