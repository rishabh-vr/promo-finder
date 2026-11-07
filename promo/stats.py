"""Small, dependency-free statistics used by the evaluations.

Kept here so every reported number has one implementation and a unit test:
  wilson(k, n)          95% interval for a proportion (good at small n, unlike the normal approx.)
  sign_test(w, l)       exact two-sided binomial test of w wins vs l losses (ties dropped)
  kappa_quadratic(a, b) quadratic-weighted Cohen's kappa for 1-5 ratings (rater agreement)
  spearman(x, y)        rank correlation with average ranks for ties
  bootstrap_ci(xs, f)   percentile bootstrap interval for any statistic f
"""
from __future__ import annotations

import math
import random


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


def sign_test(wins: int, losses: int) -> float | None:
    """Exact two-sided p-value for H0: P(win) = 0.5 among decided pairs."""
    n = wins + losses
    if n == 0:
        return None
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def min_pairs_for_significance(win_rate: float, alpha: float = 0.05, max_n: int = 2000) -> int | None:
    """Smallest number of decided pairs at which `win_rate` would give a two-sided sign-test p < alpha.
    Answers 'how many pairs do we need to review before this gap is real?'"""
    for n in range(5, max_n):
        w = round(win_rate * n)
        if (p := sign_test(w, n - w)) is not None and p < alpha:
            return n
    return None


def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 2) if den else None


def kappa_quadratic(a: list[int], b: list[int], lo: int = 1, hi: int = 5) -> float | None:
    if len(a) != len(b) or not a:
        return None
    k = hi - lo + 1
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[int(x) - lo][int(y) - lo] += 1
    n = len(a)
    ra = [sum(r) for r in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = lambda i, j: (i - j) ** 2 / (k - 1) ** 2
    o = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    e = sum(w(i, j) * ra[i] * cb[j] for i in range(k) for j in range(k)) / (n * n)
    return round(1 - o / e, 3) if e else None


def bootstrap_ci(xs: list, stat, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    if not xs:
        return None
    rng = random.Random(seed)
    vals = sorted(stat([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(n_boot))
    return round(vals[int(alpha / 2 * n_boot)], 3), round(vals[int((1 - alpha / 2) * n_boot) - 1], 3)
