"""lama_stats.py — Statistical harness for reviewer-proof ASR reporting.

The baseline (Piras et al.) reports point ASR with no confidence intervals and no
significance test for "method A beats method B". For AAAI we report, for every
ASR number:
  - bootstrap 95% CI over behaviors          (bootstrap_asr_ci)
  - paired significance vs another method    (paired_bootstrap_diff / mcnemar_test)
  - mean ± std (+ CI) across random seeds    (aggregate_seeds)

All paired tests assume the two methods are evaluated on the SAME ordered set of
behaviors, so judgments are paired index-by-index. evaluate_jailbreak() now saves
`harmbench_classifications` and `harmbench_behaviors` to make this possible.
"""
import json
import numpy as np
from scipy import stats as _sps


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_classifications(eval_json_path):
    """Load per-behavior judgments from an evaluate_jailbreak() output JSON.

    Returns (classifications: np.ndarray[int], behaviors: list[str]).
    Falls back to reading per-completion `is_jailbreak_harmbench` for older files.
    """
    with open(eval_json_path) as f:
        ev = json.load(f)
    if "harmbench_classifications" in ev:
        cls = np.asarray(ev["harmbench_classifications"], dtype=int)
        beh = ev.get("harmbench_behaviors", [None] * len(cls))
        return cls, list(beh)
    comps = ev.get("completions", [])
    cls = np.asarray([int(c.get("is_jailbreak_harmbench", 0)) for c in comps], dtype=int)
    beh = [c.get("prompt") for c in comps]
    return cls, beh


# -----------------------------------------------------------------------------
# Single-method CI
# -----------------------------------------------------------------------------
def bootstrap_asr_ci(classifications, n_bootstrap=10000, ci_alpha=0.05, seed=42):
    """Bootstrap CI for ASR = mean(classifications) over behaviors.

    Returns dict {asr, ci_lo, ci_hi, se, n}. Also reports the Wilson interval as a
    closed-form cross-check (good for binomial proportions, robust at extremes).
    """
    cls = np.asarray(classifications, dtype=float)
    n = len(cls)
    asr = float(cls.mean()) if n else 0.0
    rng = np.random.RandomState(seed)
    if n == 0:
        return {"asr": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "se": 0.0, "n": 0,
                "wilson_lo": 0.0, "wilson_hi": 0.0}
    boot = cls[rng.randint(0, n, size=(n_bootstrap, n))].mean(axis=1)
    ci_lo = float(np.percentile(boot, 100 * ci_alpha / 2))
    ci_hi = float(np.percentile(boot, 100 * (1 - ci_alpha / 2)))
    # Wilson score interval
    z = _sps.norm.ppf(1 - ci_alpha / 2)
    denom = 1 + z**2 / n
    centre = (asr + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(asr * (1 - asr) / n + z**2 / (4 * n**2))) / denom
    return {
        "asr": asr, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "se": float(boot.std()), "n": n,
        "wilson_lo": float(centre - half), "wilson_hi": float(centre + half),
    }


# -----------------------------------------------------------------------------
# Paired method comparison (same behaviors)
# -----------------------------------------------------------------------------
def paired_bootstrap_diff(class_a, class_b, n_bootstrap=10000, ci_alpha=0.05, seed=42):
    """Paired bootstrap for ASR(A) - ASR(B) on the SAME behaviors.

    Resamples behavior indices once per replicate and applies them to both methods
    (preserving pairing). Returns the difference, its CI, and a two-sided bootstrap
    p-value (fraction of replicates on the opposite side of 0, doubled).
    """
    a = np.asarray(class_a, dtype=float)
    b = np.asarray(class_b, dtype=float)
    assert len(a) == len(b), "paired test requires equal-length, behavior-aligned arrays"
    n = len(a)
    diff = float(a.mean() - b.mean())
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_bootstrap, n))
    boot = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    ci_lo = float(np.percentile(boot, 100 * ci_alpha / 2))
    ci_hi = float(np.percentile(boot, 100 * (1 - ci_alpha / 2)))
    # two-sided bootstrap p-value about 0
    p_left = float(np.mean(boot <= 0))
    p_right = float(np.mean(boot >= 0))
    p = float(min(1.0, 2 * min(p_left, p_right)))
    return {
        "asr_a": float(a.mean()), "asr_b": float(b.mean()),
        "diff": diff, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_value": p, "n": n,
        "significant_0.05": bool(ci_lo > 0 or ci_hi < 0),
    }


def mcnemar_test(class_a, class_b, exact_threshold=25):
    """McNemar's test for paired binary outcomes (same behaviors).

    Uses discordant pairs: b = A-success & B-fail, c = A-fail & B-success. Exact
    binomial test when discordant count is small, else continuity-corrected chi².
    """
    a = np.asarray(class_a, dtype=int)
    b = np.asarray(class_b, dtype=int)
    assert len(a) == len(b)
    n01 = int(np.sum((a == 1) & (b == 0)))  # A wins
    n10 = int(np.sum((a == 0) & (b == 1)))  # B wins
    disc = n01 + n10
    if disc == 0:
        return {"n01_a_wins": 0, "n10_b_wins": 0, "statistic": 0.0, "p_value": 1.0,
                "method": "none (no discordant pairs)"}
    if disc < exact_threshold:
        p = float(_sps.binomtest(min(n01, n10), disc, 0.5).pvalue)
        method = "exact binomial"
        statistic = float(min(n01, n10))
    else:
        statistic = (abs(n01 - n10) - 1) ** 2 / disc
        p = float(_sps.chi2.sf(statistic, df=1))
        method = "chi2 (continuity-corrected)"
    return {"n01_a_wins": n01, "n10_b_wins": n10, "statistic": float(statistic),
            "p_value": p, "method": method}


# -----------------------------------------------------------------------------
# Multi-seed aggregation
# -----------------------------------------------------------------------------
def aggregate_seeds(values, ci_alpha=0.05):
    """Mean ± std (+ t-based CI of the mean) across seed runs.

    `values` is a list of per-seed scalars (e.g. ASR per seed). Reports the
    standard 'mean ± std over N seeds' plus a CI of the mean for tables.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean()) if n else 0.0
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    if n > 1:
        tcrit = _sps.t.ppf(1 - ci_alpha / 2, df=n - 1)
        ci = (mean - tcrit * sem, mean + tcrit * sem)
    else:
        ci = (mean, mean)
    return {"mean": mean, "std": std, "sem": float(sem), "n_seeds": n,
            "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "values": v.tolist()}


def compare_methods_from_files(eval_path_a, eval_path_b):
    """Convenience: full paired comparison from two evaluate_jailbreak JSONs.

    Verifies behavior alignment before testing.
    """
    a, beh_a = load_classifications(eval_path_a)
    b, beh_b = load_classifications(eval_path_b)
    if beh_a and beh_b and beh_a != beh_b:
        # align by behavior text if order differs
        common = [x for x in beh_a if x in set(beh_b)]
        ia = {t: i for i, t in enumerate(beh_a)}
        ib = {t: i for i, t in enumerate(beh_b)}
        a = np.array([a[ia[t]] for t in common])
        b = np.array([b[ib[t]] for t in common])
    return {
        "asr_a": bootstrap_asr_ci(a),
        "asr_b": bootstrap_asr_ci(b),
        "paired_diff": paired_bootstrap_diff(a, b),
        "mcnemar": mcnemar_test(a, b),
    }


if __name__ == "__main__":
    # Self-test (CPU, no model needed)
    rng = np.random.RandomState(0)
    # method A clearly better than B on the same 159 behaviors
    n = 159
    base = rng.binomial(1, 0.4, n)
    a = base.copy(); b = base.copy()
    flip = rng.rand(n) < 0.2
    a[flip] = 1  # A succeeds on extra 20%
    print("ASR A CI:", bootstrap_asr_ci(a))
    print("ASR B CI:", bootstrap_asr_ci(b))
    pd = paired_bootstrap_diff(a, b)
    print("paired diff:", pd)
    mc = mcnemar_test(a, b)
    print("mcnemar:", mc)
    print("seeds:", aggregate_seeds([0.55, 0.58, 0.53, 0.60]))
    assert pd["diff"] > 0 and pd["significant_0.05"], "A should significantly beat B"
    assert mc["p_value"] < 0.05
    print("\nlama_stats self-test PASSED ✅")
