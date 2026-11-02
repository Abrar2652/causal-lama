#!/usr/bin/env python3
"""Non-refusal control contrasts (reviewers PKr6, p6Gj-W1, qp6c-W1).

The sharpest available test of refusal-specificity. Every contrast here is formed
ENTIRELY INSIDE the held-out harmless pool, so by construction neither class contains
any refusal-triggering content. Class sizes are held at the published 260 / 6000 so the
permutation null is numerically identical to the paper's.

If RCS on a non-refusal contrast is near the null (mean ~0.063, Gini ~0.04), RCS is
detecting something specific to the harmful-versus-harmless distinction. If it instead
reproduces the published profile (mean ~1.20, Gini ~0.19), RCS is a generic separability
statistic and the refusal interpretation is not supported by it alone. Intermediate
values are reported as such.

Contrasts:
  topic_code         code/technical prompts vs the rest
  register_question  interrogative-initial vs imperative-initial
  length_extreme     the 260 longest vs 6000 others
  length_effectmatched  a length contrast tuned to the SAME Cliff's delta (0.148) as the
                     real harmful-vs-harmless length difference, which is the fair
                     magnitude comparison
  random_split       a pure random split: the permutation null by another route, included
                     as a floor
"""
import csv, json, os, sys
import numpy as np

REPO = "/nas/home/jahin/causal-lama"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analysis"))

from lama_rcs import compute_rcs
import null_core as nc
from scipy.stats import spearmanr, mannwhitneyu

HOLD = os.path.join(REPO, "dataset", "representations", "qwen2-7b_harmless_holdout")
PUB = os.path.join(REPO, "dataset", "representations", "qwen2-7b_processed_confirm")
OUT = os.path.join(REPO, "results", "matched_controls")
SEED = 42
N_A, N_B = 260, 6000
TARGET_DELTA = 0.148   # measured harmful-vs-harmless Cliff's delta on token length

CODE_KW = {
    "python", "java", "javascript", "html", "css", "sql", "code", "function", "algorithm",
    "program", "programming", "variable", "array", "loop", "string", "integer", "boolean",
    "class", "method", "api", "database", "query", "script", "compile", "debug", "syntax",
    "recursion", "regex", "json", "xml", "git", "linux", "bash", "shell", "server",
    "framework", "library", "binary", "sort", "hash", "pointer", "compiler", "runtime",
}
Q_WORDS = {"what", "why", "how", "who", "when", "where", "which", "is", "are", "can",
           "does", "do", "should", "would", "could", "will", "did", "was", "were"}


def summarize(v):
    return {"mean_rcs": float(v.mean()), "peak_layer": int(np.argmax(v)),
            "peak_rcs": float(v.max()), "peak_over_mean": float(v.max() / v.mean()),
            "gini": float(nc.gini(v)), "n80": int(nc.n80(v)),
            "n80_over_L": float(nc.n80(v) / len(v)), "cv": float(v.std() / v.mean())}


def prof(A, B):
    rcs, _ = compute_rcs(A, B)
    return np.array([rcs[l] for l in range(len(rcs))])


def cliffs_delta(x, y):
    u, _ = mannwhitneyu(x, y)
    return 2 * u / (len(x) * len(y)) - 1


def weighted_length_split(lens, rng, beta, n_a=N_A, n_b=N_B):
    """Draw class A with sampling weight exp(beta * z(length)); B from the remainder."""
    z = (lens - lens.mean()) / (lens.std() + 1e-9)
    w = np.exp(np.clip(beta * z, -30, 30))
    w = w / w.sum()
    a = rng.choice(len(lens), size=n_a, replace=False, p=w)
    rest = np.setdiff1d(np.arange(len(lens)), a)
    b = rng.choice(rest, size=min(n_b, len(rest)), replace=False)
    return np.sort(a), np.sort(b)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(SEED)
    import torch

    H = torch.load(os.path.join(HOLD, "HLholdout_train.pt")).numpy().astype(np.float32)
    meta = json.load(open(os.path.join(HOLD, "prompts.json")))
    prompts = meta["prompts"]
    assert len(prompts) == H.shape[0]
    print(f"holdout reps {H.shape}  slice start={meta['start']} n={meta['n']}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    lens = np.array([len(tok(p).input_ids) for p in prompts])

    low = [p.lower() for p in prompts]
    is_code = np.array([bool(CODE_KW & set(p.replace(",", " ").replace(".", " ").split()))
                        for p in low])
    first = np.array([(p.strip().split() or [""])[0].lower().strip(",.:;?") for p in low])
    is_q = np.array([f in Q_WORDS for f in first])
    print(f"code share {is_code.mean():.3f}  question-initial share {is_q.mean():.3f}")

    # the published profile, for comparison
    HFp = torch.load(os.path.join(PUB, "HFx_train.pt")).numpy().astype(np.float32)
    HLp = torch.load(os.path.join(PUB, "HLx_train.pt")).numpy().astype(np.float32)
    base = prof(HFp, HLp)
    print(f"published: mean={base.mean():.4f} peak=L{int(np.argmax(base))} gini={nc.gini(base):.4f}")

    contrasts = {}

    def add(name, ia, ib, note=""):
        ia = np.asarray(ia); ib = np.asarray(ib)
        if len(ia) < 30 or len(ib) < 100:
            print(f"  [skip] {name}: too few ({len(ia)}, {len(ib)})"); return
        v = prof(H[ia], H[ib])
        d = cliffs_delta(lens[ia], lens[ib])
        null = nc.analytic_null_mean(len(ia), len(ib))
        c = dict(n_a=int(len(ia)), n_b=int(len(ib)), cliffs_delta_length=float(d),
                 analytic_null_mean=float(null), ratio_obs_over_null=float(v.mean() / null),
                 spearman_vs_published=float(spearmanr(v, base)[0]),
                 note=note, **summarize(v))
        c["profile"] = v.tolist()
        contrasts[name] = c
        print(f"  {name}: n={len(ia)}/{len(ib)} mean={v.mean():.4f} peak=L{int(np.argmax(v))} "
              f"gini={nc.gini(v):.4f} ratio={c['ratio_obs_over_null']:.1f}x "
              f"rho_pub={c['spearman_vs_published']:.3f} delta_len={d:+.3f}")

    print("\ncontrasts inside the harmless pool (no refusal content on either side):")

    ci = np.where(is_code)[0]; nci = np.where(~is_code)[0]
    add("topic_code", rng.choice(ci, size=min(N_A, len(ci)), replace=False),
        rng.choice(nci, size=min(N_B, len(nci)), replace=False),
        "code/technical vs non-technical: a strong non-refusal semantic contrast")

    qi = np.where(is_q)[0]; nqi = np.where(~is_q)[0]
    add("register_question", rng.choice(qi, size=min(N_A, len(qi)), replace=False),
        rng.choice(nqi, size=min(N_B, len(nqi)), replace=False),
        "interrogative vs imperative register")

    order = np.argsort(-lens)
    add("length_extreme", order[:N_A], order[N_A:N_A + N_B],
        "the 260 longest vs 6000 others: a deliberately large structural contrast")

    # tune beta so the length contrast matches the real harmful-vs-harmless delta
    lo, hi, best = 0.0, 6.0, None
    for _ in range(18):
        mid = (lo + hi) / 2
        a, b = weighted_length_split(lens, np.random.RandomState(SEED), mid)
        d = cliffs_delta(lens[a], lens[b])
        best = (mid, a, b, d)
        if d < TARGET_DELTA: lo = mid
        else: hi = mid
    beta, a, b, d = best
    print(f"  [tuned] beta={beta:.3f} gives Cliff's delta={d:+.3f} (target {TARGET_DELTA})")
    add("length_effectmatched", a, b,
        f"length contrast tuned to the real harmful-vs-harmless delta ({TARGET_DELTA})")

    perm = rng.permutation(len(prompts))
    add("random_split", perm[:N_A], perm[N_A:N_A + N_B],
        "pure random split: a floor, equivalent to one permutation draw")

    res = {"published": {"n_a": int(HFp.shape[0]), "n_b": int(HLp.shape[0]), **summarize(base)},
           "holdout_slice": {"start": meta["start"], "n": meta["n"]},
           "contrasts": contrasts}
    with open(os.path.join(OUT, "nonrefusal_controls_qwen2-7b.json"), "w") as f:
        json.dump(res, f, indent=2)

    with open(os.path.join(OUT, "nonrefusal_controls_qwen2-7b.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "n_a", "n_b", "mean_rcs", "peak_layer", "gini", "n80_over_L",
                    "cv", "analytic_null_mean", "ratio_obs_over_null",
                    "spearman_vs_published", "cliffs_delta_length", "note"])
        bb = res["published"]
        w.writerow(["published_harmful_vs_harmless", bb["n_a"], bb["n_b"],
                    f"{bb['mean_rcs']:.4f}", bb["peak_layer"], f"{bb['gini']:.4f}",
                    f"{bb['n80_over_L']:.4f}", f"{bb['cv']:.4f}",
                    f"{nc.analytic_null_mean(bb['n_a'], bb['n_b']):.4f}",
                    f"{bb['mean_rcs']/nc.analytic_null_mean(bb['n_a'], bb['n_b']):.1f}",
                    "1.0000", f"{TARGET_DELTA:+.3f}", "the paper's own contrast"])
        for n, c in contrasts.items():
            w.writerow([n, c["n_a"], c["n_b"], f"{c['mean_rcs']:.4f}", c["peak_layer"],
                        f"{c['gini']:.4f}", f"{c['n80_over_L']:.4f}", f"{c['cv']:.4f}",
                        f"{c['analytic_null_mean']:.4f}", f"{c['ratio_obs_over_null']:.1f}",
                        f"{c['spearman_vs_published']:.4f}",
                        f"{c['cliffs_delta_length']:+.3f}", c["note"]])
    print(f"\nwrote {OUT}/nonrefusal_controls_qwen2-7b.{{json,csv}}")


if __name__ == "__main__":
    main()
