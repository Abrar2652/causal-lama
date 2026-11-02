#!/usr/bin/env python3
"""Deep-but-benign contrasts, and depth statistics with uncertainty.

Closes the strongest remaining attack on the depth-versus-magnitude finding. Every
non-refusal contrast in nonrefusal_controls.py is *lexically* separable (code vs prose,
question vs imperative, long vs short), so an early-layer peak is exactly what one would
predict of them whether or not refusal is involved. The depth gap might therefore be
tracking semantic abstraction rather than refusal. Three further contrasts probe that:

  topic_moral_benign   benign ethics / values / social-norm prompts vs the rest. The
                       important one: it is topically adjacent to harm and demands moral
                       reasoning, but contains nothing that triggers refusal. If the late
                       peak is really harm-topic recognition or moral reasoning, THIS
                       should peak late. If it peaks early, topicality near harm is not
                       sufficient and actual refusal-relevant content is needed.
  reasoning_multistep  prompts requiring multi-step inference vs simple lookups: a direct
                       proxy for depth of processing.
  valence_negative     negative-affect benign prompts vs the rest: deep but harmless.

Second contribution: the published depth claim rests on argmax over 28 layers, which is a
noisy statistic, and on a rank correlation that is not independent of it. This script adds
a bootstrap over the harmful class, which is the limiting sample at n=260 and the one
reviewer qp6c-W3 questions, and reports

  peak layer with a 95% interval,
  centre of mass of the profile, normalised to [0, 1] over depth, which is stable where
  argmax is not,

for every contrast including the paper's own. Because RCS(l) depends on the small class
only once the reference centroid is fixed, resampling that class is the correct and cheap
bootstrap.

Writes results/matched_controls/deep_benign_qwen2-7b.{json,csv}
"""
import csv, json, os, sys
import numpy as np

REPO = "/nas/home/jahin/causal-lama"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analysis"))

from lama_rcs import compute_rcs
import null_core as nc
import nonrefusal_controls as nrc
from scipy.stats import spearmanr

HOLD = os.path.join(REPO, "dataset", "representations", "qwen2-7b_harmless_holdout")
PUB = os.path.join(REPO, "dataset", "representations", "qwen2-7b_processed_confirm")
OUT = os.path.join(REPO, "results", "matched_controls")
SEED, N_A, N_B, N_BOOT, EPS = 42, 260, 6000, 1000, 1e-8

MORAL_KW = {
    "ethical", "ethics", "moral", "morality", "right", "wrong", "fair", "fairness",
    "justice", "unjust", "honest", "honesty", "dishonest", "integrity", "virtue",
    "responsibility", "responsible", "duty", "obligation", "consent", "respect",
    "disrespect", "harm", "harmful", "benefit", "trust", "betray", "loyalty",
    "forgive", "forgiveness", "guilt", "blame", "apologize", "apology", "deserve",
    "should", "ought", "values", "principle", "principles", "dilemma", "conscience",
    "empathy", "compassion", "selfish", "altruism", "equity", "discrimination",
    "bias", "privacy", "cheating", "lying", "stealing", "promise",
}
REASON_KW = {
    "why", "explain", "compare", "contrast", "analyze", "analyse", "evaluate",
    "implications", "consequences", "because", "reason", "reasons", "cause", "causes",
    "effect", "effects", "calculate", "derive", "prove", "infer", "deduce", "justify",
    "tradeoff", "trade-off", "versus", "vs", "difference", "differences", "relationship",
    "steps", "step-by-step", "strategy", "plan", "design", "optimize", "solve",
}
LOOKUP_KW = {
    "what", "who", "when", "where", "name", "list", "define", "definition", "give",
    "tell", "describe", "identify", "translate", "spell", "convert", "summarize",
}
NEG_KW = {
    "problem", "problems", "conflict", "mistake", "mistakes", "failure", "fail",
    "failed", "sad", "sadness", "angry", "anger", "upset", "difficult", "hard",
    "worst", "bad", "worse", "risk", "risks", "danger", "dangerous", "fear",
    "anxiety", "anxious", "stress", "stressed", "depressed", "depression", "lonely",
    "grief", "loss", "lose", "lost", "hurt", "pain", "painful", "suffer", "suffering",
    "complain", "complaint", "argue", "argument", "fight", "broke", "broken", "crisis",
    "struggle", "struggling", "worry", "worried", "regret", "disappointed",
}


def words(p):
    return set(p.lower().replace(",", " ").replace(".", " ").replace("?", " ")
               .replace("!", " ").replace(";", " ").replace(":", " ").split())


def rcs_fast(A, nu):
    """RCS(l) for small class A against a FIXED reference centroid nu."""
    mu = A.mean(axis=0)
    gap = np.linalg.norm(mu - nu, axis=1)
    sig = np.linalg.norm(A - mu[None, :, :], axis=2).mean(axis=0)
    return gap / (sig + EPS)


def depth_stats(v):
    L = len(v)
    w = v / v.sum()
    return {"peak_layer": int(np.argmax(v)),
            "com_depth": float((np.arange(L) * w).sum() / (L - 1))}


def summarize(v):
    L = len(v)
    return {"mean_rcs": float(v.mean()), "peak_rcs": float(v.max()),
            "peak_over_mean": float(v.max() / v.mean()), "gini": float(nc.gini(v)),
            "n80_over_L": float(nc.n80(v) / L), "cv": float(v.std() / v.mean()),
            **depth_stats(v)}


def bootstrap(A, nu, n_boot=N_BOOT, seed=SEED):
    """Resample the small class; the reference centroid is held fixed."""
    rng = np.random.RandomState(seed)
    n = A.shape[0]
    peaks, coms = np.zeros(n_boot, int), np.zeros(n_boot)
    for b in range(n_boot):
        v = rcs_fast(A[rng.randint(0, n, n)], nu)
        d = depth_stats(v)
        peaks[b], coms[b] = d["peak_layer"], d["com_depth"]
    return {"peak_layer_q2.5": float(np.percentile(peaks, 2.5)),
            "peak_layer_q50": float(np.percentile(peaks, 50)),
            "peak_layer_q97.5": float(np.percentile(peaks, 97.5)),
            "peak_layer_mode": int(np.bincount(peaks).argmax()),
            "com_depth_mean": float(coms.mean()),
            "com_depth_q2.5": float(np.percentile(coms, 2.5)),
            "com_depth_q97.5": float(np.percentile(coms, 97.5)),
            "n_boot": n_boot}


def main():
    import torch
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(SEED)

    H = torch.load(os.path.join(HOLD, "HLholdout_train.pt")).numpy().astype(np.float32)
    meta = json.load(open(os.path.join(HOLD, "prompts.json")))
    prompts = meta["prompts"]
    assert len(prompts) == H.shape[0]
    HFp = torch.load(os.path.join(PUB, "HFx_train.pt")).numpy().astype(np.float32)
    HLp = torch.load(os.path.join(PUB, "HLx_train.pt")).numpy().astype(np.float32)
    print(f"holdout {H.shape}  published {HFp.shape}/{HLp.shape}", flush=True)

    # the fast path must agree with the paper's own implementation before anything else
    ref = compute_rcs(HFp, HLp)[0]
    base = np.array([ref[l] for l in range(len(ref))])
    chk = rcs_fast(HFp, HLp.mean(axis=0))
    print(f"[check] rcs_fast vs lama_rcs.compute_rcs max|diff|={np.max(np.abs(chk-base)):.3e}")
    assert np.max(np.abs(chk - base)) < 1e-4

    cached = os.path.join(REPO, "logs", "rcs_analysis_qwen2-7b.json")
    cd = json.load(open(cached))
    pub = np.array([cd["rcs_scores"][str(l)] for l in range(len(base))])
    print(f"[check] vs published logs max|diff|={np.max(np.abs(pub-base)):.3e}")

    rows = {}

    def add(name, A, nu, note, delta_len=None, n_b=None):
        v = rcs_fast(A, nu)
        null = nc.analytic_null_mean(A.shape[0], n_b)
        rows[name] = dict(n_a=int(A.shape[0]), n_b=int(n_b), analytic_null_mean=float(null),
                          ratio_obs_over_null=float(v.mean() / null),
                          spearman_vs_published=float(spearmanr(v, base)[0]),
                          cliffs_delta_length=delta_len, note=note,
                          **summarize(v), **bootstrap(A, nu))
        rows[name]["profile"] = v.tolist()
        r = rows[name]
        print(f"  {name:24s} n={r['n_a']}/{r['n_b']:5d} mean={r['mean_rcs']:.4f} "
              f"peak=L{r['peak_layer']:<2d}[{r['peak_layer_q2.5']:.0f},{r['peak_layer_q97.5']:.0f}] "
              f"com={r['com_depth']:.3f}[{r['com_depth_q2.5']:.3f},{r['com_depth_q97.5']:.3f}] "
              f"gini={r['gini']:.4f} {r['ratio_obs_over_null']:.1f}x "
              f"rho={r['spearman_vs_published']:+.3f}", flush=True)

    print("\nthe paper's own contrast:")
    add("refusal_published", HFp, HLp.mean(axis=0),
        "the published harmful-vs-harmless contrast", 0.148, HLp.shape[0])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    lens = np.array([len(tok(p).input_ids) for p in prompts])
    W = [words(p) for p in prompts]

    is_moral = np.array([len(MORAL_KW & w) >= 1 for w in W])
    deep = np.array([len(REASON_KW & w) >= 1 for w in W])
    shallow = np.array([len(LOOKUP_KW & w) >= 1 and len(REASON_KW & w) == 0 for w in W])
    is_neg = np.array([len(NEG_KW & w) >= 1 for w in W])
    print(f"\nshares: moral={is_moral.mean():.3f} multistep={deep.mean():.3f} "
          f"lookup={shallow.mean():.3f} negative={is_neg.mean():.3f}")

    print("\ndeep-but-benign contrasts, formed inside the held-out harmless pool:")

    def pair(name, mask_a, mask_b, note):
        ia = np.where(mask_a)[0]; ib = np.where(mask_b)[0]
        if len(ia) < 60 or len(ib) < 300:
            print(f"  [skip] {name}: too few ({len(ia)}, {len(ib)})"); return
        ia = rng.choice(ia, size=min(N_A, len(ia)), replace=False)
        ib = rng.choice(ib, size=min(N_B, len(ib)), replace=False)
        d = float(nrc.cliffs_delta(lens[ia], lens[ib]))
        add(name, H[np.sort(ia)], H[np.sort(ib)].mean(axis=0), note, d, len(ib))

    pair("topic_moral_benign", is_moral, ~is_moral,
         "benign ethics/values prompts vs the rest: harm-adjacent, moral reasoning, no refusal")
    pair("reasoning_multistep", deep, shallow,
         "multi-step inference vs simple lookup: a depth-of-processing proxy")
    pair("valence_negative", is_neg, ~is_neg,
         "negative-affect benign prompts vs the rest")

    print("\nthe original lexical contrasts, now with depth intervals:")
    is_code = np.array([bool(nrc.CODE_KW & w) for w in W])
    first = np.array([(p.strip().split() or [""])[0].lower().strip(",.:;?") for p in prompts])
    is_q = np.array([f in nrc.Q_WORDS for f in first])
    pair("topic_code", is_code, ~is_code, "code/technical vs prose")
    pair("register_question", is_q, ~is_q, "interrogative vs imperative")
    order = np.argsort(-lens)
    m1 = np.zeros(len(prompts), bool); m1[order[:N_A]] = True
    m2 = np.zeros(len(prompts), bool); m2[order[N_A:N_A + N_B]] = True
    pair("length_extreme", m1, m2, "the 260 longest vs 6000 others")
    perm = rng.permutation(len(prompts))
    m1 = np.zeros(len(prompts), bool); m1[perm[:N_A]] = True
    m2 = np.zeros(len(prompts), bool); m2[perm[N_A:N_A + N_B]] = True
    pair("random_split", m1, m2, "pure random split: a floor")

    # does any non-refusal contrast's depth interval overlap the paper's?
    r0 = rows["refusal_published"]
    print("\ndepth separation from the published contrast:")
    verdict = {}
    for k, r in rows.items():
        if k == "refusal_published": continue
        ov = not (r["com_depth_q97.5"] < r0["com_depth_q2.5"] or
                  r["com_depth_q2.5"] > r0["com_depth_q97.5"])
        verdict[k] = {"com_interval_overlaps_refusal": bool(ov),
                      "com_gap": float(r0["com_depth"] - r["com_depth"])}
        print(f"  {k:24s} com gap {verdict[k]['com_gap']:+.3f}  "
              f"{'OVERLAPS' if ov else 'disjoint'}")

    res = {"seed": SEED, "n_boot": N_BOOT,
           "holdout_slice": {"start": meta["start"], "n": meta["n"]},
           "fast_path_max_abs_dev_vs_lama_rcs": float(np.max(np.abs(chk - base))),
           "published_logs_max_abs_dev": float(np.max(np.abs(pub - base))),
           "contrasts": rows, "depth_separation": verdict}
    with open(os.path.join(OUT, "deep_benign_qwen2-7b.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(OUT, "deep_benign_qwen2-7b.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "n_a", "n_b", "mean_rcs", "ratio_obs_over_null", "gini",
                    "peak_layer", "peak_q2.5", "peak_q97.5", "com_depth", "com_q2.5",
                    "com_q97.5", "spearman_vs_published", "cliffs_delta_length",
                    "com_overlaps_refusal", "note"])
        for k, r in rows.items():
            w.writerow([k, r["n_a"], r["n_b"], f"{r['mean_rcs']:.4f}",
                        f"{r['ratio_obs_over_null']:.1f}", f"{r['gini']:.4f}",
                        r["peak_layer"], f"{r['peak_layer_q2.5']:.0f}",
                        f"{r['peak_layer_q97.5']:.0f}", f"{r['com_depth']:.4f}",
                        f"{r['com_depth_q2.5']:.4f}", f"{r['com_depth_q97.5']:.4f}",
                        f"{r['spearman_vs_published']:+.4f}",
                        "" if r["cliffs_delta_length"] is None else f"{r['cliffs_delta_length']:+.3f}",
                        "" if k == "refusal_published" else
                        str(verdict[k]["com_interval_overlaps_refusal"]), r["note"]])
    print(f"\nwrote {OUT}/deep_benign_qwen2-7b.{{json,csv}}")


if __name__ == "__main__":
    main()
