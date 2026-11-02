#!/usr/bin/env python3
"""Joint multi-layer causal ablation: is refusal causally layer-distributed? (reviewer PKr6)

Why this exists. The paper's causal evidence (`causal_refusal_prob.py`) ablates ONE layer at
a time and correlates the per-layer effect with RCS. LAMA, the method the paper proposes,
intervenes at many layers at once. A correlation between RCS and single-layer effects
therefore cannot establish the title's claim: it shows that high-RCS layers matter
individually, not that refusal is distributed across them. PKr6 asked for this experiment and
the first audit pass deferred it.

Metric. Mean probability mass on refusal tokens at the first generated position over harmful
prompts -- the paper's own graded metric, so single-layer numbers here are directly
comparable with `logs/causal_refusalprob_qwen2-7b.json`. No text is generated, so no
completion exists to judge or to store.

    P_refuse(S) = mean_prompts sum_{t in refusal_ids} P(t | prompt, ablating layer set S)
    DeltaP(S)   = P_refuse({}) - P_refuse(S)

Five things are measured.

  1. SINGLE-LAYER sweep, l = 0..L-1. Reproduces the paper's probe on these tensors and
     supplies the per-layer effects the additivity test needs.
  2. CUMULATIVE curve over layers ranked by RCS, k = 1..L. How fast this saturates is the
     causal analogue of the representational n80/L the paper reports: if refusal were
     localised, the single best layer would capture most of the achievable effect and the
     curve would flatten at k = 1.
  3. MATCHED-k SELECTION RULES. At equal k: top-k by RCS, bottom-k by RCS, top-k by the raw
     diff-in-means norm, and random k over several draws. This is what makes the claim causal
     rather than correlational -- if RCS-chosen layers are no better than random ones at the
     same budget, RCS does not identify intervention sites. The raw-norm arm is the
     non-circularity control, the covariate the paper's own partial correlation uses, since
     RCS shares its numerator with the ablated direction.
  4. ADDITIVITY. DeltaP(top-k) against the sum of individual DeltaP(l) over those same
     layers. Strong sub-additivity means the layers are redundant; near-additivity means they
     contribute separately. "Distributed" and "redundant" are different claims and this
     separates them.
  5. A DAMAGE CONTROL, because a large DeltaP is worthless if the intervention simply breaks
     the model. Every configuration also reports mean top-1 probability and mean next-token
     entropy on held-out harmless prompts. A drop in refusal that arrives together with an
     entropy blow-up is degradation, not refusal removal, and is reported as such.

Ablation mechanics are the repository's own: project the layer's diff-in-means direction out
of that block's `o_proj` and `down_proj` (`utils.ablation_utils.get_weighted_ablated_matrix`
at alpha = 1.0). Weight-space ablation is a projection and cannot be inverted, so originals
are cached on the host the first time a layer is touched; between configurations only layers
whose membership actually changes are written. A final check restores everything and
re-measures the baseline, which is the guard that the save/restore path is exact.

Writes results/joint_causal/joint_causal_<model>.{json,csv}
"""
import argparse, csv, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)                    # lama_rcs.load_representations uses a relative path

from utils.ablation_utils import get_weighted_ablated_matrix
from lama_rcs import (load_representations, compute_rcs, compute_direction_norms,
                      partial_spearman)
from scipy import stats

OUT_DIR = os.path.join(REPO, "results", "joint_causal")


def pick_devices(need_gib, reserve_gib=1.5):
    """Choose GPUs with the most free memory until the budget is covered (Ampere or newer:
    fp16 inference on this cluster's Turing cards mutates weights in VRAM, see
    CRITICAL_gpu_weight_corruption.md)."""
    avail = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        if p.major < 8:
            continue
        free, _ = torch.cuda.mem_get_info(i)
        usable = free / 2**30 - reserve_gib
        if usable > 0.5:
            avail.append((i, usable))
    avail.sort(key=lambda t: -t[1])
    chosen, got = [], 0.0
    for i, u in avail:
        chosen.append((i, u)); got += u
        if got >= need_gib:
            break
    return chosen, got


def shard_patch(max_memory):
    """Install a sharded loader, as analysis/extract_sharded.py does."""
    import models.language_models as lm
    from transformers import AutoTokenizer, AutoModelForCausalLM

    def sharded_load(self):
        if self.model is not None and self.tokenizer is not None:
            return
        tok = os.environ.get("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=tok, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, token=tok, torch_dtype=torch.float16,
            trust_remote_code=True, device_map="auto",
            max_memory=max_memory, low_cpu_mem_usage=True)
        self.model.requires_grad_(False); self.model.eval()
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.device = self.model.get_input_embeddings().weight.device
        placed = {str(v) for v in getattr(self.model, "hf_device_map", {}).values()}
        assert "cpu" not in placed and "disk" not in placed, \
            f"model did not fit on the given cards: {self.model.hf_device_map}"
        print(f"[shard] embedding device={self.device}", flush=True)

    lm.LanguageModel.load_model = sharded_load


class Ablator:
    """Directional ablation of an arbitrary SET of layers, reversibly."""

    def __init__(self, llm, directions):
        self.llm, self.dirs = llm, directions
        self.blocks = llm.model_block_modules
        self.orig, self.active = {}, set()

    def _save(self, l):
        if l not in self.orig:
            o, d = self.llm._get_block_proj_weights(self.blocks[l])
            self.orig[l] = (o.data.detach().to("cpu", copy=True),
                            d.data.detach().to("cpu", copy=True))

    def set_layers(self, S, alpha=1.0):
        S = {int(x) for x in S}
        for l in sorted(self.active - S):               # restore what is no longer wanted
            o, d = self.llm._get_block_proj_weights(self.blocks[l])
            oc, dc = self.orig[l]
            o.data.copy_(oc.to(o.data.device))
            d.data.copy_(dc.to(d.data.device))
        for l in sorted(S - self.active):               # ablate what is newly wanted
            self._save(l)
            o, d = self.llm._get_block_proj_weights(self.blocks[l])
            v = self.dirs[l]
            o.data = get_weighted_ablated_matrix(o.data.T, v, alpha).T.contiguous()
            d.data = get_weighted_ablated_matrix(d.data.T, v, alpha).T.contiguous()
        self.active = S
        return self


@torch.no_grad()
def first_token_stats(llm, prompts, refusal_ids, batch):
    """Mean refusal-token mass, mean top-1 probability and mean entropy at the first
    generated position. One pass serves both the effect metric and the damage control."""
    tok = llm.tokenizer
    tok.padding_side = "left"          # so the last real token is always at index -1
    in_dev = llm.model.get_input_embeddings().weight.device
    pref, top1, ent = [], [], []
    for i in range(0, len(prompts), batch):
        chunk = [llm._get_prompt(p) for p in prompts[i:i + batch]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(in_dev)
        out = llm.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        logits = out.logits[:, -1, :].float()
        p = F.softmax(logits, dim=-1)
        pref.extend(p[:, refusal_ids].sum(-1).cpu().tolist())
        top1.extend(p.max(-1).values.cpu().tolist())
        ent.extend((-(p * torch.log(p.clamp_min(1e-12))).sum(-1)).cpu().tolist())
    return float(np.mean(pref)), float(np.mean(top1)), float(np.mean(ent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2-7b")
    ap.add_argument("--reps", default="qwen2-7b_processed_confirm",
                    help="directory under dataset/representations/ supplying RCS and the "
                         "ablation directions; the default reproduces the published profile")
    ap.add_argument("--dataset", default="harmbench_test")
    ap.add_argument("--max_prompts", type=int, default=159)
    ap.add_argument("--n_harmless_probe", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k_grid", default="1,2,3,5,8,14")
    ap.add_argument("--n_random", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--need_gib", type=float, default=17.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # ---- RCS and directions, from tensors that reproduce the published profile ----------
    HF, HL = load_representations(args.reps)
    rcs, _ = compute_rcs(HF, HL)
    norms = compute_direction_norms(HF, HL)
    L = HF.shape[1]
    HF_t = torch.tensor(HF, dtype=torch.float32)
    HL_t = torch.tensor(HL, dtype=torch.float32)
    directions = {l: (HF_t[:, l, :].mean(0) - HL_t[:, l, :].mean(0)) for l in range(L)}
    rcs_v = np.array([rcs[l] for l in range(L)])
    norm_v = np.array([norms[l] for l in range(L)])
    print(f"[rcs] L={L} peak=L{int(np.argmax(rcs_v))} mean={rcs_v.mean():.4f}", flush=True)

    # ---- prompts -------------------------------------------------------------------------
    from dataset.load_dataset import load_dataset, load_dataset_split
    harmful = load_dataset(args.dataset, instructions_only=True)[:args.max_prompts]
    harmless_probe = [h for h in load_dataset_split("harmless", "train", instructions_only=True)
                      if h and h.strip()][:args.n_harmless_probe]
    print(f"[data] harmful={len(harmful)} ({args.dataset})  "
          f"harmless probe={len(harmless_probe)}", flush=True)

    # ---- model ---------------------------------------------------------------------------
    devs, got = pick_devices(args.need_gib)
    if got < args.need_gib:
        print(f"[FATAL] only {got:.1f} GiB usable across {[d for d, _ in devs]}, "
              f"need {args.need_gib}", flush=True)
        return 4
    max_memory = {i: f"{int(u)}GiB" for i, u in devs}
    print(f"[shard] {max_memory} total={got:.1f} GiB", flush=True)
    shard_patch(max_memory)
    from models.load_models import load_model
    llm = load_model(args.model, device=f"cuda:{devs[0][0]}")
    refusal_ids = list(getattr(llm, "refusal_token_id", []) or [])
    assert refusal_ids, f"{args.model} has no refusal_token_id"
    print(f"[model] refusal_token_id={refusal_ids}", flush=True)

    ab = Ablator(llm, directions)
    results, t0 = {}, time.time()

    def measure(S, label):
        ab.set_layers(S)
        pr, t1p, en = first_token_stats(llm, harmful, refusal_ids, args.batch)
        _, h_t1p, h_en = first_token_stats(llm, harmless_probe, refusal_ids, args.batch)
        rec = {"layers": sorted(int(x) for x in S), "k": len(S), "p_refuse": pr,
               "top1_harmful": t1p, "entropy_harmful": en,
               "top1_harmless": h_t1p, "entropy_harmless": h_en}
        print(f"  [{label:>22}] k={len(S):>2} P_refuse={pr:.4f} "
              f"dP={results.get('baseline', {}).get('p_refuse', float('nan')) - pr:+.4f} "
              f"H_harmless={h_en:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        return rec

    # ---- 0. baseline ---------------------------------------------------------------------
    print("\n=== baseline (no ablation) ===", flush=True)
    results["baseline"] = measure(set(), "baseline")
    base_p = results["baseline"]["p_refuse"]
    base_h_ent = results["baseline"]["entropy_harmless"]

    # ---- 1. single-layer sweep -----------------------------------------------------------
    print("\n=== single-layer sweep (reproduces the paper's probe) ===", flush=True)
    results["single"] = [measure({l}, f"single L{l}") for l in range(L)]
    d_single = np.array([base_p - r["p_refuse"] for r in results["single"]])

    # ---- 2. cumulative top-k by RCS ------------------------------------------------------
    order_rcs = list(np.argsort(-rcs_v))
    print(f"\n=== cumulative top-k by RCS (order {order_rcs[:8]}...) ===", flush=True)
    results["cumulative_rcs"] = [measure(set(order_rcs[:k]), f"top{k} RCS")
                                 for k in range(1, L + 1)]
    d_cum = np.array([base_p - r["p_refuse"] for r in results["cumulative_rcs"]])
    full = float(d_cum[-1])

    # ---- 3. matched-k selection rules ----------------------------------------------------
    k_grid = [int(x) for x in args.k_grid.split(",") if 0 < int(x) <= L]
    order_norm = list(np.argsort(-norm_v))
    print("\n=== matched-k selection rules ===", flush=True)
    sel = {"rcs_top": {}, "rcs_bottom": {}, "norm_top": {}, "random": {}}
    for k in k_grid:
        sel["rcs_top"][k] = {"delta": float(base_p - results["cumulative_rcs"][k - 1]["p_refuse"]),
                             "layers": sorted(int(x) for x in order_rcs[:k])}
        r = measure(set(order_rcs[-k:]), f"bottom{k} RCS")
        sel["rcs_bottom"][k] = {"delta": float(base_p - r["p_refuse"]), "rec": r}
        r = measure(set(order_norm[:k]), f"top{k} norm")
        sel["norm_top"][k] = {"delta": float(base_p - r["p_refuse"]), "rec": r}
        draws = []
        for j in range(args.n_random):
            S = rng.choice(L, size=k, replace=False)
            r = measure(set(S), f"random{k}.{j}")
            draws.append({"delta": float(base_p - r["p_refuse"]), "rec": r})
        d = [x["delta"] for x in draws]
        sel["random"][k] = {"draws": draws, "mean": float(np.mean(d)),
                            "std": float(np.std(d)), "min": float(np.min(d)),
                            "max": float(np.max(d))}
    results["selection"] = sel

    # ---- 4. restore guard ----------------------------------------------------------------
    print("\n=== restore guard ===", flush=True)
    results["restored"] = measure(set(), "restored")
    restore_dev = abs(results["restored"]["p_refuse"] - base_p)
    print(f"[guard] P_refuse returns to {results['restored']['p_refuse']:.6f} "
          f"against baseline {base_p:.6f}; |dev|={restore_dev:.2e}", flush=True)

    # ---- 5. statistics -------------------------------------------------------------------
    def layers_for(frac):
        """Smallest k whose cumulative effect reaches frac of the all-layer effect."""
        if full <= 0:
            return None
        hit = np.where(d_cum >= frac * full)[0]
        return int(hit[0] + 1) if len(hit) else None

    # guard: does this harness reproduce the paper's own single-layer probe? If the
    # single-layer arm matches logs/causal_refusalprob_<model>.json, the joint arms measured
    # in the same process are trustworthy on the same footing. A mismatch here would mean the
    # harness differs from the published one and nothing below could be compared to it.
    pub_guard = None
    pub_path = os.path.join(REPO, "logs", f"causal_refusalprob_{args.model}.json")
    if os.path.exists(pub_path):
        pub = json.load(open(pub_path))
        pdp = pub.get("delta_p", {})
        if len(pdp) == L:
            pv = np.array([pdp[str(l)] for l in range(L)])
            pub_guard = {
                "published_baseline_p_refuse": pub.get("baseline_p_refuse"),
                "our_baseline_p_refuse": base_p,
                "baseline_abs_dev": abs(pub.get("baseline_p_refuse", np.nan) - base_p),
                "delta_single_spearman_vs_published": float(stats.spearmanr(pv, d_single)[0]),
                "delta_single_max_abs_dev": float(np.max(np.abs(pv - d_single))),
                "published_spearman_rcs": pub.get("spearman_rcs"),
            }
            print("\n[guard] vs the paper's own single-layer probe:")
            for k, v in pub_guard.items():
                print(f"    {k}: {v}")

    sp = stats.spearmanr(rcs_v, d_single)
    spn = stats.spearmanr(norm_v, d_single)
    pr_rho, pr_p = partial_spearman(rcs_v, d_single, norm_v)
    add = {str(k): {"joint": float(d_cum[k - 1]),
                    "sum_of_singles": float(d_single[order_rcs[:k]].sum()),
                    "ratio": float(d_cum[k - 1] / d_single[order_rcs[:k]].sum())
                    if d_single[order_rcs[:k]].sum() != 0 else None}
           for k in k_grid}

    summary = {
        "baseline_p_refuse": base_p,
        "full_ablation_delta": full,
        "best_single_layer": int(np.argmax(d_single)),
        "best_single_delta": float(d_single.max()),
        "best_single_share_of_full": float(d_single.max() / full) if full > 0 else None,
        "layers_for_50pct": layers_for(0.5),
        "layers_for_80pct": layers_for(0.8),
        "layers_for_90pct": layers_for(0.9),
        "L": int(L),
        "spearman_rcs_vs_single": float(sp[0]), "spearman_rcs_vs_single_p": float(sp[1]),
        "spearman_norm_vs_single": float(spn[0]),
        "partial_rcs_given_norm": float(pr_rho), "partial_rcs_given_norm_p": float(pr_p),
        "additivity": add,
        "restore_guard_abs_dev": restore_dev,
        "baseline_harmless_entropy": base_h_ent,
        "max_harmless_entropy_over_configs": float(max(
            r["entropy_harmless"] for r in results["cumulative_rcs"])),
        "published_probe_guard": pub_guard,
    }
    print("\n================ JOINT CAUSAL SUMMARY ================")
    for k, v in summary.items():
        if k not in ("additivity",):
            print(f"  {k}: {v}")
    print(f"  additivity: {json.dumps(add)}")

    out = {"model": args.model, "reps": args.reps, "dataset": args.dataset,
           "metric": "refusal_token_mass_first_position", "alpha": 1.0,
           "n_harmful": len(harmful), "n_harmless_probe": len(harmless_probe),
           "batch": args.batch, "seed": args.seed, "k_grid": k_grid,
           "n_random": args.n_random, "refusal_token_id": refusal_ids,
           "rcs": rcs, "norms": norms, "rcs_order": [int(x) for x in order_rcs],
           "delta_single": d_single.tolist(), "delta_cumulative_rcs": d_cum.tolist(),
           "summary": summary, "configs": results,
           "host": os.uname().nodename, "device_map": max_memory}
    base = f"joint_causal_{args.model}{args.tag}"
    with open(os.path.join(OUT_DIR, base + ".json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(OUT_DIR, base + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "delta_rcs_top", "delta_rcs_bottom", "delta_norm_top",
                    "delta_random_mean", "delta_random_std", "delta_random_max",
                    "joint_over_sum_of_singles"])
        for k in k_grid:
            w.writerow([k, f"{sel['rcs_top'][k]['delta']:.4f}",
                        f"{sel['rcs_bottom'][k]['delta']:.4f}",
                        f"{sel['norm_top'][k]['delta']:.4f}",
                        f"{sel['random'][k]['mean']:.4f}",
                        f"{sel['random'][k]['std']:.4f}",
                        f"{sel['random'][k]['max']:.4f}",
                        "" if add[str(k)]["ratio"] is None else f"{add[str(k)]['ratio']:.4f}"])
    print(f"\nwrote {OUT_DIR}/{base}.{{json,csv}}  in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
