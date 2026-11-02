#!/usr/bin/env python3
"""Random-direction control for the joint ablation (anticipates the obvious objection).

`joint_causal_ablation.py` shows what ablating the refusal direction at k layers does. On its
own that cannot distinguish two explanations:

  (a) those layers carry refusal along that specific direction, or
  (b) a rank-1 projection applied at k layers degrades the model enough to depress refusal
      whatever direction is removed.

This script holds the layer set fixed at the top-k by RCS and swaps only the direction for a
random unit vector in R^d (one fresh draw per layer per replicate). `get_weighted_ablated_matrix`
normalises its input, so only orientation differs from the real run -- the magnitude of the
weight edit is matched by construction. If the real direction and a random one depress refusal
equally, the causal claim is about the layers at most, not about the direction, and the paper
cannot say RCS locates refusal.

Reads the real-direction deltas from joint_causal_<model>.json so the two are compared at
identical k and identical layer sets.

Writes results/joint_causal/random_direction_<model>.{json,csv}
"""
import argparse, csv, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from joint_causal_ablation import (REPO, OUT_DIR, Ablator, first_token_stats,
                                   pick_devices, shard_patch)
from lama_rcs import load_representations, compute_rcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2-7b")
    ap.add_argument("--reps", default="qwen2-7b_processed_confirm")
    ap.add_argument("--dataset", default="harmbench_test")
    ap.add_argument("--max_prompts", type=int, default=159)
    ap.add_argument("--n_harmless_probe", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n_draws", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--need_gib", type=float, default=17.0)
    args = ap.parse_args()

    real_path = os.path.join(OUT_DIR, f"joint_causal_{args.model}.json")
    assert os.path.exists(real_path), f"run joint_causal_ablation.py first ({real_path})"
    real = json.load(open(real_path))
    k_grid = real["k_grid"]
    order_rcs = real["rcs_order"]
    base_p_real = real["summary"]["baseline_p_refuse"]
    d_cum_real = real["delta_cumulative_rcs"]

    HF, HL = load_representations(args.reps)
    L, D = HF.shape[1], HF.shape[2]
    rcs, _ = compute_rcs(HF, HL)
    print(f"[rcs] L={L} D={D} peak=L{int(np.argmax([rcs[l] for l in range(L)]))}", flush=True)

    from dataset.load_dataset import load_dataset, load_dataset_split
    harmful = load_dataset(args.dataset, instructions_only=True)[:args.max_prompts]
    harmless_probe = [h for h in load_dataset_split("harmless", "train", instructions_only=True)
                      if h and h.strip()][:args.n_harmless_probe]

    devs, got = pick_devices(args.need_gib)
    if got < args.need_gib:
        print(f"[FATAL] only {got:.1f} GiB usable, need {args.need_gib}", flush=True)
        return 4
    max_memory = {i: f"{int(u)}GiB" for i, u in devs}
    print(f"[shard] {max_memory}", flush=True)
    shard_patch(max_memory)
    from models.load_models import load_model
    llm = load_model(args.model, device=f"cuda:{devs[0][0]}")
    refusal_ids = list(getattr(llm, "refusal_token_id", []) or [])

    rng = np.random.RandomState(args.seed)
    rows, out = [], {}

    # baseline in THIS process, so the comparison never crosses a process boundary
    ab = Ablator(llm, {})
    ab.set_layers(set())
    base_p, _, _ = first_token_stats(llm, harmful, refusal_ids, args.batch)
    print(f"[baseline] P_refuse={base_p:.4f} (joint run reported {base_p_real:.4f}, "
          f"|dev|={abs(base_p-base_p_real):.2e})", flush=True)

    for k in k_grid:
        layers = sorted(int(x) for x in order_rcs[:k])
        deltas = []
        for j in range(args.n_draws):
            dirs = {}
            for l in layers:
                v = torch.tensor(rng.randn(D), dtype=torch.float32)
                dirs[l] = v / v.norm()
            ab_r = Ablator(llm, dirs)
            ab_r.orig = ab.orig            # share the cached originals
            ab_r.set_layers(set(layers))
            pr, t1, en = first_token_stats(llm, harmful, refusal_ids, args.batch)
            _, h_t1, h_en = first_token_stats(llm, harmless_probe, refusal_ids, args.batch)
            ab_r.set_layers(set())         # restore before the next draw
            ab.orig = ab_r.orig
            deltas.append(base_p - pr)
            print(f"  [random dir k={k} draw {j}] P_refuse={pr:.4f} dP={base_p-pr:+.4f} "
                  f"H_harmless={h_en:.3f}", flush=True)
        real_d = d_cum_real[k - 1]
        out[str(k)] = {"layers": layers, "random_deltas": [float(x) for x in deltas],
                       "random_mean": float(np.mean(deltas)),
                       "random_std": float(np.std(deltas)),
                       "real_delta": float(real_d),
                       "ratio_real_over_random": float(real_d / np.mean(deltas))
                       if np.mean(deltas) != 0 else None}
        rows.append([k, f"{real_d:.4f}", f"{np.mean(deltas):.4f}", f"{np.std(deltas):.4f}",
                     "" if np.mean(deltas) == 0 else f"{real_d/np.mean(deltas):.2f}"])
        print(f"  => k={k}: real dP={real_d:.4f}  random dP={np.mean(deltas):.4f}"
              f" +- {np.std(deltas):.4f}", flush=True)

    res = {"model": args.model, "reps": args.reps, "dataset": args.dataset,
           "n_draws": args.n_draws, "seed": args.seed, "baseline_p_refuse": base_p,
           "baseline_p_refuse_joint_run": base_p_real,
           "baseline_abs_dev_across_processes": abs(base_p - base_p_real),
           "by_k": out, "host": os.uname().nodename}
    with open(os.path.join(OUT_DIR, f"random_direction_{args.model}.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(OUT_DIR, f"random_direction_{args.model}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "real_delta", "random_delta_mean", "random_delta_std",
                    "ratio_real_over_random"])
        w.writerows(rows)
    print(f"\nwrote {OUT_DIR}/random_direction_{args.model}.{{json,csv}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
