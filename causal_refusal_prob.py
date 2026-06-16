"""Graded causal validation: RCS vs Δ refusal-token probability.

Binary ASR has no signal on robustly-aligned models (gemma2/llama2 stay at ASR~0
under single-layer ablation). Following Arditi et al.'s layer-selection criterion,
we measure the *probability mass on refusal tokens* at the first generated
position, and how much single-layer ablation REDUCES it:

    DeltaP(l) = P_refuse(no ablation) - P_refuse(ablate layer l)

This is continuous (signal even when ASR=0), needs no judge, and is cheap:
ONE model load, forward-only, with per-layer weight save/restore (no reload, no
OOM). We then correlate RCS(l) with DeltaP(l) and report the non-circularity
partial correlation controlling for the diff-in-means norm.

Usage: python causal_refusal_prob.py <model> [dataset=harmbench_test] [max_prompts=159]
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from models.load_models import load_model
from utils.ablation_utils import get_weighted_ablated_matrix
from dataset.load_dataset import load_dataset
from lama_rcs import (load_representations, compute_rcs, compute_direction_norms,
                      partial_spearman)
from scipy import stats

MODEL = sys.argv[1]
DATASET = sys.argv[2] if len(sys.argv) > 2 else 'harmbench_test'
MAXP = int(sys.argv[3]) if len(sys.argv) > 3 else 159
DEVICE = 'cuda:0'
BATCH = 16


@torch.no_grad()
def mean_refusal_prob(llm, prompts, refusal_ids):
    """Mean P(refusal tokens) at the first generated position over harmful prompts."""
    tok = llm.tokenizer
    tok.padding_side = 'left'   # so the last real token is always at index -1
    probs = []
    for i in range(0, len(prompts), BATCH):
        batch = [llm._get_prompt(p) for p in prompts[i:i+BATCH]]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True,
                  max_length=1024).to(llm.model.device)
        out = llm.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        logits = out.logits[:, -1, :].float()   # left-padded: last token at -1
        p = F.softmax(logits, dim=-1)
        pref = p[:, refusal_ids].sum(-1)  # mass on refusal tokens
        probs.extend(pref.cpu().tolist())
    return float(np.mean(probs))


def main():
    HF, HL = load_representations(MODEL)
    rcs, _ = compute_rcs(HF, HL)
    norms = compute_direction_norms(HF, HL)
    HF_t = torch.tensor(HF, dtype=torch.float32)
    HL_t = torch.tensor(HL, dtype=torch.float32)
    n_layers = HF.shape[1]

    prompts = load_dataset(DATASET, instructions_only=True)[:MAXP]
    llm = load_model(MODEL, DEVICE)
    refusal_ids = list(getattr(llm, 'refusal_token_id', []) or [])
    if not refusal_ids:
        print(f"WARNING: {MODEL} has no refusal_token_id; cannot run prob metric.")
        return
    print(f"refusal_token_id={refusal_ids}  prompts={len(prompts)}")

    base_p = mean_refusal_prob(llm, prompts, refusal_ids)
    print(f"baseline P_refuse (no ablation) = {base_p:.4f}")

    blocks = llm.model_block_modules
    deltas = {}
    for l in range(min(n_layers, len(blocks))):
        d = (HF_t[:, l, :].mean(0) - HL_t[:, l, :].mean(0))
        o_proj, down_proj = llm._get_block_proj_weights(blocks[l])
        o_orig = o_proj.data.clone()
        d_orig = down_proj.data.clone()
        o_proj.data = get_weighted_ablated_matrix(o_proj.data.T, d, alpha=1.0).T
        down_proj.data = get_weighted_ablated_matrix(down_proj.data.T, d, alpha=1.0).T
        p = mean_refusal_prob(llm, prompts, refusal_ids)
        deltas[l] = base_p - p          # drop in refusal prob = causal effect
        o_proj.data = o_orig            # restore (no reload needed)
        down_proj.data = d_orig
        print(f"  layer {l:>2}  RCS={rcs[l]:.3f}  P_refuse={p:.4f}  ΔP={deltas[l]:+.4f}", flush=True)

    L = sorted(deltas)
    rcs_v = np.array([rcs[l] for l in L])
    dp_v = np.array([deltas[l] for l in L])
    nm_v = np.array([norms[l] for l in L])
    sp = stats.spearmanr(rcs_v, dp_v)
    spn = stats.spearmanr(nm_v, dp_v)
    pr_rho, pr_p = partial_spearman(rcs_v, dp_v, nm_v)
    # bootstrap CI
    rng = np.random.RandomState(42); boot = []
    for _ in range(2000):
        idx = rng.randint(0, len(L), len(L))
        if len(set(dp_v[idx])) > 1 and len(set(rcs_v[idx])) > 1:
            r = stats.spearmanr(rcs_v[idx], dp_v[idx])[0]
            if not np.isnan(r):
                boot.append(r)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (float('nan'),)*2

    print("\n================ CAUSAL (refusal-prob) ================")
    print(f"Spearman(RCS, ΔP_refuse)      = {sp[0]:.4f}  (p={sp[1]:.2e})  95%CI[{ci[0]:.3f},{ci[1]:.3f}]")
    print(f"Spearman(norm, ΔP_refuse)     = {spn[0]:.4f}")
    print(f"partial(RCS,ΔP | norm)        = {pr_rho:.4f}  (p={pr_p:.2e})  <-- NON-CIRCULARITY")
    print(f"baseline P_refuse={base_p:.3f}  ΔP range [{dp_v.min():+.3f},{dp_v.max():+.3f}]")
    v = 'PASS ρ≥0.7' if sp[0] >= 0.7 else ('MODERATE' if sp[0] >= 0.5 else 'WEAK')
    print(f"VERDICT: {v}; non-circularity {'survives' if pr_rho>=0.4 else 'weak'}")

    out = {'model': MODEL, 'metric': 'refusal_prob', 'dataset': DATASET,
           'baseline_p_refuse': base_p, 'rcs': rcs, 'delta_p': deltas, 'norms': norms,
           'spearman_rcs': float(sp[0]), 'spearman_rcs_p': float(sp[1]), 'ci': ci,
           'spearman_norm': float(spn[0]), 'partial_rcs_given_norm': pr_rho}
    json.dump(out, open(f'logs/causal_refusalprob_{MODEL}.json', 'w'), indent=2)
    print(f"saved -> logs/causal_refusalprob_{MODEL}.json")


if __name__ == '__main__':
    main()
