"""Recover the causal-validation result from already-saved per-layer completions.

The live run generated all 28 layers' completions but the per-layer ASR step was
skipped by a torch.cuda.empty_cache() NVML crash (broken driver), so ΔASR
defaulted to 0. Completions are intact — we just judge them here (one cached judge
load) and compute the RCS↔ΔASR correlation + the non-circularity partial corr.
"""
import os, sys, json, glob
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lama_rcs import (load_representations, compute_rcs, compute_direction_norms,
                      partial_spearman)
from eval_jailbreaks import harmbench_judge_fn_hf

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen2-7b'
DATASET = sys.argv[2] if len(sys.argv) > 2 else 'harmbench_test'
CDIR = f'runs/{MODEL}/causal_validation'
CACHE = f'logs/causal_asr_cache_{MODEL}_{DATASET}.json'  # resumable: 'noabl'/'L<idx>' -> ASR

def load_cache():
    return json.load(open(CACHE)) if os.path.exists(CACHE) else {}

def save_cache(c):
    json.dump(c, open(CACHE, 'w'), indent=2)

def asr_of(path):
    comps = json.load(open(path))
    cls, ninv = harmbench_judge_fn_hf([c['prompt'] for c in comps],
                                      [c['response'] for c in comps], device='cuda:0')
    return float(np.mean(cls))

# RCS + control covariate (CPU, always safe)
HF, HL = load_representations(MODEL)
rcs, _ = compute_rcs(HF, HL)
norms = compute_direction_norms(HF, HL)

cache = load_cache()
# baseline (no ablation)
if 'noabl' not in cache:
    nofile = f'{CDIR}/rcs_causal_noablation_{DATASET}_completions.json'
    cache['noabl'] = asr_of(nofile); save_cache(cache)
base_asr = cache['noabl']
print(f"baseline (no-ablation) ASR = {base_asr:.4f}")

# judge any not-yet-cached layers (resumable across crashes)
remaining = [l for l in sorted(rcs)
             if f'L{l}' not in cache
             and os.path.exists(f'{CDIR}/rcs_causal_layer{l}_{DATASET}_completions.json')]
print(f"layers to judge this run: {remaining}")
for l in remaining:
    a = asr_of(f'{CDIR}/rcs_causal_layer{l}_{DATASET}_completions.json')
    cache[f'L{l}'] = a; save_cache(cache)
    print(f"  layer {l:>2}  RCS={rcs[l]:.3f}  ASR={a:.4f}  ΔASR={a-base_asr:+.4f}", flush=True)

# assemble from cache
raw = {l: cache[f'L{l}'] for l in sorted(rcs) if f'L{l}' in cache}
delta = {l: raw[l] - base_asr for l in raw}
missing = [l for l in sorted(rcs) if f'L{l}' not in cache]
if missing:
    print(f"INCOMPLETE: still need layers {missing} — re-run to resume."); sys.exit(2)

L = sorted(delta)
rcs_v = np.array([rcs[l] for l in L])
d_v = np.array([delta[l] for l in L])
nm_v = np.array([norms[l] for l in L])

sp = stats.spearmanr(rcs_v, d_v)
spn = stats.spearmanr(nm_v, d_v)
pr_rho, pr_p = partial_spearman(rcs_v, d_v, nm_v)
# bootstrap CI for spearman(rcs,delta) (guard degenerate)
rng = np.random.RandomState(42)
boot = []
for _ in range(2000):
    idx = rng.randint(0, len(L), len(L))
    if len(set(d_v[idx])) < 2 or len(set(rcs_v[idx])) < 2:
        continue
    boot.append(stats.spearmanr(rcs_v[idx], d_v[idx])[0])
boot = np.array([b for b in boot if not np.isnan(b)])
ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if len(boot) else (float('nan'),)*2

print("\n================ CAUSAL VALIDATION (recovered) ================")
print(f"Spearman(RCS, ΔASR)        = {sp[0]:.4f}  (p={sp[1]:.2e})  95%CI[{ci[0]:.3f},{ci[1]:.3f}]")
print(f"Spearman(norm, ΔASR)       = {spn[0]:.4f}  (p={spn[1]:.2e})")
print(f"partial(RCS,ΔASR | norm)   = {pr_rho:.4f}  (p={pr_p:.2e})   <-- NON-CIRCULARITY")
print(f"baseline ASR={base_asr:.3f}  ΔASR range [{d_v.min():+.3f},{d_v.max():+.3f}]")
verdict = "PASS ρ≥0.7" if sp[0] >= 0.7 else ("MODERATE" if sp[0] >= 0.5 else "WEAK")
print(f"VERDICT: {verdict}; non-circularity {'survives' if pr_rho>=0.4 else 'WEAK'}")

out = {'model': MODEL, 'baseline_asr': base_asr, 'raw_asr': raw, 'delta_asr': delta,
       'rcs': rcs, 'direction_norms': norms,
       'spearman_rcs_dasr': float(sp[0]), 'spearman_rcs_dasr_p': float(sp[1]),
       'spearman_rcs_dasr_ci': ci,
       'spearman_norm_dasr': float(spn[0]),
       'partial_rcs_dasr_given_norm': pr_rho, 'partial_p': pr_p}
_outp = f'logs/causal_recovered_{MODEL}_{DATASET}.json'
json.dump(out, open(_outp, 'w'), indent=2)
print(f"saved -> {_outp}")
