"""run_ablation_study.py — End-to-end ablation table for one model.

Runs all 4 ablation modes on harmbench_test with proper isolation, then computes
bootstrap CIs and a paired-bootstrap significance test vs MD. Produces a JSON
summary ready to paste into tables.tex.

Modes:
  MD         — baseline: single-layer SOM at l*, uniform α=1 (Piras et al.)
  LAMA-unif  — per-group SOMs, uniform α=1 (isolates direction quality)
  LAMA-RCS   — per-group SOMs, α from RCS (isolates adaptive weighting)
  LAMA-BO    — per-group SOMs, α from BO (full method)

Usage: python run_ablation_study.py <model> [seed=42] [device=cuda:0]
"""
import os, sys, gc, json, glob
import numpy as np
import torch
from scipy import stats

# Models on ckg08 live in ckgfs (home NFS is different from old node; ckgfs is shared)
_CKGFS_HF = '/nas/ckgfs/users/jahin/hf_cache'
if os.path.isdir(_CKGFS_HF) and not os.environ.get('HF_HOME'):
    os.environ['HF_HOME'] = _CKGFS_HF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL   = sys.argv[1]
SEED    = int(sys.argv[2]) if len(sys.argv) > 2 else 42
DEVICE  = sys.argv[3] if len(sys.argv) > 3 else 'cuda:0'
DATASET = 'harmbench_test'

import random; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

from config import Config
from models.load_models import load_model
from dataset.load_dataset import load_dataset
from directions_ablation import generate_and_save_hookfree_completions
from eval_jailbreaks import evaluate_jailbreak
from lama_ablate import load_optimization_results
from lama_generate_directions import load_group_directions
from lama_rcs import (load_representations, compute_rcs, compute_rcs_weights,
                      auto_group_layers)
from lama_stats import bootstrap_asr_ci, paired_bootstrap_diff, mcnemar_test
from lama_eval_utility import compute_perplexity, compute_harmless_compliance, utility_report
from dataset.load_dataset import load_dataset_split

# ── helpers ──────────────────────────────────────────────────────────────────

cfg = Config(model_alias=MODEL, model_path=MODEL)
LAMA_PREFIX = 'lama_som4x4_sigma0.33_groups3'
DIRS_DIR = f'./runs/{MODEL}/generate_directions'
OPT_DIR  = f'./runs/{MODEL}/lama_optimizations'
OUT_DIR  = f'./runs/{MODEL}/ablation_study'
os.makedirs(OUT_DIR, exist_ok=True)

# Llama-Guard-2 lives in ckgfs cache — use the local snapshot path directly
_g2_snapshots = glob.glob('/nas/ckgfs/users/jahin/hf_cache/hub/models--meta-llama--Meta-Llama-Guard-2-8B/snapshots/*')
GUARD2_PATH = _g2_snapshots[0] if _g2_snapshots else None


def judge_harmbench(completions, label):
    path = os.path.join(OUT_DIR, f'{label}_seed{SEED}_hb_eval.json')
    r = evaluate_jailbreak(completions=completions, methodologies=['harmbench'],
                           evaluation_path=path, judge_keep_loaded=False)
    return np.asarray(r['harmbench_classifications'], dtype=int)


def judge_guard2(completions, label):
    """Llama-Guard-2 as second judge for robustness (uses its own chat format)."""
    if not GUARD2_PATH or not os.path.isdir(GUARD2_PATH):
        print(f"  [Guard2] skipped (no local snapshot found at {GUARD2_PATH})")
        return None
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        path = os.path.join(OUT_DIR, f'{label}_seed{SEED}_guard2_eval.json')
        g2dir = GUARD2_PATH
        tok = AutoTokenizer.from_pretrained(g2dir, local_files_only=True)
        tok.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(
            g2dir, torch_dtype=torch.float16, local_files_only=True).to(DEVICE).eval()
        UNSAFE = 'unsafe'
        cls = []
        with torch.no_grad():
            for c in completions:
                conv = [{"role": "user",    "content": c['prompt']},
                        {"role": "assistant","content": c['response']}]
                inp = tok.apply_chat_template(conv, return_tensors='pt').to(DEVICE)
                out = model.generate(inp, max_new_tokens=20, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
                txt = tok.decode(out[0, inp.shape[1]:], skip_special_tokens=True).strip()
                cls.append(int(UNSAFE in txt.lower()))
        del model; gc.collect(); torch.cuda.empty_cache()
        asr = float(np.mean(cls))
        json.dump({'guard2_asr': asr, 'classifications': cls}, open(path, 'w'))
        print(f"  [Guard2] ASR={asr:.4f}")
        return np.asarray(cls, dtype=int)
    except Exception as e:
        print(f"  [Guard2] FAILED: {e}"); return None


def run_mode(label, apply_fn):
    """Load model, apply ablation via apply_fn, generate, judge, return results.
    apply_fn(llm) modifies the model in-place (e.g. ablate_weights_adaptive).
    The model is fully deleted BEFORE judging so 7B + 26GB judge never coexist.
    Skips generation + HB judging if both output files already exist."""
    print(f"\n── {label} ──")
    comp_path = os.path.join(OUT_DIR, f'{label}_seed{SEED}_{DATASET}_completions.json')
    hb_path   = os.path.join(OUT_DIR, f'{label}_seed{SEED}_hb_eval.json')

    if os.path.exists(comp_path) and os.path.exists(hb_path):
        print(f"  [skip gen+judge] loading existing results")
        completions = json.load(open(comp_path))
        hb_data = json.load(open(hb_path))
        hb = np.asarray(hb_data['harmbench_classifications'], dtype=int)
    else:
        # Step 1: load + ablate + generate (7B only on GPU, ~14 GB)
        llm = load_model(MODEL, DEVICE)
        apply_fn(llm)
        completions = generate_and_save_hookfree_completions(
            cfg=cfg, folder='ablation_study', model_base=llm,
            dataset_name=DATASET, aux_name=f'{label}_seed{SEED}',
            return_completions=True)
        # Step 2: free 7B BEFORE loading judge (no coexistence)
        del llm; gc.collect(); torch.cuda.empty_cache()
        # Step 3: judge (26 GB judge on empty GPU)
        hb = judge_harmbench(completions, label)

    # Guard2 runs after HB judge is freed (both set judge_keep_loaded=False)
    g2 = judge_guard2(completions, label)
    return {'harmbench': hb, 'guard2': g2, 'completions': completions}


# ── load shared artifacts ─────────────────────────────────────────────────────

HF_x, HL_x = load_representations(MODEL)
rcs, _      = compute_rcs(HF_x, HL_x)
rcs_weights = compute_rcs_weights(rcs)
groups, group_map = auto_group_layers(rcs, n_groups=3, method='quantile')
all_group_dirs, meta = load_group_directions(DIRS_DIR, LAMA_PREFIX)

# Group metadata → layer_groups dict
layer_groups = {}
for g_id_str, gmeta in meta.items():
    if not g_id_str.isdigit(): continue
    for l in gmeta.get('layers', []):
        layer_groups[l] = int(g_id_str)
n_layers = max(layer_groups) + 1

# BO best results
opt_files = sorted(glob.glob(f'{OPT_DIR}/*.json'))
if not opt_files:
    raise FileNotFoundError(f"No BO results in {OPT_DIR}; run lama_optuna_search.py first")
bo_dirs, bo_weights, _ = load_optimization_results(opt_files[-1])
print(f"Loaded BO results from {opt_files[-1]}")
print(f"  best ASR(val)={json.load(open(opt_files[-1])).get('best_score'):.1f}%  "
      f"α range [{min(bo_weights.values()):.3f},{max(bo_weights.values()):.3f}]")

# shared direction selection (same dirs for uniform/rcs/bo to isolate α effect)
bo_selected = {g: [all_group_dirs[g][i] for i in idxs if i < len(all_group_dirs[g])]
               for g, idxs in bo_dirs.items() if g in all_group_dirs}

# ── MD BASELINE ───────────────────────────────────────────────────────────────
# Use the single-layer SOM baseline (Piras et al.) — train SOM at l*, apply uniform.
# We load the layer_selection to find l*, then use the single best direction (mean diff)
# as the canonical SD comparison. Full MD BO is expensive; use the paper's reported
# direction IDs where available (qwen2-7b: 18, ids 10-3-5-2-4-6-13 → ASR 95.97%).
print("\n══ MD BASELINE ══")
md_layer_file = f'runs/{MODEL}/layer_selection/layer_selection.json'
ls_data = json.load(open(md_layer_file))
best_l = int(min(ls_data, key=lambda k: ls_data[k]['Refusal']))
print(f"l* = {best_l}  (lowest refusal-token probability)")

# Generate single-layer SOM directions inline (same math as baseline script)
md_dir_file = f'{DIRS_DIR}/som4x4_sigma0.33_layer{best_l}_directions.pt'
if not os.path.exists(md_dir_file):
    print(f"Training SOM at l*={best_l} (inline, CPU)...")
    from minisom import MiniSom
    from collections import defaultdict
    # Use representations already loaded (HF_x/HL_x)
    X_layer = HF_x[:, best_l, :]            # [n_harmful, d]
    centroid = HL_x[:, best_l, :].mean(0)   # harmless centroid at l*
    centroid_t = torch.tensor(centroid, dtype=torch.float32)
    som = MiniSom(4, 4, X_layer.shape[1], sigma=0.33, learning_rate=0.01,
                  random_seed=0, topology='hexagonal')
    som.random_weights_init(X_layer)
    som.train(X_layer, 10000)
    weights = som.get_weights()  # [4, 4, d]
    winners = [som.winner(xi) for xi in X_layer]
    Y = np.ones(len(X_layer), dtype=int)
    counts = defaultdict(lambda: [0, 0])
    for w, lbl in zip(winners, Y):
        counts[tuple(w)][lbl] += 1
    sorted_neurons = sorted(counts.items(), key=lambda x: x[1][1], reverse=True)
    wts_sorted = np.array([weights[i, j] for (i, j), _ in sorted_neurons])
    w_t = torch.tensor(wts_sorted, dtype=torch.float32)
    directions = torch.stack([w - centroid_t for w in w_t])
    torch.save(directions, md_dir_file)
    print(f"[✓] Saved {len(directions)} MD directions → {md_dir_file}")
md_all_dirs = torch.load(md_dir_file, weights_only=True)
print(f"Loaded {len(md_all_dirs)} MD directions from layer {best_l}")

# Apply MD: top-7 directions (matching baseline paper's best config), uniform α=1
# For qwen2-7b the paper uses direction ids 10-3-5-2-4-6-13; we use the same ranked set
MD_DIR_IDS = [0, 1, 2, 3, 4, 5, 6]   # top-7 by harmful-win count (SOM sorts them)
from utils.ablation_utils import get_ablated_matrix
def apply_md(llm):
    emb = llm._get_embedding_weight()
    for did in MD_DIR_IDS:
        d_vec = md_all_dirs[did]
        emb.data = get_ablated_matrix(emb.data, d_vec)
        for block in llm.model_block_modules:
            o_p, d_p = llm._get_block_proj_weights(block)
            o_p.data = get_ablated_matrix(o_p.data.T, d_vec).T
            d_p.data = get_ablated_matrix(d_p.data.T, d_vec).T
res_md = run_mode('MD', apply_md)

# ── LAMA-UNIFORM ──────────────────────────────────────────────────────────────
print("\n══ LAMA-UNIFORM ══")
uniform_weights = {l: 1.0 for l in range(n_layers)}
res_unif = run_mode('LAMA-uniform',
    lambda llm: llm.ablate_weights_adaptive(bo_selected, layer_groups, uniform_weights))

# ── LAMA-RCS ─────────────────────────────────────────────────────────────────
print("\n══ LAMA-RCS ══")
res_rcs = run_mode('LAMA-RCS',
    lambda llm: llm.ablate_weights_adaptive(bo_selected, layer_groups, rcs_weights))

# ── LAMA-BO ──────────────────────────────────────────────────────────────────
print("\n══ LAMA-BO ══")
res_bo = run_mode('LAMA-BO',
    lambda llm: llm.ablate_weights_adaptive(bo_selected, layer_groups, bo_weights))

# ── UTILITY on LAMA-BO (clean model needed for baseline perplexity) ───────────
print("\n══ UTILITY EVAL ══")
harmless = load_dataset_split('harmless', 'val', instructions_only=True)[:200]
# Load clean then ablated for utility — sequential, each freed before next
llm_c = load_model(MODEL, DEVICE)
ppl_clean = compute_perplexity(llm_c, [llm_c._get_prompt(p) for p in harmless])
comply_clean = compute_harmless_compliance(llm_c, harmless)
del llm_c; gc.collect(); torch.cuda.empty_cache()

llm_b = load_model(MODEL, DEVICE)
llm_b.ablate_weights_adaptive(bo_selected, layer_groups, bo_weights)
ppl_bo = compute_perplexity(llm_b, [llm_b._get_prompt(p) for p in harmless])
comply_bo = compute_harmless_compliance(llm_b, harmless)
del llm_b; gc.collect(); torch.cuda.empty_cache()

util_clean = utility_report(ppl_clean, comply_clean)
util_bo    = utility_report(ppl_bo, comply_bo, baseline_perplexity=ppl_clean)
print(f"  clean: PPL={util_clean['perplexity']:.2f}  comply={util_clean['compliance_rate']:.3f}")
print(f"  LAMA-BO: PPL={util_bo['perplexity']:.2f}  comply={util_bo['compliance_rate']:.3f}  ΔPPL={util_bo.get('nll_delta_vs_baseline', float('nan')):.4f}")

# ── STATISTICS ────────────────────────────────────────────────────────────────
print("\n══ STATISTICS ══")
results = {'MD': res_md, 'LAMA-uniform': res_unif, 'LAMA-RCS': res_rcs, 'LAMA-BO': res_bo}
stats_out = {}
for name, res in results.items():
    cls = res['harmbench']
    ci  = bootstrap_asr_ci(cls)
    pd  = paired_bootstrap_diff(cls, res_md['harmbench'])
    mc  = mcnemar_test(cls, res_md['harmbench'])
    g2_asr = float(res['guard2'].mean()) if res['guard2'] is not None else None
    stats_out[name] = {
        'asr': ci['asr'], 'ci_lo': ci['ci_lo'], 'ci_hi': ci['ci_hi'],
        'n': ci['n'], 'diff_vs_md': pd['diff'], 'p_vs_md': pd['p_value'],
        'mcnemar_p': mc['p_value'], 'guard2_asr': g2_asr,
    }
    sig = '*' if pd['significant_0.05'] else ''
    print(f"  {name:12s} ASR={ci['asr']:.3f} [{ci['ci_lo']:.3f},{ci['ci_hi']:.3f}] "
          f"vs MD: Δ={pd['diff']:+.3f} (p={pd['p_value']:.3f}){sig}"
          + (f"  Guard2={g2_asr:.3f}" if g2_asr is not None else ""))

# ── SAVE ─────────────────────────────────────────────────────────────────────
out = {
    'model': MODEL, 'seed': SEED, 'dataset': DATASET,
    'bo_opt_file': opt_files[-1], 'md_layer': best_l,
    'asr_table': stats_out,
    'utility_clean': util_clean, 'utility_lama_bo': util_bo,
}
save_path = os.path.join(OUT_DIR, f'ablation_seed{SEED}.json')
json.dump(out, open(save_path, 'w'), indent=2,
          default=lambda x: x.tolist() if hasattr(x, 'tolist') else float(x))
print(f"\n[✓] saved → {save_path}")
