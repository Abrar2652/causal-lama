"""Phase 1 of causal validation: GENERATE per-layer completions only (no judge).

Decouples the two memory-heavy models: here only the 7B is ever resident
(gc.collect between layers -> ~16GB peak, no OOM). Phase 2 (recover_causal.py)
judges the saved completions with only the judge resident. Resumable: skips any
layer whose completion file already exists.

Usage: python generate_causal_completions.py <model> [dataset=harmbench_test]
"""
import os, sys, gc, glob
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from models.load_models import load_model
from directions_ablation import generate_and_save_hookfree_completions
from utils.ablation_utils import get_weighted_ablated_matrix
from lama_rcs import load_representations

MODEL = sys.argv[1]
DATASET = sys.argv[2] if len(sys.argv) > 2 else 'harmbench_test'
DEVICE = 'cuda:0'

HF, HL = load_representations(MODEL)
HF_t = torch.tensor(HF, dtype=torch.float32)
HL_t = torch.tensor(HL, dtype=torch.float32)
n_layers = HF.shape[1]
cfg = Config(model_alias=MODEL, model_path=MODEL)
cdir = os.path.join(cfg.artifact_path(), 'causal_validation')
os.makedirs(cdir, exist_ok=True)


def cpath(name):
    return os.path.join(cdir, f'{name}_{DATASET}_completions.json')


def gen(layer):
    name = 'rcs_causal_noablation' if layer is None else f'rcs_causal_layer{layer}'
    if os.path.exists(cpath(name)):
        print(f"[skip] {name} exists", flush=True)
        return
    llm = load_model(MODEL, DEVICE)
    if layer is not None:
        d = (HF_t[:, layer, :].mean(0) - HL_t[:, layer, :].mean(0))
        blocks = llm.model_block_modules
        if layer < len(blocks):
            o_proj, down_proj = llm._get_block_proj_weights(blocks[layer])
            o_proj.data = get_weighted_ablated_matrix(o_proj.data.T, d, alpha=1.0).T
            down_proj.data = get_weighted_ablated_matrix(down_proj.data.T, d, alpha=1.0).T
    generate_and_save_hookfree_completions(
        cfg=cfg, folder='causal_validation', model_base=llm,
        dataset_name=DATASET, aux_name=name, return_completions=False)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[done] {name}", flush=True)


gen(None)
for l in range(n_layers):
    gen(l)
print(f"GENERATION COMPLETE for {MODEL} ({DATASET}): "
      f"{len(glob.glob(cpath('rcs_causal_layer*').replace('layer*','layer*')))} layer files", flush=True)
