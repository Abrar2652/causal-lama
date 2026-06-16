"""
lama_optuna_search.py — Joint Bayesian Optimization for LAMA

Extends the original optuna_search.py to jointly optimize:
  1. Direction selection: which k_g directions per group
  2. Layer weights: α_l for each layer (ablation strength)

Key differences from original:
  - Integer params: direction indices per group (not global)
  - Float params: α_l per layer, initialized from RCS
  - RCS-guided initialization enables faster convergence
  - Same evaluation pipeline: load model → ablate → generate → judge
  - Directions pre-loaded once (not per trial)
"""

from __future__ import annotations

import argparse
import json
import os
import gc
import pickle
import random
from functools import partial
from typing import List, Dict, Optional

import numpy as np
import optuna
import torch
from optuna.distributions import IntDistribution, FloatDistribution

from models.load_models import load_model
from directions_ablation import generate_and_save_hookfree_completions
from config import Config
from eval_jailbreaks import evaluate_jailbreak
from lama_generate_directions import load_group_directions
from lama_rcs import compute_rcs, compute_rcs_weights, auto_group_layers, load_representations


# --------------------------------------------------------------------------- #
#  Seed setting for reproducibility                                            #
# --------------------------------------------------------------------------- #
def set_seeds(seed=42):
    """Set seeds for reproducibility across all random sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
#  LAMA evaluation function                                                    #
# --------------------------------------------------------------------------- #
def evaluate_lama_attack(
    dir_indices_per_group: Dict[int, List[int]],
    layer_weights: Dict[int, float],
    model_name: str,
    preloaded_group_directions: Dict[int, torch.Tensor],
    group_metadata: dict,
    layer_groups: Dict[int, int],
    dataset_name: str,
    device: str,
    aux_name: str,
    eval_path: str,
) -> float:
    """Evaluate a LAMA ablation configuration.
    
    Args:
        dir_indices_per_group: {group_id: [dir_idx_1, dir_idx_2, ...]}
        layer_weights: {layer_idx: alpha_l}
        model_name: model to load
        preloaded_group_directions: pre-loaded group directions (avoids per-trial disk I/O)
        group_metadata: metadata dict
        layer_groups: pre-built {layer_idx: group_id} mapping
        dataset_name: evaluation dataset
        device: torch device
        aux_name: naming prefix
        eval_path: where to save evaluation results
    
    Returns:
        asr: float, attack success rate in [0, 100] (percentage)
    """
    # Load model
    llm = load_model(model_name, device)
    cfg = Config(model_alias=model_name, model_path=model_name)
    
    # Build the selected directions per group from pre-loaded data
    selected_group_directions = {}
    for g_id, dir_indices in dir_indices_per_group.items():
        if g_id in preloaded_group_directions:
            all_dirs = preloaded_group_directions[g_id]
            valid_indices = [i for i in dir_indices if i < len(all_dirs)]
            if valid_indices:
                selected_group_directions[g_id] = [all_dirs[i] for i in valid_indices]
    
    # Apply adaptive ablation
    llm.ablate_weights_adaptive(selected_group_directions, layer_groups, layer_weights)
    
    # Build description string for naming
    dir_desc = '_'.join(
        f"g{g}:{'-'.join(map(str, dirs))}"
        for g, dirs in sorted(dir_indices_per_group.items())
    )
    local_aux = f"{aux_name}_{dir_desc}"
    
    # Generate completions
    completions = generate_and_save_hookfree_completions(
        cfg=cfg,
        folder='lama_val_optim_completions',
        model_base=llm,
        dataset_name=dataset_name,
        aux_name=local_aux,
        return_completions=True,
    )
    
    # Cleanup — gc.collect() REQUIRED before empty_cache (nn.Module circular refs);
    # else the 7B accumulates across BO trials and OOMs. judge_keep_loaded=False so
    # the judge never coexists with the next trial's model on one GPU.
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate
    os.makedirs(eval_path, exist_ok=True)
    result = evaluate_jailbreak(
        completions=completions,
        methodologies=['harmbench'],
        evaluation_path=os.path.join(eval_path, f'{local_aux}_results.json'),
        judge_keep_loaded=False,
    )
    
    # Return ASR as percentage (evaluate_jailbreak returns 0-1)
    return float(result['harmbench_success_rate']) * 100.0


# --------------------------------------------------------------------------- #
#  Optuna objective                                                            #
# --------------------------------------------------------------------------- #
def _lama_objective(
    trial: optuna.Trial,
    *,
    n_groups: int,
    dirs_per_group: int,
    max_dir_idx_per_group: Dict[int, int],
    n_layers: int,
    rcs_alpha_init: Dict[int, float],
    layer_to_group: Dict[int, int],
    model_name: str,
    preloaded_group_directions: Dict[int, torch.Tensor],
    group_metadata: dict,
    layer_groups: Dict[int, int],
    dataset_name: str,
    device: str,
    aux_name: str,
    eval_path: str,
    optimize_alpha: bool = True,
) -> float:
    """Optuna objective for LAMA: jointly optimize directions + layer weights."""
    
    # Sample direction indices per group
    dir_indices_per_group = {}
    for g_id in range(n_groups):
        max_idx = max_dir_idx_per_group.get(g_id, 15)
        group_dirs = []
        seen = set()
        for d in range(dirs_per_group):
            idx = trial.suggest_int(f"g{g_id}_dir{d}", 0, max_idx)
            # Ensure uniqueness within group
            while idx in seen and len(seen) < max_idx + 1:
                idx = (idx + 1) % (max_idx + 1)
            seen.add(idx)
            group_dirs.append(idx)
        dir_indices_per_group[g_id] = group_dirs
    
    # Sample or use fixed layer weights
    layer_weights = {}
    if optimize_alpha:
        for l in range(n_layers):
            init = rcs_alpha_init.get(l, 0.5)
            # Suggest alpha with RCS-guided initialization via bounds
            low = max(0.0, init - 0.4)
            high = min(1.0, init + 0.4)
            alpha = trial.suggest_float(f"alpha_{l}", low, high)
            layer_weights[l] = alpha
    else:
        layer_weights = dict(rcs_alpha_init)
    
    return evaluate_lama_attack(
        dir_indices_per_group=dir_indices_per_group,
        layer_weights=layer_weights,
        model_name=model_name,
        preloaded_group_directions=preloaded_group_directions,
        group_metadata=group_metadata,
        layer_groups=layer_groups,
        dataset_name=dataset_name,
        device=device,
        aux_name=aux_name,
        eval_path=eval_path,
    )


# --------------------------------------------------------------------------- #
#  Main optimization loop                                                      #
# --------------------------------------------------------------------------- #
def run_lama_optimization(args):
    """Run the full LAMA Bayesian Optimization search.
    
    Args:
        args: namespace with fields: model_name, directions_dir, directions_prefix,
              rcs_path, dataset_name, device, trials, dirs_per_group, optimize_alpha, seed
    
    Returns:
        study: optuna Study object
        best_summary: dict with best parameters, ASR, trial history
    """
    # Reproducibility
    seed = getattr(args, 'seed', 42)
    set_seeds(seed)
    
    # Load RCS analysis
    if hasattr(args, 'rcs_path') and args.rcs_path and os.path.exists(args.rcs_path):
        with open(args.rcs_path, 'r') as f:
            rcs_data = json.load(f)
        rcs_scores = {int(k): v for k, v in rcs_data['rcs_scores'].items()}
        n_layers = len(rcs_scores)
    else:
        HF_x, HL_x = load_representations(args.model_name)
        rcs_scores, _ = compute_rcs(HF_x, HL_x)
        n_layers = len(rcs_scores)
    
    # Compute initial alpha weights from RCS
    rcs_alpha_init = compute_rcs_weights(rcs_scores)
    
    # Load group metadata
    meta_path = os.path.join(args.directions_dir, f'{args.directions_prefix}_metadata.json')
    with open(meta_path, 'r') as f:
        group_metadata = json.load(f)
    
    # Pre-load group directions ONCE (not per trial)
    preloaded_group_directions, _ = load_group_directions(
        args.directions_dir, args.directions_prefix
    )
    
    # Determine max direction index per group and build layer→group mapping
    max_dir_idx_per_group = {}
    layer_to_group = {}
    layer_groups = {}
    n_groups = 0
    for g_id_str, meta in group_metadata.items():
        if g_id_str in ('saved_paths',):
            continue
        g_id = int(g_id_str)
        n_groups = max(n_groups, g_id + 1)
        max_dir_idx_per_group[g_id] = meta.get('n_directions', 16) - 1
        for layer in meta.get('layers', []):
            layer_to_group[layer] = g_id
            layer_groups[layer] = g_id
    
    print(f"\n{'='*60}")
    print(f"LAMA Bayesian Optimization")
    print(f"  Model: {args.model_name}")
    print(f"  Groups: {n_groups}")
    print(f"  Dirs per group: {args.dirs_per_group}")
    print(f"  Trials: {args.trials}")
    print(f"  Optimize α: {args.optimize_alpha}")
    print(f"  Seed: {seed}")
    print(f"  Directions pre-loaded: {sum(d.shape[0] for d in preloaded_group_directions.values())} total")
    print(f"{'='*60}\n")
    
    # Set up evaluation path
    eval_path = os.path.join(f"./runs/{args.model_name}", 'lama_val_optim_evaluations')
    aux_name = f"lama_{args.directions_prefix}"
    
    # Create objective with pre-loaded directions
    objective = partial(
        _lama_objective,
        n_groups=n_groups,
        dirs_per_group=args.dirs_per_group,
        max_dir_idx_per_group=max_dir_idx_per_group,
        n_layers=n_layers,
        rcs_alpha_init=rcs_alpha_init,
        layer_to_group=layer_to_group,
        model_name=args.model_name,
        preloaded_group_directions=preloaded_group_directions,
        group_metadata=group_metadata,
        layer_groups=layer_groups,
        dataset_name=getattr(args, 'dataset_name', 'harmbench_val'),
        device=getattr(args, 'device', 'cuda:0'),
        aux_name=aux_name,
        eval_path=eval_path,
        optimize_alpha=args.optimize_alpha,
    )
    
    # Create study
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        n_startup_trials=max(args.trials // 4, 5),
        seed=seed,
    )
    study = optuna.create_study(
        study_name=f"lama_{args.model_name}",
        direction="maximize",
        sampler=sampler
    )
    study.optimize(objective, n_trials=args.trials, n_jobs=1, show_progress_bar=True)
    
    # Save results
    out_dir = os.path.join(f"./runs/{args.model_name}", 'lama_optimizations')
    os.makedirs(out_dir, exist_ok=True)
    
    # Save study pickle
    study_path = os.path.join(out_dir, f'{aux_name}_study_t{args.trials}.pkl')
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    
    # Save best result as JSON
    best_params = study.best_params
    
    # Parse best params into structured format
    best_dir_indices = {}
    best_alphas = {}
    for key, value in best_params.items():
        if key.startswith('g') and '_dir' in key:
            parts = key.split('_')
            g_id = int(parts[0][1:])
            if g_id not in best_dir_indices:
                best_dir_indices[g_id] = []
            best_dir_indices[g_id].append(int(value))
        elif key.startswith('alpha_'):
            layer = int(key.split('_')[1])
            best_alphas[layer] = float(value)
    
    # If alpha wasn't optimized, use RCS-initialized values
    if not best_alphas:
        best_alphas = {int(k): float(v) for k, v in rcs_alpha_init.items()}
    
    best_summary = {
        'best_score': study.best_value,
        'best_dir_indices_per_group': {str(k): v for k, v in best_dir_indices.items()},
        'best_layer_weights': {str(k): round(v, 4) for k, v in best_alphas.items()},
        'best_params_raw': best_params,
        'model_name': args.model_name,
        'n_trials': args.trials,
        'n_groups': n_groups,
        'dirs_per_group': args.dirs_per_group,
        'optimize_alpha': args.optimize_alpha,
        'seed': seed,
        'rcs_alpha_init': {str(k): round(v, 4) for k, v in rcs_alpha_init.items()},
    }
    
    # Add trial history
    trial_details = []
    for t in study.trials:
        trial_details.append({
            'number': t.number,
            'state': t.state.name,
            'value': t.value,
            'params': t.params,
            'duration_sec': t.duration.total_seconds() if t.duration else None,
        })
    best_summary['trials'] = trial_details
    
    result_path = os.path.join(out_dir, f'{aux_name}_results_t{args.trials}.json')
    with open(result_path, 'w') as f:
        json.dump(best_summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"LAMA Optimization Complete")
    print(f"  Best ASR: {study.best_value:.2f}%")
    print(f"  Best directions per group: {best_dir_indices}")
    if best_alphas:
        alpha_vals = list(best_alphas.values())
        print(f"  α range: [{min(alpha_vals):.3f}, {max(alpha_vals):.3f}]")
        print(f"  α mean: {np.mean(alpha_vals):.3f}")
    print(f"  Results saved to: {result_path}")
    print(f"{'='*60}")
    
    return study, best_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LAMA: Joint Bayesian Optimization")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--directions_dir', type=str, required=True,
                        help='Directory containing per-group direction .pt files')
    parser.add_argument('--directions_prefix', type=str, default='lama',
                        help='Prefix used when saving directions (for metadata loading)')
    parser.add_argument('--rcs_path', type=str, default=None,
                        help='Path to RCS analysis JSON')
    parser.add_argument('--dataset_name', type=str, default='harmbench_val')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--trials', type=int, default=128,
                        help='Number of BO trials')
    parser.add_argument('--dirs_per_group', type=int, default=3,
                        help='Number of directions to select per group')
    parser.add_argument('--optimize_alpha', action='store_true',
                        help='Also optimize per-layer alpha weights (otherwise use RCS init)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    # Respect an externally-set CUDA_VISIBLE_DEVICES (multi-GPU parallel runs);
    # only fall back to deriving it from --device. After masking, the chosen GPU
    # is always visible as cuda:0.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]
    args.device = "cuda:0"
    run_lama_optimization(args)
