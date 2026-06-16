"""
lama_ablate.py — LAMA Ablation Entry Point

Applies layer-adaptive multi-directional ablation to a model and generates completions.
This is the LAMA equivalent of ablate.py:
  - Original: loads global directions, applies uniform ablation
  - LAMA: loads per-group directions + layer weights, applies adaptive ablation

Supports three ablation modes for the ablation study:
  - uniform: per-group SOMs + α=1.0 for all layers (isolates SOM effect)
  - rcs:     per-group SOMs + α from RCS initialization (no BO)
  - bo:      per-group SOMs + α from BO optimization (full method)

Usage:
    # Full LAMA (BO-optimized)
    python lama_ablate.py \\
        --model_name llama2-7b \\
        --directions_dir ./runs/llama2-7b/generate_directions \\
        --directions_prefix lama_som4x4_sigma0.33_groups3 \\
        --optimization_results ./runs/llama2-7b/lama_optimizations/lama_results.json \\
        --dataset_name harmbench_test \\
        --ablation_mode bo

    # Ablation study: uniform weights
    python lama_ablate.py \\
        --model_name llama2-7b \\
        --directions_dir ./runs/llama2-7b/generate_directions \\
        --directions_prefix lama_som4x4_sigma0.33_groups3 \\
        --dir_ids_per_group "0:1,2,3;1:0,4,5;2:1,2" \\
        --ablation_mode uniform
"""

import os
import json
import random
import argparse
from datetime import datetime

import numpy as np
import torch

from config import Config
from models.load_models import load_model
from directions_ablation import generate_and_save_hookfree_completions
from lama_generate_directions import load_group_directions
from lama_rcs import compute_rcs, compute_rcs_weights, load_representations


# --------------------------------------------------------------------------- #
#  Reproducibility                                                             #
# --------------------------------------------------------------------------- #
def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
#  Load BO optimization results                                                #
# --------------------------------------------------------------------------- #
def load_optimization_results(results_path):
    """Load best parameters from LAMA BO optimization.
    
    Args:
        results_path: path to optimization results JSON
    
    Returns:
        dir_indices_per_group: dict mapping group_id → list of direction indices
        layer_weights: dict mapping layer_idx → alpha_l
        opt_results: raw JSON dict
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Parse direction indices
    dir_indices_per_group = {}
    raw_dirs = results.get('best_dir_indices_per_group', {})
    for g_id_str, indices in raw_dirs.items():
        dir_indices_per_group[int(g_id_str)] = indices
    
    # Parse layer weights
    layer_weights = {}
    raw_weights = results.get('best_layer_weights', {})
    for l_str, alpha in raw_weights.items():
        layer_weights[int(l_str)] = float(alpha)
    
    print(f"Loaded optimization results from {results_path}")
    print(f"  Best ASR: {results.get('best_score', 'N/A')}")
    print(f"  Direction indices per group: {dir_indices_per_group}")
    if layer_weights:
        alpha_vals = list(layer_weights.values())
        print(f"  α range: [{min(alpha_vals):.3f}, {max(alpha_vals):.3f}]")
    
    return dir_indices_per_group, layer_weights, results


# --------------------------------------------------------------------------- #
#  Core LAMA ablation                                                          #
# --------------------------------------------------------------------------- #
def run_lama_ablation(
    model_name, device, directions_dir, directions_prefix,
    dir_indices_per_group, layer_weights, group_metadata,
    dataset_name='harmbench_test', ablation_mode='bo',
    eval_utility=False, utility_max=200, output_dir=None
):
    """Apply LAMA ablation and generate completions.
    
    Args:
        model_name: model to load
        device: torch device
        directions_dir: directory with group direction files
        directions_prefix: prefix for direction files
        dir_indices_per_group: {group_id: [dir_indices]}
        layer_weights: {layer_idx: alpha_l}
        group_metadata: metadata dict
        dataset_name: dataset to generate completions for
        ablation_mode: 'uniform' | 'rcs' | 'bo' (for naming/tracking)
    
    Returns:
        completions: list of completion dicts
        aux_name: string identifier for this run
    """
    # Load model
    print(f"\nLoading model: {model_name}...")
    llm = load_model(model_name, device)
    cfg = Config(model_alias=model_name, model_path=model_name)
    
    # Load all group directions
    group_directions_all, _ = load_group_directions(directions_dir, directions_prefix)
    
    # Select the chosen directions per group
    selected_group_directions = {}
    for g_id, dir_indices in dir_indices_per_group.items():
        if g_id in group_directions_all:
            all_dirs = group_directions_all[g_id]
            valid_indices = [i for i in dir_indices if i < len(all_dirs)]
            selected_group_directions[g_id] = [all_dirs[i] for i in valid_indices]
            print(f"  Group {g_id}: selected {len(valid_indices)} directions "
                  f"(indices: {valid_indices})")
    
    # Build layer → group mapping
    layer_groups = {}
    for g_id_str, meta in group_metadata.items():
        if g_id_str in ('saved_paths',):
            continue
        g_id = int(g_id_str)
        for layer in meta.get('layers', []):
            layer_groups[layer] = g_id
    
    # Apply adaptive ablation
    alpha_vals = list(layer_weights.values())
    print(f"\nApplying LAMA adaptive ablation (mode={ablation_mode})...")
    print(f"  α range: [{min(alpha_vals):.3f}, {max(alpha_vals):.3f}], "
          f"mean={np.mean(alpha_vals):.3f}")
    llm.ablate_weights_adaptive(selected_group_directions, layer_groups, layer_weights)
    
    # Build aux name for file output
    dir_desc = '_'.join(
        f"g{g}:{'-'.join(map(str, dirs))}"
        for g, dirs in sorted(dir_indices_per_group.items())
    )
    aux_name = f'lama_{ablation_mode}_{dir_desc}_{directions_prefix}'
    
    # Generate completions
    print(f"\nGenerating completions on {dataset_name}...")
    completions = generate_and_save_hookfree_completions(
        cfg=cfg,
        folder='lama_completions',
        model_base=llm,
        dataset_name=dataset_name,
        aux_name=aux_name,
        return_completions=True,
    )

    # Capability / safety-tax evaluation on the SAME ablated model (before freeing).
    # The baseline never measures this; ablating every writing matrix can damage
    # general ability, so we report perplexity + harmless-compliance alongside ASR.
    if eval_utility:
        try:
            from dataset.load_dataset import load_dataset_split
            from lama_eval_utility import (compute_perplexity,
                                           compute_harmless_compliance, utility_report)
            harmless = load_dataset_split('harmless', 'val', instructions_only=True)[:utility_max]
            print(f"\n[Utility] Measuring capability cost on {len(harmless)} harmless prompts...")
            ppl = compute_perplexity(llm, [llm._get_prompt(p) for p in harmless])
            comp = compute_harmless_compliance(llm, harmless)
            util = utility_report(ppl, comp)
            util.update({'ablation_mode': ablation_mode, 'aux_name': aux_name,
                         'per_text_nll': ppl['per_text_nll']})
            util_dir = output_dir or os.path.join(cfg.artifact_path(), 'lama_utility')
            os.makedirs(util_dir, exist_ok=True)
            util_path = os.path.join(util_dir, f'{aux_name}_utility.json')
            with open(util_path, 'w') as f:
                json.dump(util, f, indent=2)
            print(f"[Utility] perplexity={util['perplexity']:.3f}  "
                  f"compliance={util['compliance_rate']:.3f}  "
                  f"over_refusal={util['over_refusal_rate']:.3f}  → {util_path}")
        except Exception as e:
            print(f"[Utility] WARNING: utility eval failed: {e}")

    # Cleanup
    del llm
    torch.cuda.empty_cache()

    return completions, aux_name


def run_lama_ablation_with_rcs_weights(
    model_name, device, directions_dir, directions_prefix,
    dir_indices_per_group, group_metadata,
    dataset_name='harmbench_test'
):
    """Apply LAMA ablation using RCS-initialized weights (no BO optimization).
    
    This is useful for:
    1. Quick evaluation without running the full BO search
    2. Ablation study: isolating the effect of adaptive weights from BO optimization
    """
    # Load representations and compute RCS
    HF_x, HL_x = load_representations(model_name)
    rcs_scores, _ = compute_rcs(HF_x, HL_x)
    layer_weights = compute_rcs_weights(rcs_scores)
    
    print(f"Using RCS-initialized layer weights (no BO optimization)")
    alpha_vals = list(layer_weights.values())
    print(f"  α range: [{min(alpha_vals):.3f}, {max(alpha_vals):.3f}]")
    print(f"  α mean: {sum(alpha_vals)/len(alpha_vals):.3f}")
    
    return run_lama_ablation(
        model_name, device, directions_dir, directions_prefix,
        dir_indices_per_group, layer_weights, group_metadata,
        dataset_name, ablation_mode='rcs'
    )


def run_lama_ablation_uniform(
    model_name, device, directions_dir, directions_prefix,
    dir_indices_per_group, group_metadata, n_layers,
    dataset_name='harmbench_test'
):
    """Apply LAMA ablation with uniform α=1.0 for all layers.
    
    For the ablation study: this isolates the effect of per-group SOMs
    from the adaptive weighting. If LAMA-uniform > MD, the improvement
    comes from better directions. If LAMA-adaptive > LAMA-uniform,
    the improvement comes from adaptive weighting.
    """
    layer_weights = {l: 1.0 for l in range(n_layers)}
    
    print(f"Using UNIFORM layer weights (α=1.0 for all {n_layers} layers)")
    print(f"  This isolates the per-group SOM effect from adaptive weighting.")
    
    return run_lama_ablation(
        model_name, device, directions_dir, directions_prefix,
        dir_indices_per_group, layer_weights, group_metadata,
        dataset_name, ablation_mode='uniform'
    )


def save_experiment_config(args, completions, aux_name, layer_weights, 
                           dir_indices_per_group, output_dir):
    """Save full experiment configuration for reproducibility.
    
    Reviewers can verify exact settings from this JSON.
    """
    config = {
        'timestamp': datetime.now().isoformat(),
        'model_name': args.model_name,
        'dataset_name': args.dataset_name,
        'ablation_mode': args.ablation_mode,
        'directions_prefix': args.directions_prefix,
        'seed': getattr(args, 'seed', 42),
        'dir_indices_per_group': {str(k): v for k, v in dir_indices_per_group.items()},
        'layer_weights': {str(k): round(v, 6) for k, v in layer_weights.items()},
        'alpha_stats': {
            'min': min(layer_weights.values()),
            'max': max(layer_weights.values()),
            'mean': float(np.mean(list(layer_weights.values()))),
            'std': float(np.std(list(layer_weights.values()))),
            'total': sum(layer_weights.values()),
            'n_layers': len(layer_weights),
        },
        'n_completions': len(completions),
        'aux_name': aux_name,
    }
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{aux_name}_experiment_config.json')
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[✓] Experiment config saved to {path}")
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LAMA: Layer-Adaptive Multi-Directional Ablation")
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--directions_dir', type=str, required=True,
                        help='Directory containing per-group direction .pt files')
    parser.add_argument('--directions_prefix', type=str, required=True,
                        help='Prefix used when saving directions')
    parser.add_argument('--optimization_results', type=str, default=None,
                        help='Path to BO optimization results JSON. '
                             'If not provided, uses RCS-initialized weights.')
    parser.add_argument('--dir_ids_per_group', type=str, default=None,
                        help='Manual direction indices per group, e.g. "0:1,2,3;1:0,4,5;2:1,2"')
    parser.add_argument('--dataset_name', type=str, default='harmbench_test')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--ablation_mode', type=str, default='bo',
                        choices=['uniform', 'rcs', 'bo'],
                        help='Ablation mode: uniform (α=1), rcs (RCS init), bo (BO optimized)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eval_utility', action='store_true',
                        help='Also measure capability cost (perplexity + harmless '
                             'compliance) on the ablated model.')
    parser.add_argument('--utility_max', type=int, default=200,
                        help='Number of harmless prompts for the utility eval.')
    args = parser.parse_args()
    
    # Reproducibility
    set_seeds(args.seed)
    
    device = torch.device(args.device)
    cfg = Config(model_alias=args.model_name, model_path=args.model_name)
    
    # Load group metadata
    meta_path = os.path.join(args.directions_dir, f'{args.directions_prefix}_metadata.json')
    with open(meta_path, 'r') as f:
        group_metadata = json.load(f)
    
    # Determine n_layers from metadata
    n_layers = 0
    for g_id_str, meta in group_metadata.items():
        if g_id_str in ('saved_paths',):
            continue
        for l in meta.get('layers', []):
            n_layers = max(n_layers, l + 1)
    
    # Determine direction indices and layer weights based on mode
    completions = None
    aux_name = None
    layer_weights = None
    dir_indices_per_group = None
    
    if args.ablation_mode == 'bo':
        if not args.optimization_results or not os.path.exists(args.optimization_results):
            print("ERROR: --ablation_mode bo requires --optimization_results")
            exit(1)
        dir_indices_per_group, layer_weights, _ = load_optimization_results(
            args.optimization_results
        )
        completions, aux_name = run_lama_ablation(
            args.model_name, args.device,
            args.directions_dir, args.directions_prefix,
            dir_indices_per_group, layer_weights, group_metadata,
            args.dataset_name, ablation_mode='bo',
            eval_utility=args.eval_utility, utility_max=args.utility_max
        )
    
    elif args.ablation_mode == 'rcs':
        if not args.dir_ids_per_group:
            print("ERROR: --ablation_mode rcs requires --dir_ids_per_group")
            exit(1)
        dir_indices_per_group = {}
        for group_spec in args.dir_ids_per_group.split(';'):
            g_id_str, indices_str = group_spec.split(':')
            dir_indices_per_group[int(g_id_str)] = [int(x) for x in indices_str.split(',')]
        
        HF_x, HL_x = load_representations(args.model_name)
        rcs_scores, _ = compute_rcs(HF_x, HL_x)
        layer_weights = compute_rcs_weights(rcs_scores)
        
        completions, aux_name = run_lama_ablation(
            args.model_name, args.device,
            args.directions_dir, args.directions_prefix,
            dir_indices_per_group, layer_weights, group_metadata,
            args.dataset_name, ablation_mode='rcs',
            eval_utility=args.eval_utility, utility_max=args.utility_max
        )
    
    elif args.ablation_mode == 'uniform':
        if not args.dir_ids_per_group:
            print("ERROR: --ablation_mode uniform requires --dir_ids_per_group")
            exit(1)
        dir_indices_per_group = {}
        for group_spec in args.dir_ids_per_group.split(';'):
            g_id_str, indices_str = group_spec.split(':')
            dir_indices_per_group[int(g_id_str)] = [int(x) for x in indices_str.split(',')]
        
        layer_weights = {l: 1.0 for l in range(n_layers)}
        completions, aux_name = run_lama_ablation(
            args.model_name, args.device,
            args.directions_dir, args.directions_prefix,
            dir_indices_per_group, layer_weights, group_metadata,
            args.dataset_name, ablation_mode='uniform',
            eval_utility=args.eval_utility, utility_max=args.utility_max
        )
    
    # Save experiment config
    if completions and layer_weights and dir_indices_per_group:
        save_experiment_config(
            args, completions, aux_name, layer_weights,
            dir_indices_per_group,
            os.path.join(cfg.artifact_path(), 'lama_experiments')
        )
    
    print(f"\n{'='*50}")
    print(f"LAMA Ablation Complete")
    print(f"  Model: {args.model_name}")
    print(f"  Mode: {args.ablation_mode}")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Completions generated: {len(completions)}")
    print(f"  Output name: {aux_name}")
    print(f"{'='*50}")
    print(f"\nTo evaluate, run:")
    print(f"  python eval_jailbreaks.py \\")
    print(f"    --completions_path ./runs/{args.model_name}/lama_completions/{aux_name}_{args.dataset_name}_completions.json \\")
    print(f"    --evaluation_path ./runs/{args.model_name}/lama_evaluations/{aux_name}_results.json")
