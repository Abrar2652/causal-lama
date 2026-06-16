"""
lama_generate_directions.py — Per-Group SOM Direction Extraction for LAMA

Given RCS-based layer grouping, this module:
  1. Loads RCS analysis results (groups + representatives)
  2. For each group g, trains a SOM on harmful representations at layer l_g*
  3. Extracts group-specific directions: r_i^(g) = w_i^(g) - ν^(l_g*)
  4. Saves per-group directions for downstream ablation and BO search

This is the LAMA generalization of som_generate_directions.py:
  - Original: single SOM at one layer → one set of directions
  - LAMA: multiple SOMs, one per group → group-specific direction sets
"""

import os
import json
import argparse
import numpy as np
import torch
import warnings

from config import Config
from som import train_som
from dataset.utils import compute_centroid
from lama_rcs import load_representations, compute_rcs, auto_group_layers, select_group_representatives


def generate_group_directions(
    HF_x, HL_x, groups, representatives,
    som_x=4, som_y=4, iterations=10000, sigma=0.33, lr=0.01
):
    """Train SOMs per layer group and extract group-specific directions.
    
    For each group g with representative layer l_g*:
      1. Extract harmful representations at l_g*
      2. Train a SOM on those representations
      3. Compute harmless centroid ν at l_g*
      4. Build directions: r_i = w_i - ν for each SOM neuron w_i
    
    Args:
        HF_x: [n_harmful, n_layers, hidden_dim] harmful representations
        HL_x: [n_harmless, n_layers, hidden_dim] harmless representations
        groups: list of lists (layer indices per group)
        representatives: list of representative layer indices (one per group)
        som_x, som_y: SOM grid dimensions
        iterations: SOM training iterations
        sigma: SOM neighborhood radius
        lr: SOM learning rate
    
    Returns:
        group_directions: dict mapping group_id → Tensor [n_neurons, hidden_dim]
        group_soms: dict mapping group_id → trained SOM object
        group_metadata: dict with training info per group
    """
    if isinstance(HF_x, torch.Tensor):
        HF_x = HF_x.numpy()
    if isinstance(HL_x, torch.Tensor):
        HL_x = HL_x.numpy()
    
    group_directions = {}
    group_soms = {}
    group_metadata = {}
    
    for g_id, (group_layers, rep_layer) in enumerate(zip(groups, representatives)):
        if rep_layer is None:
            print(f"[Group {g_id}] Skipping — no representative layer")
            continue
        
        print(f"\n{'='*50}")
        print(f"Group {g_id}: layers {min(group_layers)}-{max(group_layers)}, "
              f"representative layer = {rep_layer}")
        print(f"{'='*50}")
        
        # Extract representations at the representative layer
        HF_layer = HF_x[:, rep_layer, :]  # [n_harmful, hidden_dim]
        HL_layer = HL_x[:, rep_layer, :]  # [n_harmless, hidden_dim]
        
        # Harmless centroid at this layer
        harmless_centroid = HL_layer.mean(axis=0)
        harmless_centroid_t = torch.tensor(harmless_centroid, dtype=torch.float32)
        
        # Train SOM on harmful representations
        print(f"  Training {som_x}x{som_y} SOM ({iterations} iterations, σ={sigma}, lr={lr})...")
        som = train_som(
            HF_layer,
            som_x=som_x, som_y=som_y,
            iterations=iterations,
            sigma=sigma,
            learning_rate=lr
        )
        
        # Extract SOM weights (neurons)
        weights = som.get_weights()  # [som_x, som_y, hidden_dim]
        n_neurons = som_x * som_y
        
        # Flatten weights to [n_neurons, hidden_dim]
        weights_flat = weights.reshape(n_neurons, -1)
        weights_t = torch.tensor(weights_flat, dtype=torch.float32)
        
        # Compute directions: each neuron weight - harmless centroid
        directions = weights_t - harmless_centroid_t.unsqueeze(0)
        
        # Check for zero directions
        norms = torch.norm(directions, dim=1)
        zero_mask = norms < 1e-8
        if zero_mask.any():
            n_zero = zero_mask.sum().item()
            warnings.warn(f"  [Group {g_id}] {n_zero} zero-norm directions detected")
        
        group_directions[g_id] = directions
        group_soms[g_id] = som
        group_metadata[g_id] = {
            'representative_layer': int(rep_layer),
            'layers': [int(l) for l in group_layers],
            'n_neurons': n_neurons,
            'n_directions': directions.shape[0],
            'direction_norms': norms.tolist(),
            'som_x': som_x,
            'som_y': som_y,
            'sigma': sigma,
            'lr': lr,
            'iterations': iterations,
        }
        
        print(f"  → Generated {directions.shape[0]} directions "
              f"(shape: {directions.shape})")
        print(f"  → Direction norms: min={norms.min():.4f}, max={norms.max():.4f}, "
              f"mean={norms.mean():.4f}")
    
    return group_directions, group_soms, group_metadata


def save_group_directions(group_directions, group_metadata, save_dir, aux_name='lama'):
    """Save per-group directions and metadata.
    
    Args:
        group_directions: dict mapping group_id → Tensor
        group_metadata: dict with per-group info
        save_dir: output directory
        aux_name: prefix for filenames
    
    Returns:
        saved_paths: dict mapping group_id → file path
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = {}
    
    for g_id, directions in group_directions.items():
        filename = f'{aux_name}_group{g_id}_directions.pt'
        filepath = os.path.join(save_dir, filename)
        torch.save(directions, filepath)
        saved_paths[g_id] = filepath
        print(f"[✓] Group {g_id}: saved {directions.shape[0]} directions to {filepath}")
    
    # Save metadata
    meta_path = os.path.join(save_dir, f'{aux_name}_metadata.json')
    
    # Convert for JSON serialization
    meta_serializable = {}
    for g_id, meta in group_metadata.items():
        meta_clean = {}
        for k, v in meta.items():
            if isinstance(v, (list, int, float, str)):
                meta_clean[k] = v
            elif isinstance(v, np.ndarray):
                meta_clean[k] = v.tolist()
            else:
                meta_clean[k] = str(v)
        meta_serializable[str(g_id)] = meta_clean
    
    meta_serializable['saved_paths'] = {str(k): v for k, v in saved_paths.items()}
    
    with open(meta_path, 'w') as f:
        json.dump(meta_serializable, f, indent=2)
    print(f"[✓] Metadata saved to {meta_path}")
    
    return saved_paths


def load_group_directions(save_dir, aux_name='lama'):
    """Load previously saved per-group directions.
    
    Args:
        save_dir: directory containing saved directions
        aux_name: prefix used when saving
    
    Returns:
        group_directions: dict mapping group_id → Tensor
        metadata: dict with group metadata
    """
    meta_path = os.path.join(save_dir, f'{aux_name}_metadata.json')
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    group_directions = {}
    saved_paths = metadata.get('saved_paths', {})
    
    for g_id_str, filepath in saved_paths.items():
        g_id = int(g_id_str)
        directions = torch.load(filepath, weights_only=True)
        group_directions[g_id] = directions
        print(f"[✓] Loaded group {g_id}: {directions.shape[0]} directions from {filepath}")
    
    return group_directions, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LAMA: Per-Group SOM Direction Generation")
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--som_x', type=int, default=4, help='SOM x dimension')
    parser.add_argument('--som_y', type=int, default=4, help='SOM y dimension')
    parser.add_argument('--iterations', type=int, default=10000, help='SOM training iterations')
    parser.add_argument('--sigma', type=float, default=0.33, help='SOM neighborhood radius')
    parser.add_argument('--lr', type=float, default=0.01, help='SOM learning rate')
    parser.add_argument('--n_groups', type=int, default=3, help='Number of layer groups')
    parser.add_argument('--grouping_method', type=str, default='quantile',
                        choices=['quantile', 'peaks', 'adaptive'])
    parser.add_argument('--rcs_path', type=str, default=None,
                        help='Path to pre-computed RCS analysis JSON (optional)')
    args = parser.parse_args()
    
    cfg = Config(model_alias=args.model_name, model_path=args.model_name)
    
    # Load representations
    HF_x, HL_x = load_representations(args.model_name)
    
    # Compute or load RCS
    if args.rcs_path and os.path.exists(args.rcs_path):
        print(f"Loading RCS from {args.rcs_path}...")
        with open(args.rcs_path, 'r') as f:
            rcs_data = json.load(f)
        rcs_scores = {int(k): v for k, v in rcs_data['rcs_scores'].items()}
    else:
        print("Computing RCS...")
        rcs_scores, _ = compute_rcs(HF_x, HL_x)
    
    # Layer grouping
    groups, group_map = auto_group_layers(rcs_scores, args.n_groups, args.grouping_method)
    representatives = select_group_representatives(rcs_scores, groups)
    
    print(f"\nLayer grouping ({args.n_groups} groups):")
    for g_id, (g_layers, rep) in enumerate(zip(groups, representatives)):
        print(f"  Group {g_id}: layers {min(g_layers)}-{max(g_layers)}, rep=layer {rep}")
    
    # Generate directions
    group_directions, group_soms, group_metadata = generate_group_directions(
        HF_x, HL_x, groups, representatives,
        som_x=args.som_x, som_y=args.som_y,
        iterations=args.iterations, sigma=args.sigma, lr=args.lr
    )
    
    # Save
    save_dir = os.path.join(cfg.artifact_path(), 'generate_directions')
    aux_name = f'lama_som{args.som_x}x{args.som_y}_sigma{args.sigma}_groups{args.n_groups}'
    save_group_directions(group_directions, group_metadata, save_dir, aux_name)
    
    print(f"\n{'='*50}")
    print(f"LAMA direction generation complete for {args.model_name}")
    print(f"  Groups: {args.n_groups}")
    print(f"  Total directions: {sum(d.shape[0] for d in group_directions.values())}")
    print(f"  Saved to: {save_dir}")
    print(f"{'='*50}")
