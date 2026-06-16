import os
import torch 

def get_ablated_matrix(matrix: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    vec = vec / torch.norm(vec)
    vec = vec.to(matrix)
    proj = torch.einsum('...d,d->...', matrix, vec)  # shape: [...]
    return matrix - proj.unsqueeze(-1) * vec  # shape: [..., d_model]

def get_weighted_ablated_matrix(matrix: torch.Tensor, vec: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Weighted ablation: scale the projection removal by alpha ∈ [0, 1].
    
    When alpha=1.0, this is identical to get_ablated_matrix (full ablation).
    When alpha=0.0, the matrix is returned unchanged (no ablation).
    Intermediate values give partial ablation, enabling per-layer adaptive control.
    
    Args:
        matrix: Weight matrix to ablate [..., d_model]
        vec: Direction vector to ablate [d_model]
        alpha: Ablation strength in [0, 1]
    
    Returns:
        Ablated matrix with projection scaled by alpha
    """
    if alpha == 0.0:
        return matrix
    vec = vec / torch.norm(vec)
    vec = vec.to(matrix)
    proj = torch.einsum('...d,d->...', matrix, vec)  # shape: [...]
    return matrix - alpha * proj.unsqueeze(-1) * vec  # shape: [..., d_model]

def ablate_weights(model, direction: torch.Tensor):
    model.ablate_weights(direction)
    