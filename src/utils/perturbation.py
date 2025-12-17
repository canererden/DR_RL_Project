import torch
from src.utils.svd_utils import power_iteration

def compute_rank_transition_perturbation(singular_values, current_rank, target_rank):
    """
    Computes the perturbation bound ||Delta A||_F when changing rank.
    Implements Methodology Eq. 8 (The tail energy bound).
    
    Args:
        singular_values: Full spectrum singular values (Batch, min(N,D))
        current_rank: r
        target_rank: r'
        
    Returns:
        perturbation_norm: Frobenius norm of the difference matrix
    """
    if target_rank >= current_rank:
        # Rank Increase: Theoretically 0 approximation error relative to previous low-rank,
        # but we measure difference between A_r' and A_r.
        # Diff is determined by singular values from r to r'
        start, end = current_rank, target_rank
    else:
        # Rank Decrease: Error is the lost energy from r' to r
        start, end = target_rank, current_rank
        
    # Extract singular values in the transition region (sigma_{r+1} ... sigma_{r'})
    transition_sigma = singular_values[..., start:end]
    
    # ||Delta A||_F = sqrt(sum(sigma_k^2))
    perturbation_norm = torch.sqrt(torch.sum(transition_sigma ** 2, dim=-1))
    
    return perturbation_norm

def estimate_attention_output_sensitivity(delta_A_norm, V_matrix):
    """
    Bounds the change in Attention Output O = AV.
    Implements Methodology Eq. 9: ||O' - O||_F <= ||Delta A||_2 * ||V||_F
    
    Note: Eq. 9 uses spectral norm ||Delta A||_2 for a tighter bound on the product,
    but we often use Frobenius norm ||Delta A||_F as a conservative proxy 
    since ||M||_2 <= ||M||_F.
    
    Args:
        delta_A_norm: Calculated perturbation norm (scalar or batch)
        V_matrix: Value matrix (Batch, N, D)
        
    Returns:
        sensitivity_bound: Upper bound on output deviation
    """
    # Compute Frobenius norm of V
    v_norm = torch.norm(V_matrix, p='fro', dim=(-2, -1))
    
    # Sensitivity <= ||Delta A|| * ||V||
    sensitivity_bound = delta_A_norm * v_norm
    
    return sensitivity_bound

def is_action_safe(perturbation_bound, threshold):
    """
    Safety Guardrail Logic (Section 4.3).
    Checks if the perturbation exceeds the dynamic threshold epsilon_t.
    
    Args:
        perturbation_bound: Calculated error bound
        threshold: Current adaptive epsilon_t (Eq. 10)
        
    Returns:
        Boolean mask (True if safe, False if unsafe)
    """
    return perturbation_bound <= threshold