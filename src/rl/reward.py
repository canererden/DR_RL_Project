import torch
import torch.nn.functional as F

def compute_reward(full_rank_attn, low_rank_attn, flops, perturbation_norm, config):
    """
    Computes the composite reward function R_t.
    Implements Methodology Eq. 12.
    
    Args:
        full_rank_attn: Output from standard attention (Batch, Seq, Dim) - The "Target"
        low_rank_attn: Output from DR-RL attention (Batch, Seq, Dim)
        flops: (float) Calculated FLOPs for the chosen rank
        perturbation_norm: (float) ||Delta A||_F
        config: Dictionary containing alpha, beta, gamma weights
    
    Returns:
        reward: Scalar tensor
    """
    alpha = config['rl']['alpha']
    beta = config['rl']['beta']
    gamma = config['rl']['gamma']
    
    # 1. Fidelity Reward (Cosine Similarity)
    # Flatten to (Batch*Seq, Dim) to compute similarity over vectors
    full_flat = full_rank_attn.reshape(-1, full_rank_attn.size(-1))
    low_flat = low_rank_attn.reshape(-1, low_rank_attn.size(-1))
    
    cosine_sim = F.cosine_similarity(full_flat, low_flat, dim=-1).mean()
    
    # 2. Efficiency Penalty (FLOPs)
    # Normalize FLOPs to be in a reasonable range (e.g., relative to max FLOPs)
    # Assuming max_flops is roughly 1e9, we scale it down.
    flops_penalty = flops / 1e9 
    
    # 3. Stability Penalty (Perturbation)
    # Penalize if perturbation is high
    pert_penalty = perturbation_norm
    
    # Total Reward
    # maximize Similarity, minimize FLOPs, minimize Perturbation
    reward = (alpha * cosine_sim) - (beta * flops_penalty) - (gamma * pert_penalty)
    
    return reward