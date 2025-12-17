import torch
import torch.nn.functional as F

def power_iteration(matrix, num_iters=3, eps=1e-6):
    """
    Approximates the spectral norm ||M||_2 of a batch of matrices using Power Iteration.
    Implements Methodology Eq. 18.
    
    Why: Calculating exact spectral norm requires full SVD which is O(N^3). 
    Power iteration is O(N^2) and sufficient for perturbation bounds.

    Args:
        matrix: Tensor of shape (Batch, ..., N, D)
        num_iters: Number of iterations (K=3 is usually sufficient as per paper)
        eps: Small constant for numerical stability
    
    Returns:
        sigma_max: Approximate largest singular value (Batch, ...)
    """
    # Matrix dimensions
    *batch_dims, n, d = matrix.shape
    device = matrix.device
    dtype = matrix.dtype

    # Random initialization of vector v (Batch, ..., D, 1)
    v = torch.randn(*batch_dims, d, 1, device=device, dtype=dtype)
    v = F.normalize(v, dim=-2, p=2)

    # Power Iteration: v_{k+1} = (M^T M) v_k / ||...||
    # Note: We compute M v first, then M^T (M v) to avoid constructing M^T M (O(N^2 D) vs O(ND^2))
    with torch.no_grad():
        for _ in range(num_iters):
            # u = M v
            u = torch.matmul(matrix, v) 
            # v_new = M^T u
            v_new = torch.matmul(matrix.transpose(-1, -2), u)
            # Normalize
            v = F.normalize(v_new, dim=-2, p=2)

    # Rayleigh Quotient approximation for sigma_max
    # sigma_max ~= || M v ||
    u_final = torch.matmul(matrix, v)
    sigma_max = torch.norm(u_final, dim=(-2, -1), p=2)
    
    return sigma_max

def batched_partial_svd(matrix, rank=None):
    """
    Performs Batched SVD and optionally truncates to 'rank'.
    Implements Methodology Eq. 17 and Eq. 13 (Truncation).
    
    Args:
        matrix: Input tensor (Batch, N, D)
        rank: Integer r. If None, returns full SVD.
        
    Returns:
        U_r: (Batch, N, r)
        S_r: (Batch, r) - Diagonal singular values
        V_r: (Batch, D, r)
    """
    # PyTorch uses highly optimized cuSOLVER under the hood for linalg.svd
    # 'full_matrices=False' returns thin SVD (min(N, D))
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    
    if rank is not None:
        # Truncate to top-r components (Eq. 13)
        rank = min(rank, S.size(-1))
        U_r = U[..., :rank]
        S_r = S[..., :rank]
        V_r = Vh[..., :rank, :] # Vh is V^T, so we take top rows
        return U_r, S_r, V_r.transpose(-1, -2) # Return V not V^T for consistency
    
    return U, S, Vh.transpose(-1, -2)

def compute_energy_ratio(singular_values, rank):
    """
    Computes Normalized Energy Ratio (NER) for the State Space.
    Implements Methodology Eq. 15.
    
    Args:
        singular_values: Tensor of all singular values (Batch, min(N,D))
        rank: Selected rank r
        
    Returns:
        ner: Scalar (0-1) representing retained energy
    """
    # Square singular values to get eigenvalues (Energy)
    eigenvalues = singular_values ** 2
    
    # Total energy
    total_energy = torch.sum(eigenvalues, dim=-1, keepdim=True) + 1e-8
    
    # Retained energy (Top-r)
    retained_energy = torch.sum(eigenvalues[..., :rank], dim=-1, keepdim=True)
    
    return retained_energy / total_energy