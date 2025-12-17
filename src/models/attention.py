import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.utils.svd_utils import batched_partial_svd

class DynamicLowRankAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) module capable of Dynamic Rank adaptation.
    
    This module implements:
    1. Standard Full-Rank Attention (Eq. 1)
    2. Low-Rank Approximation via SVD on Q and K (Eq. 13 & 14)
    3. FLOPs counting for Reward calculation (Eq. 7)
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config['model']['d_model']
        self.n_heads = config['model']['n_heads']
        self.head_dim = self.d_model // self.n_heads
        
        assert self.head_dim * self.n_heads == self.d_model, "d_model must be divisible by n_heads"

        # Projections (W_q, W_k, W_v) - Standard Transformer
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.o_proj = nn.Linear(self.d_model, self.d_model)
        
        self.dropout = nn.Dropout(config['model']['dropout'])

    def get_layer_stats(self):
        """
        Extracts layer-specific statistics for the RL Agent's State Space (Section 4.1).
        Returns: [mean, var, norm] of weight matrices.
        """
        with torch.no_grad():
            weights = [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight]
            stats = []
            for w in weights:
                stats.append(w.mean())
                stats.append(w.var())
                stats.append(torch.norm(w))
            return torch.stack(stats)

    def _split_heads(self, x):
        """Reshapes (Batch, Seq, Dim) -> (Batch, Heads, Seq, HeadDim)"""
        new_shape = x.size()[:-1] + (self.n_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def _apply_low_rank_approximation(self, matrix, rank):
        """
        Applies Eq. 13: Truncated SVD Reconstruction
        Input: (Batch, Heads, Seq, HeadDim)
        Output: Low-rank approximated matrix of same shape.
        """
        # Reshape to (Batch*Heads, Seq, HeadDim) for batched SVD
        b, h, s, d = matrix.shape
        flat_matrix = matrix.reshape(b * h, s, d)
        
        # Methodology Eq. 13 & 17: Batched Partial SVD
        # U_r, S_r, V_r = SVD_r(Matrix)
        U_r, S_r, V_r = batched_partial_svd(flat_matrix, rank=rank)
        
        # Reconstruct: M_approx = U_r * S_r * V_r^T
        # Diagonalize S_r for multiplication
        S_diag = torch.diag_embed(S_r)
        
        reconstructed = torch.matmul(torch.matmul(U_r, S_diag), V_r.transpose(-1, -2))
        
        # Restore shape
        return reconstructed.view(b, h, s, d)

    def forward(self, x, mask=None, rank=None):
        """
        Args:
            x: Input embeddings (Batch, Seq, Dim)
            mask: Attention mask
            rank: (int, optional) Selected rank 'r' from RL Agent. 
                  If None, performs standard Full-Rank attention.
        
        Returns:
            context: (Batch, Seq, Dim)
            metadata: dict containing FLOPs and attention matrix (for perturbation check)
        """
        batch_size, seq_len, _ = x.size()

        # 1. Linear Projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. Split Heads
        Q = self._split_heads(Q) # (B, H, S, D_h)
        K = self._split_heads(K)
        V = self._split_heads(V)

        flops = 0
        
        # 3. Dynamic Rank Selection Logic
        if rank is not None and rank < self.head_dim:
            # --- Low-Rank Path (Methodology Section 3.1 & 3.2) ---
            
            # Apply SVD approximation to Q and K (Eq. 13)
            Q = self._apply_low_rank_approximation(Q, rank)
            K = self._apply_low_rank_approximation(K, rank)
            
            # Calculate FLOPs for Reward Function (Eq. 7 & 12)
            # SVD approx cost: O(N*r*d) approx per head
            # Note: This is a theoretical FLOPs count for the Reward signal, 
            # independent of actual execution time which depends on hardware optimization.
            flops = 2 * batch_size * self.n_heads * seq_len * rank * self.head_dim
            
        else:
            # --- Full-Rank Path (Standard Attention) ---
            # FLOPs: O(N^2 * d) for QK^T
            flops = 2 * batch_size * self.n_heads * (seq_len ** 2) * self.head_dim

        # 4. Attention Computation (Eq. 1 / Eq. 14)
        # scores = Q * K^T / sqrt(d)
        d_k = self.head_dim
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. Output Aggregation
        context = torch.matmul(attn_weights, V) # (B, H, S, D_h)
        
        # Reshape and project output
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        
        output = self.o_proj(context)

        # Metadata for RL Reward and Perturbation Analysis
        metadata = {
            "flops": flops,
            "attn_weights": attn_weights if self.training else None, # Keep for perturbation check if needed
            "rank_used": rank if rank else self.head_dim
        }

        return output, metadata