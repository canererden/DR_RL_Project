import torch
import torch.nn as nn
from src.models.attention import DynamicLowRankAttention
from src.rl.agent import RankSelectionPolicy
from src.utils.perturbation import compute_rank_transition_perturbation, is_action_safe

class DRRLTransformerBlock(nn.Module):
    """
    A single Transformer block that supports Dynamic Rank Attention.
    Replaces the standard nn.TransformerDecoderLayer.
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config['model']['d_model']
        self.d_ff = config['model']['d_ff']
        self.dropout = config['model']['dropout']

        # 1. Dynamic Rank Attention (Our Custom Module)
        self.attn = DynamicLowRankAttention(config)
        self.ln1 = nn.LayerNorm(self.d_model)
        
        # 2. Feed-Forward Network (Standard)
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.d_model),
            nn.Dropout(self.dropout)
        )
        self.ln2 = nn.LayerNorm(self.d_model)

    def forward(self, x, mask=None, rank=None):
        """
        Args:
            x: Input tensor (Batch, Seq, Dim)
            mask: Attention mask
            rank: Selected rank 'r' (int) for this specific layer
        """
        # Part 1: Attention
        residual = x
        x = self.ln1(x)
        attn_out, metadata = self.attn(x, mask=mask, rank=rank)
        x = residual + attn_out # Residual connection

        # Part 2: Feed-Forward
        residual = x
        x = self.ln2(x)
        ff_out = self.ff(x)
        x = residual + ff_out # Residual connection

        return x, metadata

class DRRLTransformer(nn.Module):
    """
    Main Model Class: Integrates Transformer Blocks with the RL Policy.
    This class manages the interaction loop: 
    State -> Policy -> Action (Rank) -> Layer Execution -> Reward
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config['model']['d_model']
        self.vocab_size = 50257 # Example: GPT-2 vocab size
        
        # 1. Embeddings
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(config['model']['max_len'], self.d_model)
        self.drop = nn.Dropout(config['model']['dropout'])

        # 2. Stack of DR-RL Blocks
        self.layers = nn.ModuleList([
            DRRLTransformerBlock(config) for _ in range(config['model']['n_layers'])
        ])
        
        self.ln_f = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # 3. RL Policy Agent (Shared or Per-Layer? Paper implies shared for efficiency)
        self.policy = RankSelectionPolicy(config)
        
        # Action to Rank Mapping
        # e.g., Actions 0..6 map to Ranks 16..64
        self.r_min = config['low_rank']['r_min']
        self.r_max = config['low_rank']['r_max']
        step = 8
        self.available_ranks = list(range(self.r_min, self.r_max + 1, step))
        
        # Epsilon annealing for exploration
        self.epsilon = config['rl']['epsilon_start']

    def forward(self, input_ids, mask=None):
        """
        Forward pass with integrated Rank Selection.
        """
        batch_size, seq_len = input_ids.size()
        
        # Embeddings
        pos = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        # Storage for RL training
        all_log_probs = []
        all_metadata = [] # Stores FLOPs, etc.
        total_flops = 0
        
        # Initial previous action (start with max rank index)
        prev_action = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
        
        # --- Layer Loop ---
        for i, layer in enumerate(self.layers):
            
            # --- Step 1: State Extraction (Methodology 4.1) ---
            # Get layer statistics (Mean, Var, Norm)
            layer_stats = layer.attn.get_layer_stats().unsqueeze(0).expand(batch_size, -1)
            
            # --- Step 2: Safety Check (Methodology 4.3) ---
            # Ideally, we calculate perturbation for each potential rank.
            # For efficiency in this loop, we might assume all are safe or use a cached check.
            # Here, we create a placeholder safe_mask (True = Safe)
            safe_mask = torch.ones(batch_size, len(self.available_ranks), dtype=torch.bool, device=x.device)
            # TODO: Integrate 'is_action_safe' with 'compute_rank_transition_perturbation' here for rigorous check
            
            # --- Step 3: Agent Decision (Action) ---
            # x is the input to the current layer, serving as context h_t
            action_idx, log_prob, _ = self.policy.select_action(
                x, layer_stats, prev_action, epsilon=self.epsilon, safe_mask=safe_mask
            )
            
            # Map index to actual Rank integer
            # e.g., action 0 -> rank 16
            selected_rank_int = self.available_ranks[action_idx[0].item()] # Assuming Batch=1 or taking mode for simplicity
            # Note: For batched mixed ranks, implementation needs to handle variable rank per sample.
            # For simplicity here, we broadcast the decision of the first sample or use a unified rank.
            
            # --- Step 4: Layer Execution ---
            x, metadata = layer(x, mask=mask, rank=selected_rank_int)
            
            # Store traces
            all_log_probs.append(log_prob)
            all_metadata.append(metadata)
            total_flops += metadata['flops']
            
            # Update history
            prev_action = action_idx

        # Output Head
        x = self.ln_f(x)
        logits = self.head(x)

        return {
            "logits": logits,           # For Language Modeling Loss
            "log_probs": all_log_probs, # For RL PPO Loss
            "metadata": all_metadata,   # For Reward Calculation
            "total_flops": total_flops
        }