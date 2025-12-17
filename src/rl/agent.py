import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class StateFeatureExtractor(nn.Module):
    """
    Extracts Sequence Dynamics (h_t) from input embeddings via Conv1D.
    Part of State Space S_t in Methodology 4.1.
    """
    def __init__(self, input_dim, feature_dim=64):
        super().__init__()
        # Lightweight convolution to capture local patterns
        self.conv = nn.Conv1d(input_dim, feature_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1) # Summarize sequence to a single vector

    def forward(self, x):
        # x: (Batch, Seq, Dim) -> (Batch, Dim, Seq) for Conv1D
        x = x.transpose(1, 2)
        feat = F.relu(self.conv(x))
        feat = self.pool(feat).squeeze(-1) # (Batch, FeatureDim)
        return feat

class RankSelectionPolicy(nn.Module):
    """
    Transformer-based Policy Network for Dynamic Rank Selection.
    Implements Methodology 4.1 & 4.5.
    
    Structure:
    State -> [FeatureExtractor + LayerStats + PrevRank] -> Transformer Encoder -> MLP -> Logits
    """
    def __init__(self, config):
        super().__init__()
        
        # Hyperparameters
        self.d_model_input = config['model']['d_model']
        self.state_dim = 128 # Hidden dimension for Policy Network
        self.n_actions = (config['low_rank']['r_max'] - config['low_rank']['r_min']) // 8 + 1
        # Example: 16, 24, 32, 40, 48, 56, 64 -> 7 actions
        
        # 1. State Components Processing
        self.seq_extractor = StateFeatureExtractor(self.d_model_input, feature_dim=64)
        self.layer_stat_proj = nn.Linear(9, 32) # Mean, Var, Norm
        self.prev_rank_emb = nn.Embedding(self.n_actions, 32) # Embed previous decision
        
        # 2. Transformer Encoder (The "Lightweight GPT" part)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. Action Head (MLP)
        self.action_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions)
        )

    def forward(self, x, layer_stats, prev_action_idx):
        """
        Args:
            x: Input embeddings (Batch, Seq, Dim)
            layer_stats: (Batch, 3) statistics from Attention layer
            prev_action_idx: (Batch,) index of previously selected rank
        
        Returns:
            logits: (Batch, n_actions)
        """
        # Feature Extraction
        h_t = self.seq_extractor(x)           # (B, 64)
        w_t = self.layer_stat_proj(layer_stats) # (B, 32)
        r_prev = self.prev_rank_emb(prev_action_idx) # (B, 32)
        
        # Concatenate State: s_t = [h_t, w_t, r_{t-1}]
        state = torch.cat([h_t, w_t, r_prev], dim=-1) # (B, 128)
        
        # Transformer expects sequence dimension, pretend seq_len=1 for this decision step
        state = state.unsqueeze(1) # (B, 1, 128)
        
        encoded_state = self.transformer_encoder(state)
        
        # Predict Action Logits
        logits = self.action_head(encoded_state.squeeze(1)) # (B, n_actions)
        
        return logits

    def select_action(self, x, layer_stats, prev_action_idx, epsilon=0.0, safe_mask=None):
        """
        Epsilon-Greedy Action Selection with Safety Masking.
        
        Args:
            safe_mask: Boolean tensor (Batch, n_actions). False indicates unsafe ranks (high perturbation).
                       If provided, unsafe actions are masked out (set to -inf).
        """
        logits = self.forward(x, layer_stats, prev_action_idx)
        
        # Apply Safety Mask (Methodology 4.3)
        if safe_mask is not None:
            logits = logits.masked_fill(~safe_mask, -1e9)

        probs = F.softmax(logits, dim=-1)
        
        # Epsilon-Greedy Logic for Exploration
        if self.training and torch.rand(1).item() < epsilon:
            # Random sampling from allowed (safe) actions
            if safe_mask is not None:
                # Uniform prob over safe actions
                uniform_safe = safe_mask.float() / safe_mask.sum(dim=-1, keepdim=True)
                action_dist = Categorical(uniform_safe)
            else:
                action_dist = Categorical(torch.ones_like(probs) / self.n_actions)
        else:
            # Policy sampling
            action_dist = Categorical(probs)
            
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action, log_prob, probs