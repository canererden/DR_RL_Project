import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
import os
import sys

# Proje kök dizinini path'e ekle
sys.path.append(os.getcwd())

from src.models.transformer import DRRLTransformer
from src.rl.reward import compute_reward
from src.utils.perturbation import compute_rank_transition_perturbation

from src.data.data_loader import TextDataManager # <--- BU SATIRI EKLEYİN

# --- 1. Dummy Dataset (Gerçek veri yerine) ---S
class DummyTextDataset(Dataset):
    """
    Simulates tokenized text data for testing the pipeline.
    """
    def __init__(self, vocab_size, seq_len, num_samples=1000):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Input: tokens[:-1], Target: tokens[1:] (Standard LM task)
        return self.data[idx, :-1], self.data[idx, 1:]

# --- 2. Main Training Loop ---
def train():
    # Load Config (Normally from .yaml, here inline for simplicity)
    config = {
        'model': {'d_model': 128, 'n_heads': 4, 'n_layers': 2, 'd_ff': 512, 'dropout': 0.1, 'max_len': 128},
        'low_rank': {'r_min': 8, 'r_max': 32},
        'rl': {
            'alpha': 1.0,   # Accuracy weight
            'beta': 0.05,   # FLOPs penalty
            'gamma': 0.1,   # Perturbation penalty
            'epsilon_start': 0.5, # High exploration initially
            'epsilon_decay': 0.99
        },
        'training': {'batch_size': 16, 'lr': 1e-4, 'epochs': 5}
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {device}")

    # Initialize Model
    model = DRRLTransformer(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])
    
    # ESKİ KOD (SİLİN veya YORUMA ALIN):
    # dataset = DummyTextDataset(vocab_size=50257, seq_len=config['model']['max_len'])
    # loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    # YENİ KOD (AKTİF EDİN):
    print("⏳ Loading Wikitext-103...")
    data_manager = TextDataManager(
        dataset_name='wikitext-103', 
        batch_size=config['training']['batch_size'],
        max_len=config['model']['max_len']
    )
    # train_loader otomatik olarak HuggingFace'den indirip tokenize edecek
    train_loader, val_loader = data_manager.load_data()
    
    # --- DEĞİŞİKLİK BİTİŞİ ---
    
    criterion_lm = nn.CrossEntropyLoss()

    # --- Training Epochs ---
    for epoch in range(config['training']['epochs']):
        total_lm_loss = 0
        total_rl_loss = 0
        total_rewards = 0
        
        # Anneal Epsilon (Exploration Rate)
        model.epsilon *= config['rl']['epsilon_decay']
        
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']} | Epsilon: {model.epsilon:.4f}")
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            targets = batch['labels'].to(device)
            optimizer.zero_grad()
            
            # 1. Forward Pass (Model selects ranks internally)
            output = model(input_ids)
            
            # 2. Language Modeling Loss (Supervised)
            # Flatten logits: (Batch * Seq, Vocab)
            logits = output['logits'].reshape(-1, output['logits'].size(-1))
            flat_targets = targets.reshape(-1)
            lm_loss = criterion_lm(logits, flat_targets)
            
            # 3. Reinforcement Learning Loss (Policy Gradient)
            # We need to calculate Reward R_t for each layer's decision
            rl_loss = 0
            batch_reward_sum = 0
            
            # Iterate through layers to compute rewards
            # Note: In a real implementation, we would need 'Full-Rank' output to compute Cosine Sim accurately.
            # Here we use Perturbation & FLOPs as proxies for efficiency.
            for i, metadata in enumerate(output['metadata']):
                
                # Retrieve traces
                flops = metadata['flops']
                # Dummy tensors for simulation (Real logic needs full rank output)
                dummy_full = torch.randn(1, 128, 128) 
                dummy_low = torch.randn(1, 128, 128)
                
                # Approximate Perturbation (Eq. 8)
                # In practice, we use the singular values computed inside attention
                pert_norm = 0.1 # Placeholder: Assume low perturbation for now
                
                # Calculate Reward (Eq. 12)
                reward = compute_reward(
                    dummy_full, dummy_low, flops, pert_norm, config
                )
                
                # REINFORCE Algorithm: Loss = - log_prob * Reward
                # We want to maximize Reward, so we minimize negative Reward.
                # output['log_probs'][i] is the log_prob of the action taken at layer i
                log_prob = output['log_probs'][i]
                
                # Check if log_prob requires grad (it should, coming from Agent)
                if log_prob.requires_grad:
                    rl_loss -= log_prob.mean() * reward # Mean over batch
                
                batch_reward_sum += reward.item()

            # 4. Total Loss & Backprop
            # alpha_loss balances LM vs RL objective
            total_loss = lm_loss + 0.1 * rl_loss 
            
            total_loss.backward()
            optimizer.step()
            
            total_lm_loss += lm_loss.item()
            total_rl_loss += rl_loss.item() if isinstance(rl_loss, torch.Tensor) else 0
            total_rewards += batch_reward_sum

            if batch_idx % 10 == 0:
                print(f"   Step {batch_idx} | LM Loss: {lm_loss.item():.4f} | Avg Reward: {batch_reward_sum/config['model']['n_layers']:.4f}")

        print(f"Epoch Summary -> Avg LM Loss: {total_lm_loss/len(loader):.4f} | Avg RL Loss: {total_rl_loss/len(loader):.4f}")

    # Save Model
    torch.save(model.state_dict(), "results/checkpoints/dr_rl_model.pth")
    print("✅ Model saved successfully.")

if __name__ == "__main__":
    train()