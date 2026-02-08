# Dynamic Rank Reinforcement Learning (DR-RL)

[![License: MIT](https://github.com/canererden/DR_RL_Project/releases)](https://github.com/canererden/DR_RL_Project/releases)
[![Python](https://github.com/canererden/DR_RL_Project/releases%2B-blue)](https://github.com/canererden/DR_RL_Project/releases)
[![PyTorch](https://github.com/canererden/DR_RL_Project/releases%2B-orange)](https://github.com/canererden/DR_RL_Project/releases)
[![arXiv](https://github.com/canererden/DR_RL_Project/releases)](https://github.com/canererden/DR_RL_Project/releases)


> Official PyTorch implementation of the paper: **"Dynamic Rank Reinforcement Learning for Adaptive Low-Rank Multi-Head Self-Attention in Large Language Models"**

## Architecture

The framework consists of three core components integrated into the Transformer architecture:

1.  **Dynamic Low-Rank Attention:** Replaces standard MHSA. Uses Incremental SVD to approximate $Q$ and $K$ matrices based on the selected rank $r_t$.
2.  **RL Policy Network:** A lightweight Transformer encoder that observes sequence dynamics ($h_t$), layer statistics ($w_t$), and previous actions ($r_{t-1}$) to select the optimal rank.
3.  **Perturbation Guardrail:** Calculates $\|\Delta A\|_F$ to mask unsafe actions that exceed the stability threshold $\epsilon_t$.

## 📂 Project Structure

```bash
DR_RL_Project/
├── configs/               # Hyperparameter configurations (YAML)
├── src/
│   ├── data/              # Data loading & Tokenization pipeline (Wikitext-103, etc.)
│   ├── models/            # Core Architecture
│   │   ├── https://github.com/canererden/DR_RL_Project/releases   # Dynamic Low-Rank Attention Module (Eq. 1, 13, 14)
│   │   └── https://github.com/canererden/DR_RL_Project/releases # DR-RL Integrated Transformer
│   ├── rl/                # Reinforcement Learning Components
│   │   ├── https://github.com/canererden/DR_RL_Project/releases       # Policy Network (State-Action Logic)
│   │   └── https://github.com/canererden/DR_RL_Project/releases      # Reward Function (Fidelity vs. FLOPs) (Eq. 12)
│   └── utils/             # Math & Linear Algebra Backend
│       ├── https://github.com/canererden/DR_RL_Project/releases Perturbation Theory Bounds (Eq. 8, 9)
│       └── https://github.com/canererden/DR_RL_Project/releases   # Batched Partial SVD & Power Iteration
├── scripts/
│   └── https://github.com/canererden/DR_RL_Project/releases           # Main Training Loop (Hybrid: Supervised + RL)
├── https://github.com/canererden/DR_RL_Project/releases       # Dependencies
└── https://github.com/canererden/DR_RL_Project/releases       # Project initialization script

```

## Installation

### Prerequisites

* Linux or Windows (with CUDA support recommended)
* Python 3.8+
* NVIDIA GPU (Tested on A100, compatible with RTX series)

### Setup

1. Clone the repository:
```bash
git clone [https://github.com/canererden/DR_RL_Project/releases](https://github.com/canererden/DR_RL_Project/releases)
cd DR_RL_Project

```


2. Install dependencies:
```bash
pip install -r https://github.com/canererden/DR_RL_Project/releases

```


*Note: Ensure you have `torch` installed with CUDA support matching your driver.*

## Usage

### Training

The training script handles data downloading (Wikitext-103), preprocessing, and the hybrid training loop.

```bash
python https://github.com/canererden/DR_RL_Project/releases

```

The script performs the following steps:

1. **Data Loading:** Automatically downloads `wikitext-103-v1` via Hugging Face.
2. **Tokenization:** Processes text using the GPT-2 tokenizer.
3. **Optimization:** Runs the training loop optimizing the dual objective:
$$ \mathcal{L}*{total} = \mathcal{L}*{LM} + \lambda \cdot \mathcal{L}_{RL} $$

### Configuration

Hyperparameters can be adjusted in `https://github.com/canererden/DR_RL_Project/releases` (or moved to `https://github.com/canererden/DR_RL_Project/releases`):

* `r_min` / `r_max`: Bounds for rank selection (e.g., 16 to 64).
* `alpha`: Weight for Cosine Similarity reward.
* `beta`: Penalty weight for FLOPs.
* `gamma`: Penalty weight for Perturbation Norm.
* `epsilon_decay`: Exploration decay rate for the RL agent.

## Methodology & Equations

The core optimization is driven by the reward function R_t:

$$ R_t = \alpha \cdot \text{sim}(\mathbf{A}*{\text{full}}, \mathbf{A}*{r_t}) - \beta \cdot \text{FLOPs}(r_t) - \gamma \cdot |\Delta \mathbf{A}|_F $$

Where $$\|\Delta \mathbf{A}\|_F$$ is bounded by the tail energy of singular values:

$$ |\Delta \mathbf{A}|*F \approx \sqrt{\sum*{k=r+1}^{r'} \sigma_k^2} $$

## Contributing

This project is intended for research purposes. If you identify issues with the perturbation bounds or SVD implementation, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/canererden/DR_RL_Project/releases) file for details.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{erden2025drrl,
  title={Dynamic Rank Reinforcement Learning for Adaptive Low-Rank Multi-Head Self-Attention in Large Language Models},
  author={Erden, Caner},
  journal={arXiv preprint arXiv:2512.15973},
  volume={arXiv:2512.15973v1},
  url={https://github.com/canererden/DR_RL_Project/releases},
  year={2025}
}

```
