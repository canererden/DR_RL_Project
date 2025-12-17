# Dynamic Rank Reinforcement Learning (DR-RL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research_Prototype-green)]()

> Official PyTorch implementation of the paper: **"Dynamic Rank Reinforcement Learning for Adaptive Low-Rank Multi-Head Self-Attention in Large Language Models"**

## Abstract

Large Language Models (LLMs) suffer from quadratic computational complexity in their Multi-Head Self-Attention (MHSA) mechanisms. Traditional low-rank approximations rely on static rank assumptions, failing to capture the dynamic linguistic complexity of different input sequences.

**DR-RL** is a novel framework that bridges the gap between theoretical rigor and adaptive efficiency. It formulates rank selection as a sequential decision-making problem, optimized via a **Transformer-based Reinforcement Learning agent**. The framework leverages **Online Matrix Perturbation Theory** to guarantee stability during rank transitions, ensuring that efficiency gains do not compromise model fidelity.

Key achievements:
* **~40% FLOPs Reduction** in long-sequence regimes ($L > 4096$).
* **Pareto-Optimal Performance:** Matches full-rank perplexity with significantly lower computational cost.
* **Mathematically Grounded:** Rank updates are constrained by spectral perturbation bounds.

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
│   │   ├── attention.py   # Dynamic Low-Rank Attention Module (Eq. 1, 13, 14)
│   │   └── transformer.py # DR-RL Integrated Transformer
│   ├── rl/                # Reinforcement Learning Components
│   │   ├── agent.py       # Policy Network (State-Action Logic)
│   │   └── reward.py      # Reward Function (Fidelity vs. FLOPs) (Eq. 12)
│   └── utils/             # Math & Linear Algebra Backend
│       ├── perturbation.py# Perturbation Theory Bounds (Eq. 8, 9)
│       └── svd_utils.py   # Batched Partial SVD & Power Iteration
├── scripts/
│   └── train.py           # Main Training Loop (Hybrid: Supervised + RL)
├── requirements.txt       # Dependencies
└── setup_project.py       # Project initialization script

```

## Installation

### Prerequisites

* Linux or Windows (with CUDA support recommended)
* Python 3.8+
* NVIDIA GPU (Tested on A100, compatible with RTX series)

### Setup

1. Clone the repository:
```bash
git clone [https://github.com/username/DR-RL.git](https://github.com/username/DR-RL.git)
cd DR_RL_Project

```


2. Install dependencies:
```bash
pip install -r requirements.txt

```


*Note: Ensure you have `torch` installed with CUDA support matching your driver.*

## Usage

### Training

The training script handles data downloading (Wikitext-103), preprocessing, and the hybrid training loop.

```bash
python scripts/train.py

```

The script performs the following steps:

1. **Data Loading:** Automatically downloads `wikitext-103-v1` via Hugging Face.
2. **Tokenization:** Processes text using the GPT-2 tokenizer.
3. **Optimization:** Runs the training loop optimizing the dual objective:
$$ \mathcal{L}*{total} = \mathcal{L}*{LM} + \lambda \cdot \mathcal{L}_{RL} $$

### Configuration

Hyperparameters can be adjusted in `scripts/train.py` (or moved to `configs/default_config.yaml`):

* `r_min` / `r_max`: Bounds for rank selection (e.g., 16 to 64).
* `alpha`: Weight for Cosine Similarity reward.
* `beta`: Penalty weight for FLOPs.
* `gamma`: Penalty weight for Perturbation Norm.
* `epsilon_decay`: Exploration decay rate for the RL agent.

## Methodology & Equations

The core optimization is driven by the reward function R_t:

$$ R_t = \alpha \cdot \text{sim}(\mathbf{A}*{\text{full}}, \mathbf{A}*{r_t}) - \beta \cdot \text{FLOPs}(r_t) - \gamma \cdot |\Delta \mathbf{A}|_F $$

Where \|\Delta \mathbf{A}\|_F is bounded by the tail energy of singular values:

$$ |\Delta \mathbf{A}|*F \approx \sqrt{\sum*{k=r+1}^{r'} \sigma_k^2} $$

## Contributing

This project is intended for research purposes. If you identify issues with the perturbation bounds or SVD implementation, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{erden2025drrl,
  title={Dynamic Rank Reinforcement Learning for Adaptive Low-Rank Multi-Head Self-Attention in Large Language Models},
  author={Erden, Caner},
  journal={arXiv preprint},
  year={2025}
}

```