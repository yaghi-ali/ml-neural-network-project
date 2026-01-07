# ml-neural-network-project

# Neural Networks Study Project (Machine Learning)

This repository contains my study project on neural networks, focused on building, training, and evaluating models for supervised learning.
The work includes data preprocessing, model design (MLP/CNN), training strategies, and performance analysis.

## Objectives
- Implement a full ML pipeline: preprocessing → training → evaluation
- Compare model architectures (baseline vs neural networks)
- Track metrics and analyze errors (false positives/negatives where relevant)

## Project Structure
- `src/` : main Python code (dataset, model, training, evaluation)
- `notebooks/` : experiments and visual exploration (optional)
- `results/` : exported plots and metrics (figures, CSV/JSON)
- `data/` : local data (not pushed if large/private)

## Methods (Summary)
- Data processing: normalization, windowing (if time-series), train/val/test split
- Models: MLP (dense), optional CNN (1D) depending on signals/time-series
- Training: early stopping, learning-rate tuning, metric monitoring
- Evaluation: MAE/MSE for regression or accuracy/F1/AUROC for classification

## Results (Example)
Add your real results here when ready:
- Best model: MLP (3 dense layers)
- Metric: MAE = X.XX (validation), MSE = X.XX (test)
- Notes: improved stability after normalization + early stopping

## How to Run

### 1) Create environment & install dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
