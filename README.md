# HCAN

`HCAN` is a small, plug and play PyTorch implementation of **HCAN**: a 
Hierarchical Classification Auxiliary Network, which consists of a tail layer
matched with its own loss function (`HCANLoss`) that can be added on top of any existing forecasting model.

The idea is to provide a more direct, modular implementation of the 
hierarchical head for multi-step, multi-channel forecasting that combines:

- **Direct regression** (`y_hat`) for the forecast values.
- **Hierarchical classification + regression** at two resolutions:
    - *Coarse* bins (e.g. up/down)
    - *Fine* bins (more granular quantile ranges)
- **Evidential (Dirichlet) uncertainty** via non-negative “evidence” outputs and an uncertainty-aware loss.

This architecture is based on the paper:
> **Hierarchical Classification and Regression for Multi-step Time Series Forecasting**  
> original implementation: https://github.com/syrGitHub/HCAN  
> original paper: https://arxiv.org/abs/2405.18975

The purpose of my implementation is to provide more direct, PyTorch module that can be plugged into existing forecasting
models.

## Features

The repo also includes an example training script on a tiny sample dataset (S\&P 500 return + volatility) under
`tests/`.

## What the model returns

`HCAN.forward(x)` returns a dictionary:

- `y_hat`: forecast values, shape `(B, T, C)`
- `coarse_delta`, `fine_delta`: per-class offsets within a bin, shapes `(B, T, C, Kc/Kf)`
- `coarse_evidence`, `fine_evidence`: non-negative evidence per class, shapes `(B, T, C, Kc/Kf)`
- `coarse_from_fine_evidence`: fine evidence summed into coarse buckets (used for consistency loss)

Where:

- `B` = batch size
- `T` = `pred_len` (forecast horizon)
- `C` = number of channels / target dimensions
- `Kc`, `Kf` = number of coarse / fine classes

## Loss

`HCANLoss` builds **quantile-based bin edges per channel** from a reference target history (`y_trues`) and computes a
weighted sum of:

- MSE on `y_hat`
- uncertainty-aware classification losses (Dirichlet / evidential)
- relative regression of within-bin offsets
- hierarchical consistency between coarse evidence and aggregated fine evidence

It returns a dict with `L_TOTAL` and the individual components.

## Minimal usage

```python
import torch
from hcan import HCAN, HCANLoss

B, T, C = 32, 21, 2
x = torch.randn(B, T, C)

model = HCAN(pred_len=T, channels=C)
outputs = model(x)

# y_trues should be representative of the full history of targets (any shape ending in C)
criterion = HCANLoss(y_trues=torch.randn(1000, T, C))
loss = criterion(outputs, torch.randn(B, T, C))["L_TOTAL"]
loss.backward()
```

## Example / demo script

`tests/test.py` trains a small LSTM encoder followed by an `HCAN` head on a sample parquet file.

From the repo root, run:

```powershell
python .\tests\test.py
```

## Test results

HCAN vs no HCAN after 200 training epochs achieves comparable return MSE but
far better sign accuracy, and lower MSE on volatility:

```
Training NoHCAN: 100%|██████████████████████████████| 200/200 [00:02<00:00, 80.23it/s, loss=8.91e-8]
NoHCAN sign accuracy: 0.0
NoHCAN returns MSE: 5.0933668e-05
NoHCAN volatility MSE: 5.2984535e-05

Training WithHCAN: 100%|████████████████████████████| 200/200 [00:05<00:00, 39.64it/s, loss=0.00362]
WithHCAN sign accuracy: 0.6043566362715299
WithHCAN returns MSE: 5.875215e-05
WithHCAN volatility MSE: 1.6380407e-05
```

Notes:

- The script reads `SPX_sample.parquet`. Run it from `tests/` or keep the working directory at the repo root (as above).
- It uses CUDA if available.

## Project layout

- `hcan/hcan.py`: HCAN module + `HCANLoss`
- `hcan/__init__.py`: package exports
- `tests/test.py`: runnable demo training loop
- `tests/SPX_sample.parquet`: sample dataset