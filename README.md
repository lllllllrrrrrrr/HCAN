# hcan

`hcan` is a small PyTorch implementation of **HCAN**: a hierarchical head for multi-step, multi-channel forecasting that combines:

- **Direct regression** (`y_hat`) for the forecast values.
- **Hierarchical classification + regression** at two resolutions:
  - *Coarse* bins (e.g. up/down)
  - *Fine* bins (more granular quantile ranges)
- **Evidential (Dirichlet) uncertainty** via non-negative “evidence” outputs and an uncertainty-aware loss.

The repo also includes an example training script on a tiny sample dataset (S\&P 500 return + volatility) under `tests/`.

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

`HCANLoss` builds **quantile-based bin edges per channel** from a reference target history (`y_trues`) and computes a weighted sum of:

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

# y_trues should be a representative history of targets (any shape ending in C)
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

Notes:
- The script reads `SPX_sample.parquet`. Run it from `tests/` or keep the working directory at the repo root (as above).
- It uses CUDA if available.

## Project layout

- `hcan/hcan.py`: HCAN module + `HCANLoss`
- `hcan/__init__.py`: package exports
- `tests/test.py`: runnable demo training loop
- `tests/SPX_sample.parquet`: sample dataset

## License

No license file is included yet. If you plan to publish this, consider adding a `LICENSE` file.
