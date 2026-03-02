from typing import Literal

import torch
from .hcan import HCANClassifier, GroupTokenizer
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimpleHCAN(nn.Module):
    def __init__(self, *,
                 pred_len: int,
                 channels: int,
                 hidden_dim: int,
                 num_coarse: int = 2,
                 num_fine: int = 4,
                 ) -> None:
        super().__init__()
        assert num_fine % num_coarse == 0, "num_fine must be a multiple of num_coarse"
        self.pred_len = pred_len
        self.channels = channels

        # Hierarchical classifiers
        self.coarse_classifier = HCANClassifier(pred_len=pred_len, hidden_dim=hidden_dim, num_classes=num_coarse)
        self.fine_classifier = HCANClassifier(pred_len=pred_len, hidden_dim=hidden_dim, num_classes=num_fine)
        self.x_projector = nn.Linear(pred_len, hidden_dim)

        # Path to combine features and predict
        self.h_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, pred_len)
        return

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        B, T, C = x.shape
        assert C == self.channels, f"Expected input with {self.channels} channels, got {C}"
        assert T == self.pred_len, f"Expected input with {self.pred_len} time steps, got {T}"

        x_ct = x.permute(0, 2, 1).contiguous()  # (B,C,T)

        # Hierarchical classification logits and hidden states
        h_c, _, coarse_logit = self.coarse_classifier(x_ct)
        h_f, _, fine_logit = self.fine_classifier(x_ct)

        # Combine hidden states from classifiers
        h_combined = torch.cat([h_c, h_f], dim=-1)  # (B, C, H*2)
        h_proj = F.relu(self.h_proj(h_combined))  # (B, C, H)

        # Add input features (as a residual connection) to the combined classifier features
        combined_features = h_proj + self.x_projector(x_ct)  # (B, C, H)

        # Regression prediction from combined features
        y_hat = self.predictor(combined_features).permute(0, 2, 1).contiguous()  # (B,T,C)
        return {
            "y_hat": y_hat,
            "coarse_logit": coarse_logit,
            "fine_logit": fine_logit,
        }


class SimpleHCANLoss(nn.Module):
    def __init__(self, *,
                 y_trues: Tensor,  # (N, C) tensor of true values for each class, used for tokenization
                 hcan: SimpleHCAN,
                 lambda_cls: float = 1.0,
                 ) -> None:
        super().__init__()
        self.lambda_cls = lambda_cls
        self.toks = {
            'coarse': GroupTokenizer(num_classes=hcan.coarse_classifier.num_classes, y_trues=y_trues),
            'fine': GroupTokenizer(num_classes=hcan.fine_classifier.num_classes, y_trues=y_trues),
        }
        return

    def _aux_loss(self, phase: Literal['coarse', 'fine'], x: dict[str, Tensor], y: Tensor) -> Tensor:
        return F.cross_entropy(
            x[f'{phase}_logit'].permute(0, 3, 1, 2),  # (B,T,C,Kc)
            self.toks[phase].tokenize(y)[0],  # (B,T,C)
            reduction='none',
        )

    def forward(self, x: dict[str, Tensor], y: Tensor) -> Tensor:
        if y.ndim == 2:
            y = y.unsqueeze(-1)

        aux_loss = sum(self._aux_loss(k, x, y) for k in ['coarse', 'fine'])

        loss_direct = F.mse_loss(x["y_hat"], y, reduction='none')
        total_loss = loss_direct + self.lambda_cls * aux_loss
        return total_loss.mean()
