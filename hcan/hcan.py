from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def kl_dirichlet(alpha: Tensor, beta: Tensor, eps: float = 1e-6) -> Tensor:
    alpha = alpha.clamp_min(eps)
    beta = beta.clamp_min(eps)
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    sum_beta = torch.sum(beta, dim=-1, keepdim=True)

    term1 = torch.lgamma(sum_alpha) - torch.lgamma(sum_beta)
    term2 = (torch.lgamma(beta) - torch.lgamma(alpha)).sum(dim=-1, keepdim=True)
    term3 = ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=-1, keepdim=True)
    return (term1 + term2 + term3).squeeze(-1)


def symmetric_kl_divergence(p: Tensor, q: Tensor, eps: float = 1e-8) -> Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p_norm = p / p.sum(dim=-1, keepdim=True)
    q_norm = q / q.sum(dim=-1, keepdim=True)

    kl_pq = F.kl_div(q_norm.log(), p_norm, reduction="batchmean", log_target=False)
    kl_qp = F.kl_div(p_norm.log(), q_norm, reduction="batchmean", log_target=False)
    return 0.5 * (kl_pq + kl_qp)


def un_ce_loss_from_logits(labels: Tensor,
                           evidence_logits: Tensor,
                           num_classes: int,
                           global_step: int,
                           annealing_step: int,
                           ) -> tuple[Tensor, Tensor, Tensor]:
    labels = labels.long()
    alpha = F.softplus(evidence_logits) + 1
    S = alpha.sum(dim=-1, keepdim=True)
    E = alpha - 1.
    onehot = F.one_hot(labels, num_classes).to(dtype=alpha.dtype)

    bu = E / S
    lam = 1. - bu
    A = (lam * onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1, keepdim=True)

    anneal = 1. if annealing_step <= 0 else min(1., float(global_step) / float(annealing_step))
    alp = E * (1. - onehot) + 1.
    B = anneal * kl_dirichlet(alp, torch.ones_like(alp)).unsqueeze(-1)

    pred = alpha / S
    uncertainty = float(num_classes) / S
    return (A + B).mean(), pred, uncertainty


class GroupTokenizer(nn.Module):
    def __init__(self,
                 num_classes: int,
                 y_trues: Tensor,
                 method: Literal['quantile', 'uniform'] = 'quantile',
                 eps: float = 1e-12,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.register_buffer("left_edges", torch.empty(0))  # C, K
        self.register_buffer("right_edges", torch.empty(0))  # C, K
        self._fit(y_trues, method)
        return

    @torch.no_grad()
    def _fit(self, y_trues: Tensor, method: Literal['quantile', 'uniform']) -> None:
        if y_trues.ndim == 1:
            y_trues = y_trues.unsqueeze(-1)

        C = y_trues.size(-1)
        l = torch.empty((C, self.num_classes), device=y_trues.device, dtype=y_trues.dtype)
        r = torch.empty((C, self.num_classes), device=y_trues.device, dtype=y_trues.dtype)
        if method == 'quantile':
            l, r = self._fit_quantile(y_trues, l, r)
        elif method == 'uniform':
            l, r = self._fit_uniform(y_trues, l, r)
        else:
            raise ValueError(f"Unknown method {method}")
        self.left_edges = l
        self.right_edges = r
        return

    @torch.no_grad()
    def _fit_quantile(self, flat: Tensor, l: Tensor, r: Tensor) -> tuple[Tensor, Tensor]:
        L = flat.size(0)
        idx = torch.arange(0, self.num_classes + 1, device=flat.device, dtype=flat.dtype)
        idx = torch.floor(idx * (L - 1) / self.num_classes).long().clamp(max=L - 1).long()

        for c in range(flat.size(-1)):
            vals, _ = torch.sort(flat[:, c])
            mn, mx = vals[0], vals[-1]
            for i in range(self.num_classes):
                l[c, i] = mn if i == 0 else vals[idx[i]]
                r[c, i] = mx if i == self.num_classes - 1 else vals[idx[i + 1]]
        return l, r

    @torch.no_grad()
    def _fit_uniform(self, flat: Tensor, l: Tensor, r: Tensor) -> tuple[Tensor, Tensor]:
        for c in range(flat.size(-1)):
            mn, mx = flat[:, c].min(), flat[:, c].max()

            # Create K bins of equal width from min to max
            edges = torch.linspace(mn, mx, self.num_classes + 1, device=flat.device)
            l[c, :] = edges[:-1]
            r[c, :] = edges[1:]
            r[c, -1] = mx # Ensure the last bin includes the max value
        return l, r

    def tokenize(self, y: Tensor) -> tuple[Tensor, Tensor]:
        if y.ndim == 2:
            y = y.unsqueeze(-1)

        B, T, C = y.size()
        K = self.num_classes

        left = self.left_edges.to(device=y.device, dtype=y.dtype)
        right = self.right_edges.to(device=y.device, dtype=y.dtype)

        y_exp = y.unsqueeze(-1)  # (B,T,C,1)
        in_left = y_exp >= left.unsqueeze(0).unsqueeze(0)  # (B,T,C,K)
        in_right = y_exp < right.unsqueeze(0).unsqueeze(0)  # (B,T,C,K)
        in_bin = in_left & in_right

        has_bin = in_bin.any(dim=-1)  # (B,T,C)
        labels = torch.argmax(in_bin.to(torch.int64), dim=-1)  # (B,T,C) arbitrary if no bin
        labels = torch.where(has_bin, labels, torch.full_like(labels, K - 1))

        left_sel = left.unsqueeze(0).unsqueeze(0).expand(B, T, C, K).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(
            -1)
        right_sel = right.unsqueeze(0).unsqueeze(0).expand(B, T, C, K).gather(dim=-1,
                                                                              index=labels.unsqueeze(-1)).squeeze(-1)

        width = (right_sel - left_sel).clamp_min(self.eps)
        delta = ((y - left_sel) / width).clamp(0, 1)  # (B,T,C)

        reg = y.new_full((B, T, C, K), -1.)
        reg.scatter_(dim=-1, index=labels.unsqueeze(-1), src=delta.unsqueeze(-1))
        return labels, reg


def group_average_logits(fine_logits: Tensor, num_coarse: int) -> Tensor:
    Kf = fine_logits.size(-1)
    assert Kf % num_coarse == 0
    g = Kf // num_coarse
    return fine_logits.view(*fine_logits.shape[:-1], num_coarse, g).mean(dim=-1)


class HCANClassifier(nn.Module):
    def __init__(self, pred_len: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.proj = nn.Linear(pred_len, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, pred_len * num_classes)
        self.classifier = nn.Linear(hidden_dim, pred_len * num_classes)
        return

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, C, T = x.shape
        if T != self.pred_len:
            raise ValueError(f"Expected input shape (B,{self.pred_len},C), got {tuple(x.shape)}")

        h = self.proj(x)  # (B,C,H)
        pred = self.predictor(h)  # (B,C, T * K)
        logit = self.classifier(h)  # (B,C, T * K)
        pred = pred.view(B, C, T, self.num_classes).permute(0, 2, 1, 3).contiguous()  # (B,T,C,K)
        logit = logit.view(B, C, T, self.num_classes).permute(0, 2, 1, 3).contiguous()  # (B,T,C,K)
        return h, pred, logit


class HCAN(nn.Module):
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
        self.hidden_dim = hidden_dim
        self.num_coarse = num_coarse
        self.num_fine = num_fine

        # Direct
        self.g_proj1 = nn.Linear(pred_len, hidden_dim)
        self.g_proj2 = nn.Linear(hidden_dim, pred_len)
        self.predictor = nn.Linear(pred_len, pred_len)

        self.coarse = HCANClassifier(pred_len=pred_len, hidden_dim=hidden_dim, num_classes=num_coarse)
        self.fine = HCANClassifier(pred_len=pred_len, hidden_dim=hidden_dim, num_classes=num_fine)
        return

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        B, T, C = x.shape
        assert C == self.channels, f"Expected input with {self.channels} channels, got {C}"
        assert T == self.pred_len, f"Expected input with {self.pred_len} time steps, got {T}"

        x_ct = x.permute(0, 2, 1).contiguous()  # (B,C,T)
        h_c, coarse_pred, coarse_logit = self.coarse(x_ct)
        h_f, fine_pred, fine_logit = self.fine(x_ct)

        fine_logit_switch = group_average_logits(fine_logit, self.num_coarse)  # (B,T,C,Kc)

        g_proj = self.g_proj1(x_ct)  # (B,C,H)
        att = torch.matmul(h_c, h_f.transpose(1, 2)) / (self.hidden_dim ** 0.5)  # (B,C,C)
        att = F.softmax(att, dim=-1)
        g_att = torch.matmul(att, g_proj)  # (B,C,H)
        g_out = self.g_proj2(g_att)  # (B,C,T)
        z = g_out + x_ct  # (B,C,T)
        y_hat = self.predictor(z).permute(0, 2, 1).contiguous()  # (B,T,C)
        return {
            "y_hat": y_hat,
            "coarse_logit": coarse_logit,
            "coarse_prediction": coarse_pred,
            "fine_logit": fine_logit,
            "fine_prediction": fine_pred,
            'fine_logit_switch': fine_logit_switch,
        }


class HCANLoss(nn.Module):
    """
    Loss with target normalization:
    - Normalize y (per channel) using train statistics: y_norm = (y - mu) / sigma
    - Build bin bounds in normalized space
    - Compute point MSE in normalized space so it's O(1)
    """

    def __init__(self, *,
                 y_trues: Tensor,
                 hcan: HCAN,
                 lambda_cls: float = 1.0,
                 lambda_reg: float = 1.0,
                 lambda_acl: float = 1.0,
                 lambda_direct: float = 1.0,
                 annealing_step: int = 1000,
                 reg_loss: Literal['mse', 'smooth_l1'] = 'smooth_l1',
                 ) -> None:
        super().__init__()
        self.num_coarse = hcan.coarse.num_classes
        self.num_fine = hcan.fine.num_classes
        self.annealing_step = annealing_step
        self.lambda_direct = lambda_direct
        self.lambda_acl = lambda_acl
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls

        if reg_loss == 'mse':
            self.reg_loss = nn.MSELoss()
        else:
            self.reg_loss = nn.SmoothL1Loss()

        self.register_buffer("_global_step", torch.zeros((), dtype=torch.long))
        self.coarse_tok = GroupTokenizer(num_classes=self.num_coarse, y_trues=y_trues)
        self.fine_tok = GroupTokenizer(num_classes=self.num_fine, y_trues=y_trues)
        return

    def _reg(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        # pred, target: (B,T,C), mask: (B,T,C)
        if mask.sum() == 0:
            return pred.new_tensor(0.)
        return self.reg_loss(pred[mask], target[mask])

    def forward(self, x: dict[str, Tensor], y: Tensor) -> Tensor:
        if y.ndim == 2:
            y = y.unsqueeze(-1)

        self._global_step += 1
        global_step = int(self._global_step.item())

        label_c, reg_c = self.coarse_tok.tokenize(y)  # (B,T,C), (B,T,C, Kc)
        label_f, reg_f = self.fine_tok.tokenize(y)  # (B,T,C), (B,T,C, Kf)

        y_hat = x["y_hat"]  # (B,T,C)
        coarse_pred = x["coarse_prediction"]  # (B,T,C,Kc)
        fine_pred = x["fine_prediction"]  # (B,T,C,Kf)
        fine_logit_switch = x['fine_logit_switch']  # (B,T,C,Kc)
        coarse_logit = x['coarse_logit']  # (B,T,C,Kc)
        fine_logit = x['fine_logit']  # (B,T,C,Kf)

        loss_acl = symmetric_kl_divergence(
            F.softmax(coarse_logit, dim=-1),
            F.softmax(fine_logit_switch, dim=-1),
        )

        B, T, C, Kc = coarse_logit.shape
        coarse_uac, coarse_prob, coarse_unc = un_ce_loss_from_logits(
            labels=label_c.reshape(B, -1),
            evidence_logits=coarse_logit.reshape(B, -1, Kc),
            num_classes=self.num_coarse,
            global_step=global_step,
            annealing_step=self.annealing_step,
        )

        Kf = fine_logit.shape[-1]
        fine_uac, fine_prob, fine_unc = un_ce_loss_from_logits(
            labels=label_f.reshape(B, -1),
            evidence_logits=fine_logit.reshape(B, -1, Kf),
            num_classes=self.num_fine,
            global_step=global_step,
            annealing_step=self.annealing_step,
        )

        mask_c, mask_f = reg_c >= 0, reg_f >= 0
        loss_reg_c = self._reg(coarse_pred, reg_c, mask_c)
        loss_reg_f = self._reg(fine_pred, reg_f, mask_f)

        loss_direct = F.mse_loss(y_hat, y, reduction='mean')
        return (
                self.lambda_direct * loss_direct +
                self.lambda_acl * loss_acl +
                self.lambda_cls * (coarse_uac + fine_uac) +
                self.lambda_reg * (loss_reg_c + loss_reg_f)
        )
