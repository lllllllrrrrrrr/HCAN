import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def kl_dirichlet(alpha: Tensor, beta: Tensor) -> Tensor:
    """
    Kullback-Leibler divergence D_KL(P||Q) for two Dirichelet distributions P and Q
    with parameters alpha and beta respectively.
    """
    alpha = alpha.clamp_min(1e-6)
    beta = beta.clamp_min(1e-6)
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    term1 = torch.lgamma(sum_alpha) - torch.lgamma(torch.sum(beta, dim=-1, keepdim=True))
    term2 = (torch.lgamma(beta) - torch.lgamma(alpha)).sum(dim=-1, keepdim=True)
    term3 = ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=-1, keepdim=True)
    return (term1 + term2 + term3).squeeze(-1)


def evidence_to_dirichlet(evidence: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Evidence e >= 0 -> Dirichet params a=e+1, strength S, expected probs p, belief b
    """
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=-1, keepdim=True)
    p = alpha / S
    b = evidence / S
    return alpha, S, p, b


class HCANClassifier(nn.Module):
    def __init__(self,
                 pred_len: int,
                 hidden_dim: int,
                 num_classes: int,
                 ):
        super().__init__()
        self.pred_len = pred_len
        self.num_classes = num_classes

        self.proj = nn.Linear(pred_len, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, pred_len * num_classes)  # delta reg per class
        self.classifier = nn.Linear(hidden_dim, pred_len * num_classes)  # evidence per class
        self.softplus = nn.Softplus()
        return

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        :param x: Tensor (B, C, T)
        :return:
          h:     (B, C, H) - hidden representation
          delta: (B, T, C, K) - predicted offsets per class
          e:     (B, T, C, K) - evidence per class
        """
        B, C, T = x.shape
        assert T == self.pred_len, f"Expected T == pred_len == {self.pred_len}, got T={T}"

        h = self.proj(x)  # (B, C, H)
        delta = self.predictor(h)  # (B, C, T*K)
        delta = delta.view(B, C, T, self.num_classes).permute(0, 2, 1, 3).contiguous()  # (B, T, C, K)

        e = self.classifier(h)  # (B, C, T*K)
        e = self.softplus(e)
        e = e.view(B, C, T, self.num_classes).permute(0, 2, 1, 3).contiguous()  # (B, T, C, K)
        return h, delta, e


class HCAN(nn.Module):
    def __init__(self,
                 pred_len: int,
                 channels: int,
                 hidden_dim: int = 64,
                 num_coarse: int = 2,
                 num_fine: int = 4,
                 ):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.pred_len = pred_len

        # direct path
        self.g_proj1 = nn.Linear(pred_len, hidden_dim)
        self.g_proj2 = nn.Linear(hidden_dim, pred_len)
        self.predictor = nn.Linear(pred_len, pred_len)

        self.coarse = HCANClassifier(pred_len, hidden_dim, num_coarse)
        self.fine = HCANClassifier(pred_len, hidden_dim, num_fine)
        return

    def pairwise_sum(self, x: Tensor) -> Tensor:
        """
        Convert fine class outputs to coarse class outputs by summing pairs

        :param x: Tensor (B, T, C, Kf) where Kf is even and Kc = Kf //2
        :return: Tensor (B, T, C, Kc)
        """
        Kf = x.size()[-1]
        assert Kf % 2 == 0
        even = x[..., 0:Kf:2]
        odd = x[..., 1:Kf:2]
        return even + odd

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Forward pass of HCAN model.

        :param x: Tensor (B, T, C) where T = pred_len, C = channels
        """
        B, T, C = x.size()
        assert T == self.pred_len
        assert C == self.channels

        x_ct = x.permute(0, 2, 1).contiguous()  # (B, C, T)

        # classifier paths
        c, coarse_delta, coarse_e = self.coarse(x_ct)
        f, fine_delta, fine_e = self.fine(x_ct)

        # from fine to coarse
        coarse_from_fine_evidence = self.pairwise_sum(fine_e)  # (B, T, C, Kc)

        # direct path
        g = self.g_proj1(x_ct)  # (B, C, hidden_dim)

        f_mat = torch.matmul(c, f.transpose(1, 2))
        att = F.softmax(f_mat, dim=-1)

        # attend g across channels
        g_att = torch.matmul(att, g)  # (B, C, hidden_dim)
        g_out = self.g_proj2(g_att)  # (B, C, T)

        # residual with original x_ct, then predictor over T
        g_out = g_out + x_ct  # (B, C, T)
        y_hat = self.predictor(g_out)  # (B, C, T)
        y_hat = y_hat.permute(0, 2, 1).contiguous()  # (B, T, C)

        return {
            'y_hat': y_hat,  # (B, T, C)
            'coarse_delta': coarse_delta,  # (B, T, C, Kc)
            'coarse_evidence': coarse_e,  # (B, T, C, Kc)
            'fine_delta': fine_delta,  # (B, T, C, Kf)
            'fine_evidence': fine_e,  # (B, T, C, Kf)
            'coarse_from_fine_evidence': coarse_from_fine_evidence,  # (B, T, C, Kc)
        }


class HCANLoss(nn.Module):
    def __init__(self, *,
                 y_trues: Tensor,
                 num_coarse: int = 2,
                 num_fine: int = 4,
                 w_uac: float = 1,
                 w_rr: float = 1,
                 w_mse: float = 1,
                 w_hcl: float = 1,
                 w_ua: float = 1,
                 w_kl: float = 1,
                 ) -> None:
        super().__init__()
        assert num_fine % num_coarse == 0, "num_fine must be multiple of num_coarse"
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.w_uac = w_uac
        self.w_rr = w_rr
        self.w_mse = w_mse
        self.w_hcl = w_hcl
        self.w_ua = w_ua
        self.w_kl = w_kl

        # set global quantile edges per channel
        edges_f, edges_c = [], []
        for c in range(y_trues.size(-1)):
            edges_c.append(self._build_bin_edges(y_trues[..., c], num_bins=num_coarse))
            edges_f.append(self._build_bin_edges(y_trues[..., c], num_bins=num_fine))

        self.register_buffer("bin_edges_f", torch.stack(edges_f, dim=0))  # (C, Kf-1)
        self.register_buffer("bin_edges_c", torch.stack(edges_c, dim=0))
        return

    @staticmethod
    def _build_bin_edges(y_trues: Tensor, num_bins: int) -> Tensor:
        """
        Build global quantile-based bin edges (single channel).

        :param y_trues: Tensor
        :param num_bins: number of bins
        :return: Tensor (num_bins+1)
        """
        with torch.no_grad():
            q = torch.linspace(
                start=0,
                end=1,
                steps=num_bins + 1,
                device=y_trues.device,
                dtype=y_trues.dtype,
            )[1:-1]  # exclude 0 and 1
            return torch.quantile(y_trues.flatten(), q=q)

    @staticmethod
    def _discretize_and_offsets(y: Tensor, edges: Tensor) -> tuple[Tensor, Tensor]:
        """
        Per_channel discretization of targets and offsets within bins.
        :param edges:
        :return:
        """
        B, T, C = y.size()
        assert edges.ndim == 2 and edges.shape[0] == C

        cmp = y.unsqueeze(-1) > edges.unsqueeze(0).unsqueeze(0)  # (B, T, C, K-1)
        k_idx = cmp.sum(dim=-1)

        # build finite left-edges per-channel
        y_min_c = y.amin(dim=(0, 1))  # (C, )
        left_edges = torch.cat([y_min_c.unsqueeze(-1), edges], dim=-1)  # (C, K)

        # gather left edges for each (B, T, C) based on k_idx
        left_edge_k = left_edges.unsqueeze(0).unsqueeze(0).expand(B, T, C, -1)  # (B, T, C, K)
        left = torch.gather(left_edge_k, dim=-1, index=k_idx.unsqueeze(-1)).squeeze(-1)  # (B, T, C)

        delta = y - left  # (B, T, C)
        return k_idx, delta

    def _uncertainty_aware_loss(self, evidence: Tensor, target_onehot: Tensor) -> Tensor:
        """
        Uncertainty-aware loss component.

        :param evidence: Tensor (B, T, C, K)
        :param target_onehot: Tensor (B, T, C, K)
        :return: loss Tensor scalar average over (B, T, C)
        """
        dg = torch.special.digamma
        alpha, S, _, b = evidence_to_dirichlet(evidence)  # (B, T, C, K)
        alpha_tilde = target_onehot + (1.0 - target_onehot) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=-1, keepdim=True)
        b_tilde = (alpha_tilde - 1) / S_tilde
        omega = (1.0 - b_tilde) * target_onehot

        ua = (omega * dg(S_tilde) - dg(alpha_tilde)).sum(dim=-1)  # (B, T, C)
        kl = kl_dirichlet(alpha_tilde, torch.ones_like(alpha_tilde))  # (B, T, C)

        return self.w_ua * ua.mean() + self.w_kl * kl.mean()

    def _relative_regression_loss(self, delta_pred: Tensor, delta_target: Tensor, target_onehot: Tensor) -> Tensor:
        """
        Relative regression loss component.

        :param delta_pred: Tensor (B, T, C, K)
        :param delta_target: Tensor (B, T, C)
        :return: loss Tensor scalar average over (B, T, C)
        """
        x = (delta_pred * target_onehot).sum(dim=-1)  # (B, T, C)
        return F.smooth_l1_loss(x, delta_target, reduction='mean')

    def _hcl(self, coarse_e: Tensor, coarse_from_fine_e: Tensor) -> Tensor:
        """
        Hierarchical consistency loss component.

        :param coarse_e: Tensor (B, T, C, Kc)
        :param coarse_from_fine_e: Tensor (B, T, C, Kc)
        """
        p = torch.softmax(coarse_e, dim=-1)
        q = torch.softmax(coarse_from_fine_e, dim=-1)
        kl1 = (p * (p.clamp_min(1e-8).log() - q.clamp_min(1e-8).log())).sum(dim=-1)  # (B, T, C)
        kl2 = (q * (q.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())).sum(dim=-1)
        return ((kl1 + kl2) * .5).mean()

    def forward(self, outputs: dict[str, Tensor], y: Tensor) -> dict[str, Tensor]:
        """
        Compute HCAN loss
        :param outputs: HCAN layer outputs
         :param coarse_delta: Tensor (B, T, C, Kc)
         :param coarse_evidence: Tensor (B, T, C, Kc)
         :param fine_delta: Tensor (B, T, C, Kf)
         :param fine_evidence: Tensor (B, T, C, Kf)
         :param coarse_from_fine_evidence: Tensor (B, T, C, Kc)
        :param y: Tensor (B, T, C)
        """
        y_hat = outputs['y_hat']
        coarse_evidence = outputs['coarse_evidence']
        coarse_delta = outputs['coarse_delta']
        fine_delta = outputs['fine_delta']
        fine_evidence = outputs['fine_evidence']
        coarse_from_fine_evidence = outputs['coarse_from_fine_evidence']

        # discretize per (timestep, channel)
        c_f_idx, delta_f_true = self._discretize_and_offsets(y, self.bin_edges_f)  # (B, T, C), (B, T, C)
        c_c_idx, delta_c_true = self._discretize_and_offsets(y, self.bin_edges_c)  # (B, T, C), (B, T, C)

        # onehot targets
        f_onehot = F.one_hot(c_f_idx, num_classes=self.num_fine).to(dtype=y.dtype)  # (B, T, C, Kf)
        c_onehot = F.one_hot(c_c_idx, num_classes=self.num_coarse).to(dtype=y.dtype)  # (B, T, C, Kc)

        # losses
        L_f_UAC = self._uncertainty_aware_loss(fine_evidence, f_onehot)
        L_f_RR = self._relative_regression_loss(fine_delta, delta_f_true, f_onehot)

        L_c_UAC = self._uncertainty_aware_loss(coarse_evidence, c_onehot)
        L_c_RR = self._relative_regression_loss(coarse_delta, delta_c_true, c_onehot)

        L_HCL = self._hcl(coarse_evidence, coarse_from_fine_evidence)
        L_MSE = F.mse_loss(y_hat, y, reduction='mean')

        L_HIER = L_f_UAC + self.w_rr * L_f_RR + self.w_uac * (L_c_UAC + self.w_rr * L_c_RR)
        L_TOTAL = L_HIER + self.w_mse * L_MSE + self.w_hcl * L_HCL
        return {
            'L_TOTAL': L_TOTAL,
            'L_HIER': L_HIER,
            'L_MSE': L_MSE,
            'L_HCL': L_HCL,
            'L_f_UAC': L_f_UAC,
            'L_f_RR': L_f_RR,
            'L_c_UAC': L_c_UAC,
            'L_c_RR': L_c_RR,
        }
