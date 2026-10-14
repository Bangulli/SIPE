"""Experiment 3 (roadmap section 5): Gaussian mixture with content-dependent nuisance.

Model, per content regime k: c | k ~ N(mu_k, C_k), n | k ~ N(0, N_k), x = c + n.
Scanner-agnostic map (one embedding in, global parameters only):

    f(x) = sum_k p(k | x) [mu_k + P_k (x - mu_k)],   P_k = C_k (C_k + a N_k)^-1,
    p(k | x) ∝ pi_k N(W x; W mu_k, W (C_k + N_k) W^T).

a = 1 is the posterior mean of the plug-in model; a != 1 is a tuned attenuation (as
lambda of the soft-shrinkage eraser and alpha of roadmap Experiment 2A). Gating uses a
single view x, projected on the top `gate_dim` principal directions W of the train
views (the marginal of the same Gaussian model; full 768-d gates are near-hard and
dominated by poorly estimated low-variance directions).

Plug-in fit (not maximum likelihood of the latent model), pairing unit = tile-level
matched pairs: consensus c_i = mean over scanners of x_{i,s}, residual
r_{i,s} = x_{i,s} - c_i; regimes = k-means on the PCA of train consensus (the
clustering of `sipe.eval.content_dependence`, same seed -> same clusters). Per cluster:
mu_k = mean c, C_k = Cov(c) (ddof 0), N_k = E[r r^T]. These are the 2A definitions:
with K = 1 and a = 1 the map is the affine least-squares predictor of the consensus
(G0). N describes the marginal residual over scanners (no scanner-conditioned model).

Variants (roadmap ablation): G0 global Gaussian, K = 1; G1 mixture, shared N_0;
G2-scale N_k = s_k N_0 with the maximum-likelihood s_k = tr(N_0^-1 N_k) / D (Exp 1's
scale-only control used the trace ratio); G2 N_k = (1 - g) N_k + g N_0.
Regularization: C_k = (1 - b) C_k + b C_w (C_w = pooled within-cluster covariance)
and a ridge `ridge` x mean eigenvalue on every C_k and N_k.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

VARIANTS = ("G0", "G1", "G2-scale", "G2")


# --- plug-in moments ------------------------------------------------------------


def _ridge(m: np.ndarray, ridge: float) -> np.ndarray:
    d = m.shape[0]
    return m + ridge * np.trace(m) / d * np.eye(d)


def _second_moment(r: np.ndarray) -> np.ndarray:
    flat = r.reshape(-1, r.shape[-1])
    return flat.T @ flat / len(flat)


def _cov(c: np.ndarray) -> np.ndarray:
    centered = c - c.mean(axis=0)
    return centered.T @ centered / len(c)


@dataclass
class ContentClusters:
    """PCA + k-means of consensus vectors (as `sipe.eval.content_dependence.fit`)."""

    pca: PCA
    kmeans: KMeans

    @classmethod
    def fit(cls, consensus: np.ndarray, k: int, dim: int, seed: int):
        pca = PCA(n_components=dim, random_state=seed).fit(consensus)
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed)
        kmeans.fit(pca.transform(consensus))
        return cls(pca, kmeans)

    def assign(self, consensus: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(self.pca.transform(consensus))


@dataclass
class Moments:
    """Raw plug-in moments of paired views [n, S, D] grouped by content cluster."""

    pi: np.ndarray  # [K]
    mu: np.ndarray  # [K, D]
    C: np.ndarray  # [K, D, D] Cov(consensus | k)
    N: np.ndarray  # [K, D, D] E[r r^T | k]
    mu_all: np.ndarray  # [D]
    C_all: np.ndarray  # Cov(consensus), all units
    N_all: np.ndarray  # E[r r^T], all units = N_0

    @classmethod
    def fit(cls, views: np.ndarray, labels: np.ndarray, k: int) -> Moments:
        consensus = views.mean(axis=1)
        residuals = views - consensus[:, None]
        pi, mu, C, N = [], [], [], []
        for j in range(k):
            mask = labels == j
            if mask.sum() < 2:
                raise ValueError(f"Cluster {j} has {mask.sum()} units.")
            pi.append(mask.mean())
            mu.append(consensus[mask].mean(axis=0))
            C.append(_cov(consensus[mask]))
            N.append(_second_moment(residuals[mask]))
        return cls(
            pi=np.asarray(pi),
            mu=np.stack(mu),
            C=np.stack(C),
            N=np.stack(N),
            mu_all=consensus.mean(axis=0),
            C_all=_cov(consensus),
            N_all=_second_moment(residuals),
        )

    @property
    def C_within(self) -> np.ndarray:
        return np.einsum("k,kij->ij", self.pi, self.C)

    def content_covs(self, beta: float, ridge: float) -> np.ndarray:
        return np.stack(
            [_ridge((1 - beta) * c + beta * self.C_within, ridge) for c in self.C]
        )

    def nuisance_covs(self, variant: str, gamma: float, ridge: float) -> np.ndarray:
        n0 = self.N_all
        if variant == "G1":
            covs = [n0 for _ in self.N]
        elif variant == "G2-scale":
            # Maximum-likelihood scale of N(0, s N_0) for the cluster's residuals
            # (_ridge is relative, so ridge(s N_0) = s ridge(N_0)).
            prec = np.linalg.inv(_ridge(n0, ridge))
            d = n0.shape[0]
            covs = [np.trace(prec @ n) / d * n0 for n in self.N]
        elif variant == "G2":
            covs = [(1 - gamma) * n + gamma * n0 for n in self.N]
        else:
            raise ValueError(f"Unknown mixture variant {variant!r}.")
        return np.stack([_ridge(n, ridge) for n in covs])


@dataclass
class Mixture:
    """Fitted plug-in mixture: what the map and the likelihoods need."""

    variant: str
    pi: np.ndarray
    mu: np.ndarray
    C: np.ndarray
    N: np.ndarray
    beta: float = float("nan")
    gamma: float = float("nan")

    @property
    def k(self) -> int:
        return len(self.pi)

    @classmethod
    def build(
        cls,
        moments: Moments,
        variant: str,
        beta: float = 0.0,
        gamma: float = 0.0,
        ridge: float = 1e-4,
    ) -> Mixture:
        if variant == "G0":
            return cls(
                variant,
                np.ones(1),
                moments.mu_all[None],
                _ridge(moments.C_all, ridge)[None],
                _ridge(moments.N_all, ridge)[None],
            )
        return cls(
            variant,
            moments.pi,
            moments.mu,
            moments.content_covs(beta, ridge),
            moments.nuisance_covs(variant, gamma, ridge),
            beta=beta,
            gamma=gamma if variant == "G2" else float("nan"),
        )

    def operators(self, alpha: float) -> np.ndarray:
        """P_k = C_k (C_k + alpha N_k)^-1 (C, N symmetric)."""
        return np.stack(
            [np.linalg.solve(c + alpha * n, c).T for c, n in zip(self.C, self.N)]
        )


# --- likelihoods ----------------------------------------------------------------


def gaussian_ll(
    z: np.ndarray, mean: np.ndarray | float, cov: np.ndarray, device=None
) -> np.ndarray:
    """Log-density of the rows of z [..., d] under N(mean, cov), float64 (torch)."""
    d = cov.shape[0]
    t = torch.as_tensor(z.reshape(-1, d), dtype=torch.float64, device=device)
    t = t - torch.as_tensor(mean, dtype=torch.float64, device=device)
    chol = torch.linalg.cholesky(
        torch.as_tensor(cov, dtype=torch.float64, device=device)
    )
    sol = torch.linalg.solve_triangular(chol, t.T, upper=False)
    logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
    ll = -0.5 * (d * np.log(2 * np.pi) + logdet + (sol**2).sum(dim=0))
    return ll.cpu().numpy().reshape(z.shape[:-1])


def heldout_likelihoods(
    model: Mixture, views: np.ndarray, labels: np.ndarray | None, device=None
) -> dict[str, float]:
    """Mean log-likelihoods per dimension (nats/dim) of paired views [n, S, D].

    - `ll_consensus`: consensus c_i under N(mu_k, C_k), k = its content cluster;
    - `ll_residual`: residuals r_{i,s} under N(0, N_k), k = the unit's cluster (the
      nuisance likelihood of D3; G0 uses its single component);
    - `ll_x`: single views under the mixture marginal sum_k pi_k N(mu_k, C_k + N_k).
    """
    d = views.shape[-1]
    consensus = views.mean(axis=1)
    residuals = views - consensus[:, None]
    labels = np.zeros(len(views), int) if model.k == 1 else labels
    ll_c = np.empty(len(views))
    ll_r = np.empty(len(views))
    for j in range(model.k):
        mask = labels == j
        if mask.any():
            ll_c[mask] = gaussian_ll(consensus[mask], model.mu[j], model.C[j], device)
            ll_r[mask] = gaussian_ll(residuals[mask], 0.0, model.N[j], device).mean(
                axis=1
            )
    comp = np.stack(
        [
            np.log(p) + gaussian_ll(views, m, c + n, device)
            for p, m, c, n in zip(model.pi, model.mu, model.C, model.N)
        ]
    )
    ll_x = np.logaddexp.reduce(comp, axis=0)
    return {
        "ll_consensus": float(ll_c.mean() / d),
        "ll_residual": float(ll_r.mean() / d),
        "ll_x": float(ll_x.mean() / d),
    }


# --- the map --------------------------------------------------------------------


@dataclass
class Gate:
    """PCA projection of single views for p(k | x)."""

    center: np.ndarray  # [D]
    W: np.ndarray  # [d, D]

    @classmethod
    def fit(cls, x: np.ndarray, dim: int, seed: int) -> Gate:
        pca = PCA(n_components=dim, random_state=seed).fit(x)
        return cls(pca.mean_, pca.components_)


class MixtureShrinkageMap(nn.Module):
    """x [B, D] -> sum_k p(k | x) (mu_k + (x - mu_k) P_k^T); loadable by --posthoc."""

    def __init__(self, k: int, dim: int, gate_dim: int) -> None:
        super().__init__()
        self.register_buffer("mu", torch.zeros(k, dim))
        self.register_buffer("P", torch.zeros(k, dim, dim))
        self.register_buffer("gate_center", torch.zeros(dim))
        self.register_buffer("gate_W", torch.zeros(gate_dim, dim))
        self.register_buffer("gate_mean", torch.zeros(k, gate_dim))
        # Inverse Cholesky factor of each gate covariance and log pi_k - logdet / 2.
        self.register_buffer("gate_prec_chol", torch.zeros(k, gate_dim, gate_dim))
        self.register_buffer("gate_log_norm", torch.zeros(k))

    @classmethod
    def from_fit(cls, model: Mixture, gate: Gate, alpha: float) -> MixtureShrinkageMap:
        d = gate.W.shape[0]
        out = cls(model.k, model.mu.shape[1], d)
        gate_cov = np.einsum("ij,kjl,ml->kim", gate.W, model.C + model.N, gate.W)
        chol = np.linalg.cholesky(gate_cov)
        prec_chol = np.linalg.inv(chol)
        logdet = 2.0 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
        values = {
            "mu": model.mu,
            "P": model.operators(alpha),
            "gate_center": gate.center,
            "gate_W": gate.W,
            "gate_mean": (model.mu - gate.center) @ gate.W.T,
            "gate_prec_chol": prec_chol,
            "gate_log_norm": np.log(model.pi) - 0.5 * logdet,
        }
        for name, value in values.items():
            getattr(out, name).copy_(torch.from_numpy(np.asarray(value)))
        return out

    def responsibilities(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.gate_center) @ self.gate_W.T  # [B, d]
        u = torch.einsum(
            "kij,bkj->bki", self.gate_prec_chol, z[:, None] - self.gate_mean
        )
        return torch.softmax(self.gate_log_norm - 0.5 * (u**2).sum(dim=-1), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resp = self.responsibilities(x)  # [B, K]
        shrunk = torch.einsum("kij,bkj->bki", self.P, x[:, None] - self.mu) + self.mu
        return torch.einsum("bk,bki->bi", resp, shrunk)
