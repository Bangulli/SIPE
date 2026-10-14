"""Experiment 4 (roadmap section 6): analytic second-order correction, consensus base.

Map (one embedding in, global parameters only), u = x - mu:

    f(x) = mu + W_u u + W_q (q(u) - q_bar).

Closed-form ridge regression of the tile consensus on phi(u) = [u; q(u) - q_bar]:

    min_W  E_{i,s} ||W phi(u_{i,s}) - (c_i - mu)||^2 + g s_q ||W_q||_F^2,

with the penalty on the nonlinear block only and s_q = mean diagonal of
E[(q - q_bar)(q - q_bar)^T] (g is scale-free). Base objective = roadmap Experiment 2B
(consensus regression), not the eraser objective written in section 6 (user decision
2026-10-14: the linear limit must be the stronger G0, not the eraser).

Linear limit: the u block uses G0's ridged moments, E[u u^T] = C_r + N_r and
E[u (c - mu)^T] = C_r (exact for balanced empirical moments: sum_s r_{i,s} = 0 per
tile). So g -> inf gives W_q = 0 and W_u = C_r (C_r + N_r)^-1, the G0 map at a = 1 of
`sipe.estimators.mixture` (same `ridge`).

Pairing unit: tile-level matched pairs (consensus over all scanners). Features are
fixed before the regression, from train moments:
- nuisance directions v_j: top generalized eigenvectors of N_r v = rho (C_r + N_r) v
  (rho = nuisance share of the variance along v), unit variance of v^T u;
- content directions w_j: top principal directions of C_r, unit variance of w^T u;
- Q1: squares (v_j^T u)^2, j < m; Q2: products (w_j^T u)(v_k^T u), j, k < m;
  Q2-rand: Q2 with random orthonormal v_k (unit variance), the control that the
  nuisance directions matter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn

VARIANTS = ("Q1", "Q2", "Q2-rand")


def _unit_variance(dirs: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Scale rows d so that d^T cov d = 1."""
    var = np.einsum("ij,jk,ik->i", dirs, cov, dirs)
    return dirs / np.sqrt(var)[:, None]


def nuisance_directions(C: np.ndarray, N: np.ndarray, m: int) -> np.ndarray:
    """Top-m generalized eigenvectors of (N, C + N), rows, largest rho first."""
    _, vecs = scipy.linalg.eigh(N, C + N)  # ascending; v^T (C + N) v = 1
    return vecs[:, ::-1][:, :m].T.copy()


def content_directions(C: np.ndarray, N: np.ndarray, m: int) -> np.ndarray:
    _, vecs = np.linalg.eigh(C)
    return _unit_variance(vecs[:, ::-1][:, :m].T, C + N)


def random_directions(C: np.ndarray, N: np.ndarray, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(C.shape[0], m)))
    return _unit_variance(q.T, C + N)


@dataclass
class QuadraticFeatures:
    """q(u): squares of nuisance projections, or content x nuisance products."""

    nuisance: np.ndarray  # [m_v, D]
    content: np.ndarray | None = None  # [m_w, D]; None = squares (Q1)

    @property
    def n(self) -> int:
        m_v = len(self.nuisance)
        return m_v if self.content is None else m_v * len(self.content)

    def __call__(self, u: np.ndarray) -> np.ndarray:
        a = u @ self.nuisance.T
        if self.content is None:
            return a**2
        b = u @ self.content.T
        return (b[:, :, None] * a[:, None, :]).reshape(len(u), -1)

    @classmethod
    def build(
        cls, variant: str, C: np.ndarray, N: np.ndarray, m: int, seed: int = 0
    ) -> QuadraticFeatures:
        if variant == "Q1":
            return cls(nuisance_directions(C, N, m))
        if variant == "Q2":
            return cls(nuisance_directions(C, N, m), content_directions(C, N, m))
        if variant == "Q2-rand":
            return cls(random_directions(C, N, m, seed), content_directions(C, N, m))
        raise ValueError(f"Unknown quadratic variant {variant!r}.")


@dataclass
class QuadraticFit:
    """Moments of phi(u) and the consensus target; solve() for any g."""

    features: QuadraticFeatures
    mu: np.ndarray  # [D]
    q_mean: np.ndarray  # [m]
    G_uu: np.ndarray  # C_r + N_r
    H_u: np.ndarray  # C_r
    G_qu: np.ndarray  # [m, D] E[q~ u^T]
    G_qq: np.ndarray  # [m, m]
    H_q: np.ndarray  # [m, D] E[q~ (c - mu)^T]

    @classmethod
    def fit(
        cls,
        views: np.ndarray,
        features: QuadraticFeatures,
        mu: np.ndarray,
        C: np.ndarray,
        N: np.ndarray,
        batch: int = 4096,
    ) -> QuadraticFit:
        """views [n, S, D] float64; mu, C, N = G0's mean and ridged moments."""
        n, s, d = views.shape
        target = np.repeat(views.mean(axis=1) - mu, s, axis=0)
        u = views.reshape(-1, d) - mu
        q_sum = np.zeros(features.n)
        for i in range(0, len(u), batch):
            q_sum += features(u[i : i + batch]).sum(axis=0)
        q_mean = q_sum / len(u)
        m = features.n
        G_qu, G_qq, H_q = np.zeros((m, d)), np.zeros((m, m)), np.zeros((m, d))
        for i in range(0, len(u), batch):
            ub = u[i : i + batch]
            q = features(ub) - q_mean
            G_qu += q.T @ ub
            G_qq += q.T @ q
            H_q += q.T @ target[i : i + batch]
        return cls(features, mu, q_mean, C + N, C, G_qu / len(u), G_qq / len(u),
                   H_q / len(u))  # fmt: skip

    @property
    def scale(self) -> float:
        return float(np.trace(self.G_qq) / len(self.G_qq))

    def solve(self, gamma: float) -> tuple[np.ndarray, np.ndarray]:
        """(W_u [D, D], W_q [D, m]); gamma = inf is the linear G0 map."""
        d, m = len(self.mu), len(self.G_qq)
        if np.isinf(gamma):
            return np.linalg.solve(self.G_uu, self.H_u).T, np.zeros((d, m))
        A = np.block(
            [
                [self.G_uu, self.G_qu.T],
                [self.G_qu, self.G_qq + gamma * self.scale * np.eye(m)],
            ]
        )
        X = scipy.linalg.solve(A, np.vstack([self.H_u, self.H_q]), assume_a="sym")
        return X[:d].T, X[d:].T

    def predict(self, x: np.ndarray, gamma: float) -> np.ndarray:
        W_u, W_q = self.solve(gamma)
        u = x - self.mu
        return self.mu + u @ W_u.T + (self.features(u) - self.q_mean) @ W_q.T


class QuadraticShrinkageMap(nn.Module):
    """x [B, D] -> mu + W_u u + W_q (q(u) - q_bar); loadable by --posthoc."""

    def __init__(self, dim: int, n_nuisance: int, n_content: int = 0) -> None:
        super().__init__()
        m = n_nuisance * (n_content or 1)
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("W_u", torch.zeros(dim, dim))
        self.register_buffer("nuisance_dirs", torch.zeros(n_nuisance, dim))
        self.register_buffer("content_dirs", torch.zeros(n_content, dim))
        self.register_buffer("q_mean", torch.zeros(m))
        self.register_buffer("W_q", torch.zeros(dim, m))

    @classmethod
    def from_fit(cls, fit: QuadraticFit, gamma: float) -> QuadraticShrinkageMap:
        feats = fit.features
        n_content = 0 if feats.content is None else len(feats.content)
        out = cls(len(fit.mu), len(feats.nuisance), n_content)
        W_u, W_q = fit.solve(gamma)
        values = {
            "mu": fit.mu,
            "W_u": W_u,
            "nuisance_dirs": feats.nuisance,
            "content_dirs": np.zeros((0, len(fit.mu))) if n_content == 0
            else feats.content,
            "q_mean": fit.q_mean,
            "W_q": W_q,
        }  # fmt: skip
        for name, value in values.items():
            getattr(out, name).copy_(torch.from_numpy(np.asarray(value)))
        return out

    def features(self, u: torch.Tensor) -> torch.Tensor:
        a = u @ self.nuisance_dirs.T
        if self.content_dirs.shape[0] == 0:
            return a**2
        b = u @ self.content_dirs.T
        return (b[:, :, None] * a[:, None, :]).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x - self.mu
        return self.mu + u @ self.W_u.T + (self.features(u) - self.q_mean) @ self.W_q.T
