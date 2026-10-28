"""Closed-form V0-S: the stationary point of the CGM training objective with a linear
encoder (Experiment 7; docs/experiment-log.md, Exp 7 history, 2026-10-28).

Why: the V0-S init (`ConsensusGenerativeHead.init_from_moments`) fits the generative
model's moments (B B^T = within-tile noise W, A = exact posterior), but training
optimizes a different objective. Per source view s of a tile with V views
(`sipe.training.consensus_gen_module`: NLL averaged over every ordered (s, t) pair,
self included; KL per source):

    J = mean_t E_q[-log N(u_t; delta_k(t) + L z, W_k(t))] + KL(q || N(0, I)),
    q(z | u_s) = N(M u_s, S_q),   S_q = A A^T,

i.e. predict a random view of the tile from one view. With a linear encoder (r = 0),
linear generator (h = 0) and b = 0, J is a function of second moments only. With
y_t = u_t - delta_k(t), w_k = share of target views from scanner k,
P_uu = E[u_s u_s^T], P_yu^k = E[y_t u_s^T | t from scanner k] and
R = M P_uu M^T + S_q, the block minimizers are

    S_q = (I + L^T Wbar^-1 L)^-1,           Wbar^-1 = sum_k w_k W_k^-1
    L   = Wbar sum_k w_k W_k^-1 P_yu^k M^T R^-1
    M   = (L^T Wbar^-1 L + I)^-1 sum_k w_k L^T W_k^-1 P_yu^k P_uu^-1
    W_k = E_k := P_yy^k - L M P_uy^k - P_yu^k M^T L^T + L R L^T
          (shared: W = sum_k w_k E_k).

The L update is explicit because every tile has every scanner and the loss averages
over every source of the tile, so the source moments given the target scanner are the
same P_uu for every k (exact for complete tiles). delta_k = the scanner mean of u and
b = 0 are already stationary (u is centered over all views). Block coordinate descent
from the moment init: J never increases. `noise` = "shared" (one W, the init's
assumption) or "per_scanner" (W_k, as trained V0-S: per_scanner=True).

Null space: features can satisfy an exact linear constraint (H0-mini GAP: one
direction with P_uu eigenvalue ~1e-17, the final LayerNorm). There every E_k is
singular, log|W| -> -inf and J is unbounded below; the training loss never sees it
(B_k keeps the init's eigenvalue floor, the data carry nothing there). The fixed
point therefore runs in the data subspace Q (eigenvalues of P_uu above
`null_tol` x the largest): u' = Q^T u, z keeps D dims, and the map is set to
L = Q L', M = M' Q^T (zero response to and output in the null space, where every
input is zero anyway).

Pairing unit: tile-level pairs over all scanners of the views array. The map is a
CGMMap with r = h = None (the evaluated V0-S map form), center / scale / delta from
the init.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from sipe.model.consensus_gen import CGMMap, ConsensusGenerativeHead

NOISE = ("shared", "per_scanner")


@dataclass
class FixedPoint:
    map: CGMMap
    noise: torch.Tensor  # [K, D', D'] W_k in the data subspace (shared: K copies)
    posterior: torch.Tensor  # [D, D] S_q
    basis: torch.Tensor  # [D, D'] Q, orthonormal basis of the data subspace
    snapshots: dict[int, CGMMap] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)
    converged: bool = False


def _sym(a: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a + a.T)


def moments(u: torch.Tensor, delta: torch.Tensor) -> dict[str, torch.Tensor]:
    """u [n, V, D] standardized views (view j = scanner j), delta [V, D]."""
    n, v, _ = u.shape
    y = u - delta[None]
    ubar = u.mean(1)  # E over the source of the tile
    flat = u.reshape(n * v, -1)
    return {
        "P_uu": flat.T @ flat / (n * v),
        "P_yu": torch.einsum("nkd,ne->kde", y, ubar) / n,  # [V, D, D]
        "P_yy": torch.einsum("nkd,nke->kde", y, y) / n,
    }


def objective(
    L: torch.Tensor,
    M: torch.Tensor,
    S_q: torch.Tensor,
    W: torch.Tensor,
    mom: dict[str, torch.Tensor],
) -> torch.Tensor:
    """J per source view (no constants): sum_k w_k [tr(W_k^-1 E_k) + log|W_k|] / 2
    + (tr S_q + tr(M P_uu M^T) - log|S_q|) / 2. Differentiable (tests)."""
    k = W.shape[0]
    R = M @ mom["P_uu"] @ M.T + S_q
    nll = 0.0
    for j in range(k):
        Pyu = mom["P_yu"][j]
        E = mom["P_yy"][j] - L @ M @ Pyu.T - Pyu @ M.T @ L.T + L @ R @ L.T
        nll = nll + (
            torch.trace(torch.linalg.solve(W[j], E)) + torch.linalg.slogdet(W[j])[1]
        ) / (2 * k)
    kl = 0.5 * (
        torch.trace(S_q)
        + torch.trace(M @ mom["P_uu"] @ M.T)
        - torch.linalg.slogdet(S_q)[1]
    )
    return nll + kl


def fit(
    views: np.ndarray,
    device: torch.device,
    noise: str = "per_scanner",
    ridge: float = 1e-3,
    max_iter: int = 2000,
    tol: float = 1e-7,
    log_every: int = 0,
    log=None,
    null_tol: float = 1e-10,
    noise_weight: float = 1.0,
    snapshots: tuple[int, ...] = (),
) -> FixedPoint:
    """Fixed point from the moment init (`init_from_moments`, `ridge` as the init).

    views [n, V, D], view j = scanner j, every tile complete. Stops when the
    relative change of L M (the map's linear part) falls below `tol`. `snapshots`:
    iterations at which a copy of the map is kept (the path from the init, 0).
    `noise_weight`: start from the init with the noise scaled by beta
    (`init_from_moments(noise_weight=)`); the objective is unchanged."""
    if noise not in NOISE:
        raise ValueError(f"noise must be one of {NOISE}, got {noise!r}.")
    n, v, d = views.shape
    head = ConsensusGenerativeHead(d, num_scanners=v, per_scanner=True,
                                   encoder_hidden_dim=0).to(device)  # fmt: skip
    x = torch.tensor(views, dtype=torch.float64, device=device)
    head.init_from_moments(
        x.reshape(-1, d),
        torch.arange(v, device=device).repeat(n),
        torch.arange(n, device=device).repeat_interleave(v),
        ridge=ridge,
        noise_weight=noise_weight,
    )
    fmap = head.map
    center = fmap.center.double()
    scale = fmap.scale.double()
    delta = head.delta.detach().double()
    u = (x - center) / scale
    del x
    flat = u.reshape(n * v, d)
    evals, evecs = torch.linalg.eigh(flat.T @ flat / (n * v))
    Q = evecs[:, evals > null_tol * evals[-1]]  # [D, D'] data subspace
    del flat
    mom = moments(u @ Q, delta @ Q)
    del u
    eye = torch.eye(d, dtype=torch.float64, device=device)
    L = Q.T @ fmap.L.detach().double()  # [D', D]
    M = fmap.M.detach().double() @ Q  # [D, D']
    B = head.noise_factor()[0].detach().double()
    W = (Q.T @ B @ B.T @ Q).expand(v, Q.shape[1], Q.shape[1]).clone()
    post = head.posterior_factor()[0].detach().double()
    S_q = post @ post.T
    tr_w0 = float(torch.diagonal(W, dim1=-2, dim2=-1).sum(-1).mean())

    def record(it: int, change: float) -> None:
        row = {
            "iter": it,
            "J": float(objective(L, M, S_q, W, mom)) / d,
            "tr_noise_ratio": float(torch.diagonal(W, dim1=-2, dim2=-1).sum(-1).mean())
            / tr_w0,
            "tr_posterior": float(torch.trace(S_q)),
            "rel_change_LM": change,
        }
        out.history.append(row)
        if log is not None and log_every and it % log_every == 0:
            log("fixed point %s it %d: %s", noise, it, row)

    out = FixedPoint(map=fmap, noise=W, posterior=S_q, basis=Q)

    def snapshot(it: int) -> None:
        if it in snapshots:
            snap = copy.deepcopy(fmap)
            with torch.no_grad():
                snap.L.copy_(Q @ L)
                snap.M.copy_(M @ Q.T)
                snap.b.zero_()
            out.snapshots[it] = snap.cpu().eval()

    record(0, float("nan"))
    snapshot(0)
    T = L @ M
    w = 1.0 / v
    for it in range(1, max_iter + 1):
        W_inv = torch.linalg.inv(W)
        Wbar_inv = _sym(W_inv.mean(0))
        # S_q | L, W
        S_q = _sym(torch.linalg.inv(eye + L.T @ Wbar_inv @ L))
        # L | M, S_q, W
        R = _sym(M @ mom["P_uu"] @ M.T + S_q)
        G = (W_inv @ mom["P_yu"]).sum(0) * w  # sum_k w_k W_k^-1 P_yu^k
        L = torch.linalg.solve(Wbar_inv, G @ M.T) @ torch.linalg.inv(R)
        # M | L, S_q, W
        M = torch.linalg.solve(
            L.T @ Wbar_inv @ L + eye, L.T @ G
        ) @ torch.linalg.inv(mom["P_uu"])
        # W | L, M, S_q
        R = _sym(M @ mom["P_uu"] @ M.T + S_q)
        E = torch.stack(
            [
                _sym(mom["P_yy"][j] - L @ M @ mom["P_yu"][j].T
                     - mom["P_yu"][j] @ M.T @ L.T + L @ R @ L.T)
                for j in range(v)
            ]
        )  # fmt: skip
        W = E if noise == "per_scanner" else E.mean(0).expand_as(E).clone()
        T_new = L @ M
        change = float((T_new - T).norm() / T.norm())
        T = T_new
        out.noise, out.posterior = W, S_q
        snapshot(it)
        if change < tol or it == max_iter or (log_every and it % log_every == 0):
            record(it, change)
        if change < tol:
            out.converged = True
            break

    with torch.no_grad():
        fmap.L.copy_(Q @ L)
        fmap.M.copy_(M @ Q.T)
        fmap.b.zero_()
    out.map = fmap.cpu().eval()
    return out


def linear_distance(a: CGMMap, b: CGMMap) -> float:
    """||L_a M_a - L_b M_b||_F / ||L_b M_b||_F (the maps' linear parts, both with
    r = h = None; centers differ only through the train split used)."""
    ta = a.L.detach().double() @ a.M.detach().double()
    tb = b.L.detach().double() @ b.M.detach().double()
    return float((ta - tb).norm() / tb.norm())
