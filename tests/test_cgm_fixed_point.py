"""Closed-form V0-S (sipe.estimators.cgm_fixed_point): the moment objective is the
training loss in expectation, the fixed point is stationary, J never increases."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sipe.estimators import cgm_fixed_point as fp
from sipe.model.consensus_gen import ConsensusGenerativeHead
from sipe.training.consensus_gen_module import tile_view_pairs

CPU = torch.device("cpu")


def _views(n=400, v=3, d=6, seed=0, null=False):
    rng = np.random.default_rng(seed)
    content = rng.normal(size=(n, 1, d)) @ rng.normal(size=(d, d))
    offsets = rng.normal(size=(1, v, d))
    # scanner-dependent noise scale, so per-scanner W_k differ
    noise = rng.normal(size=(n, v, d)) * np.linspace(0.5, 1.5, v)[None, :, None]
    x = content + offsets + noise
    if null:  # an exact linear constraint on every view (as H0-mini's LayerNorm)
        a = np.ones(d) / np.sqrt(d)
        x = x - (x @ a)[..., None] * a
    return x


def _state(views, noise, max_iter):
    out = fp.fit(views, CPU, noise=noise, max_iter=max_iter, tol=0.0)
    n, v, d = views.shape
    fmap = out.map
    x = torch.tensor(views, dtype=torch.float64)
    u = (x - fmap.center.double()) / fmap.scale.double()
    head = ConsensusGenerativeHead(d, v, per_scanner=True, encoder_hidden_dim=0)
    head.init_from_moments(
        x.reshape(-1, d), torch.arange(v).repeat(n),
        torch.arange(n).repeat_interleave(v), ridge=1e-3,
    )  # fmt: skip
    delta = head.delta.detach().double()
    return out, u, delta, fp.moments(u, delta)


@pytest.mark.parametrize("noise", fp.NOISE)
def test_objective_is_the_training_loss_in_expectation(noise):
    views = _views()
    out, u, delta, mom = _state(views, noise, max_iter=3)
    n, v, d = u.shape
    L = out.map.L.detach().double()
    M = out.map.M.detach().double()
    Q = out.basis  # full rank here: a rotation of the feature space
    W, S_q = Q @ out.noise @ Q.T, out.posterior
    # Training loss of sipe.training.consensus_gen_module with z = m and the
    # sampling term E_q added analytically: tr(W_k^-1 L S_q L^T) / 2.
    head = ConsensusGenerativeHead(d, v, per_scanner=True, encoder_hidden_dim=0)
    head = head.double()
    with torch.no_grad():
        head.map.L.copy_(L)
        head.map.M.copy_(M)
        head.delta.copy_(delta)
        head.noise_factor.set_(torch.linalg.cholesky(W))
        head.posterior_factor.set_(torch.linalg.cholesky(S_q)[None])
    flat = u.reshape(-1, d)
    scanners = torch.arange(v).repeat(n)
    source, target = tile_view_pairs(torch.arange(n).repeat_interleave(v))
    m = flat @ M.T
    g = m @ L.T
    nll = head.nll(flat[target], g[source], scanners[target]).mean()
    W_inv = torch.linalg.inv(W)
    sampling = torch.stack(
        [0.5 * torch.trace(W_inv[k] @ L @ S_q @ L.T) for k in range(v)]
    )[scanners[target]].mean()
    kl = head.kl(m).mean()
    loss = nll + sampling + kl
    const = 0.5 * d * np.log(2 * np.pi) - 0.5 * d
    rotated = fp.moments(u @ Q, delta @ Q)
    expected = fp.objective(Q.T @ L, M @ Q, S_q, out.noise, rotated) + const
    torch.testing.assert_close(fp.objective(L, M, S_q, W, mom), expected - const)
    torch.testing.assert_close(loss, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("noise", fp.NOISE)
def test_fixed_point_is_stationary_and_monotone(noise):
    views = _views()
    out = fp.fit(views, CPU, noise=noise, max_iter=3000, tol=1e-12, log_every=1)
    J = [h["J"] for h in out.history]
    assert all(b <= a + 1e-12 for a, b in zip(J, J[1:]))
    assert J[-1] < J[0]
    _, u, delta, _ = _state(views, noise, max_iter=1)
    Q = out.basis
    assert Q.shape == (6, 6)
    mom = fp.moments(u @ Q, delta @ Q)
    L = (Q.T @ out.map.L.detach().double()).requires_grad_()
    M = (out.map.M.detach().double() @ Q).requires_grad_()
    S_q = out.posterior.clone().requires_grad_()
    if noise == "shared":
        W0 = out.noise[0].clone().requires_grad_()
        W = W0.expand(u.shape[1], -1, -1)
        params = [L, M, S_q, W0]
    else:
        W = out.noise.clone().requires_grad_()
        params = [L, M, S_q, W]
    grads = torch.autograd.grad(fp.objective(L, M, S_q, W, mom), params)
    for p, gr in zip(params, grads):
        assert gr.norm() < 1e-6 * max(1.0, p.norm()), gr.norm()


def test_shared_noise_is_one_matrix_and_inflated():
    views = _views()
    out = fp.fit(views, CPU, noise="shared", max_iter=500)
    torch.testing.assert_close(out.noise[0], out.noise[2])
    # predicting another view from one view adds to the within-tile noise
    assert out.history[-1]["tr_noise_ratio"] > 1.0
    assert fp.linear_distance(out.map, out.map) == 0.0


def test_null_direction_is_projected_out():
    views = _views(null=True)
    out = fp.fit(views, CPU, noise="per_scanner", max_iter=3000, tol=1e-10,
                 log_every=1)  # fmt: skip
    assert out.converged and out.basis.shape == (6, 5)
    J = [h["J"] for h in out.history]
    assert all(b <= a + 1e-12 for a, b in zip(J, J[1:]))
    a = torch.ones(6, dtype=torch.float64) / 6**0.5
    T = out.map.L.detach().double() @ out.map.M.detach().double()
    assert (T @ a).norm() < 1e-8 and (a @ T).norm() < 1e-8
