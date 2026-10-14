from __future__ import annotations

import torch
from torch.distributions import MultivariateNormal, kl_divergence

from sipe.bench.posthoc import load_posthoc_map
from sipe.model.consensus_gen import CGMMap, ConsensusGenerativeHead
from sipe.training.consensus_gen_module import tile_view_pairs

DIM, SCANNERS = 6, 3


def _views(tiles: int = 400, seed: int = 0):
    """Linear-Gaussian views: x = mu + W s + offset_k + noise, all scanners per tile."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(DIM, DIM, generator=g)
    offsets = torch.randn(SCANNERS, DIM, generator=g)
    signal = torch.randn(tiles, DIM, generator=g) @ W.T + 3.0
    x = signal[:, None] + offsets + 0.5 * torch.randn(tiles, SCANNERS, DIM, generator=g)
    scanners = torch.arange(SCANNERS).repeat(tiles)
    locations = torch.arange(tiles).repeat_interleave(SCANNERS)
    return x.reshape(-1, DIM), scanners, locations


def _head(per_scanner: bool = True, hidden: int = 8) -> ConsensusGenerativeHead:
    torch.manual_seed(0)
    head = ConsensusGenerativeHead(DIM, SCANNERS, per_scanner, hidden, hidden).double()
    head.init_from_moments(*_views())
    return head


def test_init_is_linear_gaussian_posterior() -> None:
    head = _head()
    fmap = head.map
    x = _views(seed=1)[0]
    u = fmap.standardize(x)
    # Zero-initialized residual layers: f is the linear posterior-mean map.
    expected = fmap.center + fmap.scale * (u @ fmap.M.T @ fmap.L.T)
    torch.testing.assert_close(fmap(x), expected)
    # M = L^T Cov(u)^-1 with Cov(u) = L L^T + N0 + offsets; A A^T = I - M L.
    A = head.posterior_factor()[0]
    eye = torch.eye(DIM, dtype=A.dtype)
    torch.testing.assert_close(A @ A.T, eye - fmap.M @ fmap.L, atol=1e-6, rtol=1e-6)
    # The map moves views of a tile toward each other.
    views = fmap(x).reshape(-1, SCANNERS, DIM)
    raw = x.reshape(-1, SCANNERS, DIM)
    assert views.var(1).sum(-1).mean() < raw.var(1).sum(-1).mean()


def test_nll_and_kl_match_dense() -> None:
    head = _head()
    with torch.no_grad():
        head.noise_factor.offdiag.normal_(0, 0.1)
        head.posterior_factor.offdiag.normal_(0, 0.1)
    u = torch.randn(9, DIM, dtype=torch.double)
    g = torch.randn(9, DIM, dtype=torch.double)
    k = torch.tensor([0, 1, 2] * 3)
    B = head.noise_factor()
    dense = torch.stack(
        [
            -MultivariateNormal(g[i] + head.delta[k[i]], scale_tril=B[k[i]]).log_prob(
                u[i]
            )
            for i in range(9)
        ]
    )
    torch.testing.assert_close(head.nll(u, g, k), dense)
    m = torch.randn(4, DIM, dtype=torch.double)
    A = head.posterior_factor()[0]
    prior = MultivariateNormal(
        torch.zeros(DIM, dtype=A.dtype), torch.eye(DIM, dtype=A.dtype)
    )
    dense_kl = torch.stack(
        [kl_divergence(MultivariateNormal(mi, scale_tril=A), prior) for mi in m]
    )
    torch.testing.assert_close(head.kl(m), dense_kl)


def test_per_scanner_equals_exchangeable_without_offsets() -> None:
    per, exch = _head(True), _head(False)
    with torch.no_grad():
        per.delta.zero_()
        per.noise_factor.set_(exch.noise_factor())
    u = torch.randn(6, DIM, dtype=torch.double)
    g = torch.randn(6, DIM, dtype=torch.double)
    k = torch.tensor([0, 1, 2, 2, 1, 0])
    torch.testing.assert_close(per.nll(u, g, k), exch.nll(u, g, k))


def test_tile_view_pairs() -> None:
    source, target = tile_view_pairs(torch.tensor([0, 0, 1, 1, 1]))
    pairs = set(zip(source.tolist(), target.tolist()))
    expected = {(a, b) for a in (0, 1) for b in (0, 1)}
    expected |= {(a, b) for a in (2, 3, 4) for b in (2, 3, 4)}
    assert pairs == expected


def test_map_file_roundtrip(tmp_path) -> None:
    head = _head().float()
    with torch.no_grad():
        head.map.r[-1].weight.normal_(0, 0.1)  # make the nonlinear part non-trivial
    path = tmp_path / "map_cgm.pt"
    torch.save(
        {
            "class_path": f"{CGMMap.__module__}.CGMMap",
            "init_args": {
                "dim": DIM,
                "encoder_hidden_dim": 8,
                "generator_hidden_dim": 8,
            },
            "state_dict": head.map.state_dict(),
            "pooling": "gap",
        },
        path,
    )
    loaded = load_posthoc_map(path)
    x = torch.randn(5, DIM)
    torch.testing.assert_close(loaded.map(x), head.map.eval()(x))
