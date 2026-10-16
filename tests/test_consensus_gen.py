from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch.distributions import MultivariateNormal, kl_divergence

from sipe.bench.posthoc import load_posthoc_map
from sipe.model.consensus_gen import (
    CGMMap,
    ConsensusGenerativeHead,
    ContentShiftHead,
)
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


# --- ContentShiftHead (arm +D) ---------------------------------------------------

RANK = 3
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _shift_head(hidden: int = 8) -> ContentShiftHead:
    torch.manual_seed(0)
    head = ContentShiftHead(RANK, DIM, SCANNERS, True, hidden, hidden).double()
    head.init_from_moments(*_views())
    return head


def test_shift_head_init_equals_base_init() -> None:
    base, shift = _head(), _shift_head()
    # Same seed, same data: identical map (the shift parameters draw no randoms).
    base_state, shift_state = base.map.state_dict(), shift.map.state_dict()
    assert base_state.keys() == shift_state.keys()
    for name, value in base_state.items():
        torch.testing.assert_close(value, shift_state[name], msg=name)
    torch.testing.assert_close(base.delta, shift.delta)
    torch.testing.assert_close(base.noise_factor(), shift.noise_factor())
    z = torch.randn(7, DIM, dtype=torch.double)
    k = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    assert shift.shift(z, k).abs().max() == 0  # A = 0: D_k = 0
    torch.testing.assert_close(shift.residual_penalty(), base.residual_penalty())
    # P = top-RANK eigenvectors of L, orthonormal.
    P = shift.shift_P
    torch.testing.assert_close(P.T @ P, torch.eye(RANK, dtype=P.dtype))
    evals, evecs = torch.linalg.eigh(shift.map.L)
    top = evecs[:, -RANK:]
    torch.testing.assert_close((top @ top.T) @ P, P)
    assert shift.fit_metadata() == {"per_scanner": True, "shift_rank": RANK}
    assert base.fit_metadata() == {"per_scanner": True}


def test_shift_sums_to_zero_and_nll_matches_dense() -> None:
    head = _shift_head()
    with torch.no_grad():
        head.shift_A.normal_()
        head.noise_factor.offdiag.normal_(0, 0.1)
    factors = head.shift_factors()
    D = factors @ head.shift_P.T  # [S, dim, dim]
    w = head.scanner_weight
    torch.testing.assert_close(torch.einsum("s,sij->ij", w, D),
                               torch.zeros(DIM, DIM, dtype=D.dtype))  # fmt: skip
    u = torch.randn(9, DIM, dtype=torch.double)
    g = torch.randn(9, DIM, dtype=torch.double)
    z = torch.randn(9, DIM, dtype=torch.double)
    k = torch.tensor([0, 1, 2] * 3)
    mean = head.likelihood_mean(g, z, k)
    B = head.noise_factor()
    dense = torch.stack(
        [-MultivariateNormal(g[i] + D[k[i]] @ z[i] + head.delta[k[i]],
                             scale_tril=B[k[i]]).log_prob(u[i])
         for i in range(9)]
    )  # fmt: skip
    torch.testing.assert_close(head.nll(u, mean, k), dense)
    base_pen = ConsensusGenerativeHead.residual_penalty(head)
    torch.testing.assert_close(head.residual_penalty() - base_pen, D.square().sum())


def test_shift_learns_content_dependent_offsets() -> None:
    """Views with u_k = s + delta_k + D_k s: the nll falls once A moves off 0."""
    head = _shift_head(hidden=0)
    g = torch.Generator().manual_seed(3)
    z = torch.randn(300, DIM, generator=g, dtype=torch.double)
    D_true = 0.5 * torch.randn(SCANNERS, DIM, DIM, generator=g, dtype=torch.double)
    D_true -= D_true.mean(0)
    k = torch.arange(SCANNERS).repeat(100)
    with torch.no_grad():
        gz = head.map.generate(z)
        u = gz + head.delta[k] + torch.einsum("bij,bj->bi", D_true[k], z)
        u += 0.1 * torch.randn(u.shape, generator=g, dtype=torch.double)
    # Gradient reaches A at init (P only once A != 0).
    loss = head.nll(u, head.likelihood_mean(gz, z, k), k).mean()
    loss.backward()
    assert head.shift_A.grad.abs().sum() > 0
    start = float(loss.detach())
    opt = torch.optim.Adam([head.shift_A, head.shift_P], lr=0.05)
    for _ in range(200):
        opt.zero_grad()
        loss = head.nll(u, head.likelihood_mean(gz, z, k), k).mean()
        loss.backward()
        opt.step()
    assert float(loss) < start - 1.0


def test_shift_head_rejects_exchangeable_and_bad_rank() -> None:
    with pytest.raises(ValueError):
        ContentShiftHead(RANK, DIM, SCANNERS, per_scanner=False)
    with pytest.raises(ValueError):
        ContentShiftHead(DIM + 1, DIM, SCANNERS)


def test_shift_config_matches_base() -> None:
    """consensus_gen_shift.yaml = consensus_gen.yaml except the network class / rank."""
    base = yaml.safe_load((CONFIGS / "consensus_gen.yaml").read_text())
    shift = yaml.safe_load((CONFIGS / "consensus_gen_shift.yaml").read_text())
    net_b = base["model"]["init_args"].pop("network")
    net_s = shift["model"]["init_args"].pop("network")
    assert shift == base
    assert net_s["class_path"] == "sipe.model.consensus_gen.ContentShiftCGM"
    assert net_s["init_args"].pop("shift_rank") > 0
    assert net_s["init_args"] == net_b["init_args"]


def test_cgm_select_labels_shift_arm(tmp_path) -> None:
    from sipe.eval.cgm_select import ARM_ORDER, fit_info

    for name, arm in (("consensus_gen.yaml", "V1-S"),
                      ("consensus_gen_shift.yaml", "V1-S+D")):  # fmt: skip
        run = tmp_path / name
        run.mkdir()
        (run / "config.yaml").write_text((CONFIGS / name).read_text())
        assert fit_info(run)["arm"] == arm
    assert ARM_ORDER.index("V1-S") < ARM_ORDER.index("V1-S+D") < ARM_ORDER.index("V2")
