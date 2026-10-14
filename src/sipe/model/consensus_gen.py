"""Consensus generative model (CGM, Experiment 7 in docs/experiment7-draft.md), backbone
in the loop: frozen H0-mini GAP x [768] -> one global map f(x) = g(m(x)).

Generative model (fit time), tile i seen by scanners k, in standardized coordinates
u = (x - center) / scale:

    z_i ~ N(0, I),   u_{i,k} = g(z_i) + delta_k + B_k eps,   g(z) = b + L z + h(z)

The acquisition latent is integrated out analytically (Gaussian with full covariance
B_k B_k^T, one Cholesky factor per scanner, or one shared factor and delta = 0 in the
exchangeable arm). The scanner label is used only in this likelihood, at fit time.

Encoder (single view): q(z|x) = N(m(u), A A^T), m(u) = M u + r(u).
Inference: f(x) = center + scale * g(m(u)). No scanner ID, no gate, no paired view.

`init_from_moments` sets L, M, A, delta_k, B_k to the linear-Gaussian solution from
the train moments (C = cross-view covariance, N0 = within-tile covariance); r and h
end in zero-initialized layers, so at init f(x) = center + scale L M u, the
linear-Gaussian posterior-mean predictor of the noise-free signal (close to G0, which
predicts the 5-scanner mean instead).

Arms (draft, Step 3): `per_scanner` (V*-S vs exchangeable), `encoder_hidden_dim` (0:
linear encoder, V0; >0: V1), `generator_hidden_dim` (>0: nonlinear h, V2).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sipe.model.encoders import build_encoder


def _mlp(dim: int, hidden_dim: int) -> nn.Sequential:
    """Residual branch dim -> hidden -> hidden -> dim; last layer zero-initialized."""
    mlp = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim),
    )
    nn.init.zeros_(mlp[-1].weight)
    nn.init.zeros_(mlp[-1].bias)
    return mlp


class CholeskyFactor(nn.Module):
    """Lower-triangular factors, positive diagonal: tril(W, -1) + diag(exp(d))."""

    def __init__(self, dim: int, count: int = 1) -> None:
        super().__init__()
        self.offdiag = nn.Parameter(torch.zeros(count, dim, dim))
        self.log_diag = nn.Parameter(torch.zeros(count, dim))

    def forward(self) -> torch.Tensor:
        """[count, dim, dim]."""
        return torch.tril(self.offdiag, -1) + torch.diag_embed(self.log_diag.exp())

    @torch.no_grad()
    def set_(self, factors: torch.Tensor) -> None:
        """Set from lower-triangular factors [count, dim, dim] (positive diagonal)."""
        factors = factors.to(self.offdiag).expand_as(self.offdiag)
        self.offdiag.copy_(torch.tril(factors, -1))
        self.log_diag.copy_(torch.diagonal(factors, dim1=-2, dim2=-1).log())


class CGMMap(nn.Module):
    """The evaluated map f: [B, dim] backbone GAP -> [B, dim] consensus estimate.

    Saved alone as a `--posthoc` module map file (`sipe.bench.posthoc`).
    """

    def __init__(
        self,
        dim: int = 768,
        encoder_hidden_dim: int = 1024,
        generator_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.register_buffer("center", torch.zeros(dim))
        self.register_buffer("scale", torch.ones(()))
        self.M = nn.Parameter(torch.eye(dim))
        self.L = nn.Parameter(torch.eye(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.r = _mlp(dim, encoder_hidden_dim) if encoder_hidden_dim > 0 else None
        self.h = _mlp(dim, generator_hidden_dim) if generator_hidden_dim > 0 else None

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.center) / self.scale

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """Posterior mean m(u) [B, dim]."""
        m = u @ self.M.T
        return m if self.r is None else m + self.r(u)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """g(z) in standardized coordinates."""
        g = self.b + z @ self.L.T
        return g if self.h is None else g + self.h(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.center + self.scale * self.generate(
            self.encode(self.standardize(x))
        )


def _psd_clip(matrix: torch.Tensor, floor: float) -> torch.Tensor:
    """Symmetrize and raise eigenvalues to at least `floor` (float64)."""
    evals, evecs = torch.linalg.eigh(0.5 * (matrix + matrix.T))
    return (evecs * evals.clamp_min(floor)) @ evecs.T


def _sqrtm(matrix: torch.Tensor) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(matrix)
    return (evecs * evals.clamp_min(0).sqrt()) @ evecs.T


class ConsensusGenerativeHead(nn.Module):
    """CGMMap plus the fit-time-only parts: posterior factor A, offsets delta_k and
    acquisition-noise factors B_k (or B_0)."""

    def __init__(
        self,
        dim: int = 768,
        num_scanners: int = 5,
        per_scanner: bool = True,
        encoder_hidden_dim: int = 1024,
        generator_hidden_dim: int = 0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_scanners = int(num_scanners)
        self.per_scanner = bool(per_scanner)
        self.map = CGMMap(dim, encoder_hidden_dim, generator_hidden_dim)
        self.posterior_factor = CholeskyFactor(dim)
        self.delta = nn.Parameter(
            torch.zeros(num_scanners, dim), requires_grad=self.per_scanner
        )
        self.noise_factor = CholeskyFactor(dim, num_scanners if per_scanner else 1)
        # Persisted, so a resumed fit does not re-run the moment initialization.
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def init_from_moments(
        self,
        x: torch.Tensor,
        scanners: torch.Tensor,
        locations: torch.Tensor,
        ridge: float = 1e-3,
    ) -> dict[str, float]:
        """Linear-Gaussian init from train views x [n, dim] (float64 inside).

        `locations` groups the views of a tile (any integer ids). Returns a few
        diagnostics (traces) for logging.
        """
        x = x.double()
        dim = self.dim
        center = x.mean(0)
        scale = (x - center).square().sum(1).mean().div(dim).sqrt()
        u = (x - center) / scale
        delta = torch.zeros(self.num_scanners, dim, dtype=u.dtype, device=u.device)
        if self.per_scanner:
            for k in range(self.num_scanners):
                if bool((scanners == k).any()):
                    delta[k] = u[scanners == k].mean(0)
        # Offsets as a random effect for the scanner-blind encoder.
        weights = torch.bincount(scanners, minlength=self.num_scanners).to(u.dtype)
        weights = weights / weights.sum()
        offset_cov = (delta.T * weights) @ delta
        res = u - delta[scanners]
        res = res - res.mean(0)

        _, loc = torch.unique(locations, return_inverse=True)
        sums = torch.zeros(int(loc.max()) + 1, dim, dtype=u.dtype, device=u.device)
        sums.index_add_(0, loc, res)
        counts = torch.bincount(loc).to(u.dtype)
        pairs = float((counts * (counts - 1)).sum())
        if pairs == 0:
            raise ValueError("init_from_moments needs >= 2 views of some tiles.")
        gram = res.T @ res
        signal = (sums.T @ sums - gram) / pairs  # cross-view covariance C
        total = gram / res.shape[0]
        # Within-tile covariance, PSD by construction: sum over tiles of
        # (V X^T X - T T^T) / (V (V - 1)) per pair, here with per-tile V.
        within = torch.zeros_like(gram)
        for v in counts.unique().tolist():
            if v < 2:
                continue
            tiles = (counts == v).nonzero().flatten()
            mask = torch.isin(loc, tiles)
            rv, tv = res[mask], sums[tiles]
            within += (v * rv.T @ rv - tv.T @ tv) / (v - 1)
        within /= float(counts[counts >= 2].sum())

        floor = ridge * float(torch.trace(total)) / dim
        signal = _psd_clip(signal, floor)
        noise = _psd_clip(within, floor)
        L = _sqrtm(signal)
        marginal = signal + noise + offset_cov
        M = torch.linalg.solve(marginal, L).T  # L^T marginal^-1 (L, marginal sym.)
        post = _psd_clip(torch.eye(dim, dtype=u.dtype, device=u.device) - M @ L, 1e-4)

        fmap = self.map
        fmap.center.copy_(center)
        fmap.scale.copy_(scale)
        fmap.M.copy_(M)
        fmap.L.copy_(L)
        fmap.b.zero_()
        self.posterior_factor.set_(torch.linalg.cholesky(post)[None])
        self.delta.copy_(delta)
        self.noise_factor.set_(torch.linalg.cholesky(noise)[None])
        self.initialized.fill_(True)
        return {
            "init/scale": float(scale),
            "init/tr_signal": float(torch.trace(signal)),
            "init/tr_noise": float(torch.trace(noise)),
            "init/tr_offsets": float(torch.trace(offset_cov)),
            "init/num_views": float(x.shape[0]),
        }

    def sample(self, m: torch.Tensor) -> torch.Tensor:
        """z ~ q(z|x) = N(m, A A^T)."""
        return m + torch.randn_like(m) @ self.posterior_factor()[0].T

    def kl(self, m: torch.Tensor) -> torch.Tensor:
        """KL(N(m, A A^T) || N(0, I)) per row [B]."""
        A = self.posterior_factor()[0]
        logdet = 2 * self.posterior_factor.log_diag[0].sum()
        return 0.5 * (A.square().sum() + m.square().sum(1) - self.dim - logdet)

    def nll(
        self, u: torch.Tensor, g: torch.Tensor, scanners: torch.Tensor
    ) -> torch.Tensor:
        """-log N(u; g + delta_k, B_k B_k^T) per row [B]."""
        factors = self.noise_factor()
        if self.per_scanner:
            resid = u - g - self.delta[scanners]
            groups = [(scanners == k, k) for k in range(self.num_scanners)]
        else:
            resid = u - g
            groups = [(torch.ones_like(scanners, dtype=torch.bool), 0)]
        out = resid.new_zeros(resid.shape[0])
        const = 0.5 * self.dim * math.log(2 * math.pi)
        for mask, k in groups:
            if not bool(mask.any()):
                continue
            y = torch.linalg.solve_triangular(
                factors[k], resid[mask].T, upper=False
            )  # [dim, n_k]
            logdet = self.noise_factor.log_diag[k].sum()
            out[mask] = 0.5 * y.square().sum(0) + logdet + const
        return out

    def residual_penalty(self) -> torch.Tensor:
        """Squared L2 norm of the r and h output layers (the rho term)."""
        total = self.map.b.new_zeros(())
        for mlp in (self.map.r, self.map.h):
            if mlp is not None:
                total = (
                    total + mlp[-1].weight.square().sum() + mlp[-1].bias.square().sum()
                )
        return total


class ConsensusGenModel(nn.Module):
    """Frozen backbone -> GAP over patch tokens (= backbone_gap, as the feature cache
    and `sipe.bench.posthoc`) -> ConsensusGenerativeHead."""

    needs_domains = False

    def __init__(
        self,
        encoder: str = "h0-mini",
        num_scanners: int = 5,
        per_scanner: bool = True,
        encoder_hidden_dim: int = 1024,
        generator_hidden_dim: int = 0,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        bundle = build_encoder(encoder, pretrained=pretrained)
        self.encoder_name = encoder
        self.backbone = bundle.backbone
        self.encoder_meta = bundle.meta
        self.eval_transform = bundle.transform
        self.feature_dim = self.encoder_meta.embed_dim
        self.head = ConsensusGenerativeHead(
            dim=self.feature_dim,
            num_scanners=num_scanners,
            per_scanner=per_scanner,
            encoder_hidden_dim=encoder_hidden_dim,
            generator_hidden_dim=generator_hidden_dim,
        )
        self.map_init_args = {
            "dim": self.feature_dim,
            "encoder_hidden_dim": int(encoder_hidden_dim),
            "generator_hidden_dim": int(generator_hidden_dim),
        }

    def freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)

    def backbone_gap(self, images: torch.Tensor) -> torch.Tensor:
        """As PosthocFeatures.gap: patch tokens only, mean, float32."""
        tokens = self.backbone.forward_features(images)
        return tokens[:, self.backbone.num_prefix_tokens :].mean(dim=1).float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.backbone_gap(images)
        with torch.autocast(x.device.type, enabled=False):
            return self.head.map(x)
