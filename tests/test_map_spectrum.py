import numpy as np
import torch

from sipe.bench.posthoc import AffineMap
from sipe.estimators.mixture import Mixture, Moments, _ridge
from sipe.eval.map_spectrum import fit_alpha, g0_basis, mean_jacobian


def _views(n=400, s=5, d=12, seed=0):
    rng = np.random.default_rng(seed)
    content = rng.normal(size=(n, 1, d)) @ rng.normal(size=(d, d))
    noise = rng.normal(size=(n, s, d)) @ (0.5 * rng.normal(size=(d, d)))
    return content + noise


def test_g0_is_diagonal_in_basis_with_gain_lambda_over_lambda_plus_alpha():
    views, ridge, alpha = _views(), 1e-4, 8.0
    lam, v, _, n, mu = g0_basis(views, ridge)
    moments = Moments.fit(views, np.zeros(len(views), int), 1)
    p = Mixture.build(moments, "G0", ridge=ridge).operators(alpha)[0]
    a = v.T @ p @ (_ridge(n, ridge) @ v)
    np.testing.assert_allclose(np.diag(a), lam / (lam + alpha), atol=1e-8)
    np.testing.assert_allclose(a - np.diag(np.diag(a)), 0, atol=1e-8)
    fitted, rmse = fit_alpha(lam, np.diag(a))
    assert abs(fitted - alpha) / alpha < 0.01 and rmse < 1e-3


def test_mean_jacobian_of_affine_map_is_its_operator():
    rng = np.random.default_rng(1)
    p = rng.normal(size=(6, 6))
    fmap = AffineMap(torch.tensor(p, dtype=torch.float32), torch.zeros(6))
    j, spread = mean_jacobian(fmap, rng.normal(size=(10, 6)), batch=4)
    np.testing.assert_allclose(j, p, atol=1e-5)
    assert spread < 1e-6
