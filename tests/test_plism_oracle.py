import numpy as np

from sipe.bench.plism_oracle import N_COORDS, condition_mean_oracle


def test_condition_mean_oracle_moves_mean_keeps_differences():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(50, N_COORDS + 8)).astype(np.float32)
    feats[:, N_COORDS:] += 5.0
    m = rng.normal(size=8)
    out = condition_mean_oracle(feats, m)
    assert out.dtype == np.float32 and out.shape == feats.shape
    np.testing.assert_array_equal(out[:, :N_COORDS], feats[:, :N_COORDS])
    np.testing.assert_allclose(out[:, N_COORDS:].mean(axis=0), m, atol=1e-5)
    np.testing.assert_allclose(
        out[1:, N_COORDS:] - out[:-1, N_COORDS:],
        feats[1:, N_COORDS:] - feats[:-1, N_COORDS:],
        atol=1e-5,
    )
