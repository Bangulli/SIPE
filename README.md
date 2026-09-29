# CATS Lightning refactor

This package is based on the current `arch.py`, but moves all training-specific
domain-adversarial machinery out of the architecture.

## Suggested project layout

```text
src/sipe/
├── model/
│   └── arch.py
├── training/
│   ├── __init__.py
│   ├── grl.py
│   └── cats_module.py
└── cli.py

configs/
└── cats.yaml
```

## Responsibility split

`model/arch.py`
- backbone
- disentangler
- re-entangler
- image decoder
- `encode`, `decode`, and inference `forward`

`training/grl.py`
- generic gradient-reversal primitive

`training/cats_module.py`
- specified and unspecified domain classifiers
- GRL application
- reconstruction loss
- domain/adversarial losses
- optional linear GRL warm-up
- Lightning train/validation/test/predict hooks
- metrics/logging

`cli.py`
- LightningCLI entry point

The first version intentionally does **not** include cycle consistency. It is a
clean reconstruction + adversarial baseline. A paired or cycle objective can be
added later without modifying `arch.py`.

## Batch contract

The default batch format is:

```python
{
    "image": image_tensor,   # [B, C, H, W]
    "domain": domain_id,     # [B], integer class ids
}
```

The keys can be changed with `model.image_key` and `model.domain_key`.

## Run

```bash
python -m sipe.cli fit --config configs/cats.yaml
```

or, if you add this to `pyproject.toml`:

```toml
[project.scripts]
sipe = "sipe.cli:main"
```

then:

```bash
sipe fit --config configs/cats.yaml
```

LightningCLI can automatically construct the single optimizer/scheduler from
the top-level `optimizer` / `lr_scheduler` config groups, so
`CATSModule.configure_optimizers()` is intentionally omitted.
