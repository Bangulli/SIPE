# CATS legacy-faithful step curriculum

This refactor keeps the old training behavior as closely as possible while
making the schedule optimizer-step based.

## Preserved from the old trainer

- recon / adverse / cycle phases;
- fresh AdamW state at each phase boundary;
- `CosineAnnealingWarmRestarts`;
- `adverse_alpha` and `adverse_norm`;
- backbone and entangler freezing;
- the original `ImageReconLoss`;
- the original `AdversarialClassifLoss`;
- the original cycle graph;
- `20 * reconstruction + 1 * S-cycle + 0.5 * Z-cycle`.

The only intentional task-level difference is that SCORPION has one
scanner/domain branch instead of the old stain + organ + pathology branches.

## Step conversion

The config uses:

- 5 epochs -> 5k optimizer steps
- 100 epochs -> 100k optimizer steps
- 20 epochs -> 20k optimizer steps

and preserves the old restart ratios:

- 5 -> 5k
- 25 -> 25k
- 20 -> 20k

Those absolute step counts are a compute-budget choice, not a claim that one
old epoch equals 1000 optimizer updates.

The old alpha list changed once per epoch. Here it changes once per
`adverse_alpha_interval_steps`; after the list is exhausted the last value is
held.

## Exact cycle graph

```text
x
 -> encode -> s1, z1
 -> roll s1
 -> decode(roll(s1), z1)
 -> detach generated image only
 -> encode -> s2, z2
 -> unroll s2
 -> L1(s2_unrolled, s1)
 -> L1(z2, z1)
```

`s1` and `z1` are not detached.

## Run

```bash
uv run sipe fit --config configs/cats_legacy_steps.yaml
```

For a smoke test:

```bash
uv run sipe fit \
  --config configs/cats_legacy_steps.yaml \
  --trainer.logger=false \
  --trainer.callbacks=[] \
  --trainer.enable_checkpointing=false \
  --trainer.max_steps=100
```
