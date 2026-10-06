# Data sanity tests

Two tests guard result validity. Run them with `uv run pytest tests/test_data_sanity.py`.

- **Dataset leakage:** no WSI (`slide_id`) appears in more than one of train/val/test.
- **Scanner labels:** each sample's `domain` label matches the scanner in its filename. The
  per-patch target expansion used for the `z` classifier stays aligned with the sample.
