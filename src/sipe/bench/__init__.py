"""Benchmark CATS checkpoints with the official PLISM and HEST code.

`checkpoint`, `encoders` and `provenance` need only the training env (torch, timm,
lightning). `plism` and `hest` import their benchmark package at module level and
run only in the bench env (`bench/run`).
"""
