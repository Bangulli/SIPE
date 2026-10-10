from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sipe.bench.checkpoint import checkpoint_module_class
from sipe.cli import CLI_KWARGS, SIPECLI
from sipe.training.cats_module import CATSModule
from sipe.training.paired_vae_module import PairedVAEModule
from sipe.training.scanner_vae_module import ScannerVAEModule

ROOT = Path(__file__).resolve().parents[1]
# Stale since before the single-CLI change (CATSModule args they use no longer exist).
STALE = {"cats.yaml", "cats_curriculum.yaml"}
CONFIGS = sorted(
    p.name for p in (ROOT / "configs").glob("*.yaml") if p.name not in STALE
)


class ParseOnlyCLI(SIPECLI):
    """Parse (and apply links) without a run dir or instantiating anything."""

    def before_instantiate_classes(self) -> None:
        pass

    def instantiate_classes(self) -> None:
        pass


@pytest.mark.parametrize("config", CONFIGS)
def test_config_parses_with_single_cli(config: str) -> None:
    """`sipe fit` resolves model.class_path and links the encoder to the data."""
    path = ROOT / "configs" / config
    # run=False: no subcommand, the parsed config is the `fit` one at top level.
    parsed = ParseOnlyCLI(**CLI_KWARGS, args=["--config", str(path)], run=False).config
    source = yaml.safe_load(path.read_text())["model"]
    assert parsed.model.class_path == source["class_path"]
    network = source["init_args"]["network"]
    encoder = network.get("init_args", network)["encoder"]
    assert parsed.model.init_args.network.init_args.encoder == encoder
    assert parsed.data.init_args.encoder == encoder


def test_checkpoint_module_class() -> None:
    assert checkpoint_module_class({}) is CATSModule  # predates the key
    assert checkpoint_module_class({"sipe_module": "scanner_vae"}) is ScannerVAEModule
    assert checkpoint_module_class({"sipe_module": "paired_vae"}) is PairedVAEModule
    path = "sipe.training.paired_vae_module.PairedVAEModule"
    assert checkpoint_module_class({"sipe_module": path}) is PairedVAEModule
    with pytest.raises(TypeError):
        checkpoint_module_class({"sipe_module": "torch.nn.Linear"})
