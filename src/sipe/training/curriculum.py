from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# "feature": CATSv2 feature-space objective (not in legacy; see CATSModule).
_VALID_MODES = {"recon", "adverse", "cycle", "feature"}


@dataclass(frozen=True)
class CurriculumPhase:
    """Step-based equivalent of one legacy Curriculum entry."""

    name: str
    mode: str
    steps: int
    lr: float
    adverse_alpha: float | list[float] = 1.0
    adverse_alpha_interval_steps: int = 1
    restart_steps: int | None = None
    adverse_norm: bool = True
    freeze_backbone: bool = True
    freeze_tangler: bool = False
    reset_optimizer_state: bool = True

    def __post_init__(self) -> None:
        mode = self.mode.lower()
        if mode == "adversarial":
            mode = "adverse"
        object.__setattr__(self, "mode", mode)

        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown curriculum mode {self.mode!r}.")
        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if self.restart_steps is not None and self.restart_steps <= 0:
            raise ValueError("restart_steps must be positive.")
        if self.adverse_alpha_interval_steps <= 0:
            raise ValueError("adverse_alpha_interval_steps must be positive.")

        if isinstance(self.adverse_alpha, list):
            if not self.adverse_alpha:
                raise ValueError("adverse_alpha list cannot be empty.")
            if any(a < 0 for a in self.adverse_alpha):
                raise ValueError("adverse_alpha values must be non-negative.")
        elif self.adverse_alpha < 0:
            raise ValueError("adverse_alpha must be non-negative.")

    def alpha_at(self, local_step: int) -> float:
        if not isinstance(self.adverse_alpha, list):
            return float(self.adverse_alpha)
        index = min(
            local_step // self.adverse_alpha_interval_steps,
            len(self.adverse_alpha) - 1,
        )
        return float(self.adverse_alpha[index])


class StepCurriculum:
    def __init__(
        self,
        phases: list[CurriculumPhase | Mapping[str, Any]],
    ) -> None:
        if not phases:
            raise ValueError("At least one curriculum phase is required.")

        parsed = []
        for index, phase in enumerate(phases):
            if isinstance(phase, CurriculumPhase):
                parsed.append(phase)
                continue

            values = dict(phase)
            if "type" in values and "mode" not in values:
                values["mode"] = values.pop("type")
            if "restarts" in values and "restart_steps" not in values:
                values["restart_steps"] = values.pop("restarts")
            if "norm" in values and "adverse_norm" not in values:
                values["adverse_norm"] = values.pop("norm")
            if "freeze_bb" in values and "freeze_backbone" not in values:
                values["freeze_backbone"] = values.pop("freeze_bb")
            values.setdefault("name", f"phase_{index}")
            parsed.append(CurriculumPhase(**values))

        self.phases = tuple(parsed)

        starts = []
        total = 0
        for phase in self.phases:
            starts.append(total)
            total += phase.steps
        self.starts = tuple(starts)
        self.total_steps = total

    def at(self, global_step: int) -> tuple[int, CurriculumPhase, int]:
        for index, (start, phase) in enumerate(zip(self.starts, self.phases)):
            if global_step < start + phase.steps:
                return index, phase, global_step - start
        raise RuntimeError(
            f"global_step={global_step} is past curriculum end ({self.total_steps})."
        )

    def phase_start(self, phase_index: int) -> int:
        return self.starts[phase_index]
