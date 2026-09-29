from __future__ import annotations

import torch
import torch.nn as nn


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    """Identity forward pass with a sign-reversed backward gradient."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(
        self,
        x: torch.Tensor,
        *,
        alpha: float | None = None,
    ) -> torch.Tensor:
        value = self.alpha if alpha is None else float(alpha)
        return _GradientReversalFunction.apply(x, value)
