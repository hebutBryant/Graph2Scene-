"""Execution-order gate and optional attention-logit intervention."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass(frozen=True)
class AttentionContext:
    layer: str
    step: int
    timestep: Optional[float]
    attention_type: str
    query_start: int
    query_end: int
    key_length: int
    prefix_length: int


# Called on a small chunk of target-image logits [batch, heads, queries, keys].
# Return logits with the same shape. Invalid/padded keys are masked afterwards.
LogitModifier = Callable[[torch.Tensor, AttentionContext], torch.Tensor]


class AttentionController:
    def __init__(
        self,
        capture_layer: str = "transformer_blocks.0.attn",
        capture_step: int = 0,
        modifier: Optional[LogitModifier] = None,
        modify_layer: Optional[str] = None,
        modify_start_step: int = 0,
        modify_end_step: Optional[int] = None,
    ):
        self.capture_layer = capture_layer
        self.capture_step = capture_step
        self.modifier = modifier
        self.modify_layer = modify_layer or capture_layer
        self.modify_start_step = modify_start_step
        self.modify_end_step = modify_end_step
        self.step = 0
        self.timestep_provider = None

    def set_step(self, step: int) -> None:
        self.step = step

    def timestep(self) -> Optional[float]:
        if self.timestep_provider is None:
            return None
        value = self.timestep_provider()
        return None if value is None else float(value.item() if hasattr(value, "item") else value)

    def should_capture(self, layer: str) -> bool:
        return layer == self.capture_layer and self.step == self.capture_step

    def should_modify(self, layer: str) -> bool:
        return (
            self.modifier is not None
            and layer == self.modify_layer
            and self.modify_start_step <= self.step
            and (self.modify_end_step is None or self.step <= self.modify_end_step)
        )
