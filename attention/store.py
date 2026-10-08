"""Keep only one sampled attention row per selected layer and step."""

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class AttentionRecord:
    layer: str
    step: int
    timestep: float | None
    query_shape: Tuple[int, ...]
    key_shape: Tuple[int, ...]
    cross_shape: Tuple[int, ...]
    self_shape: Tuple[int, ...]
    prefix_mass: float
    target_mass: float
    sampled_row: torch.Tensor  # mean over batch/heads; CPU float16, one query only


class AttentionStore:
    def __init__(self):
        self.records: List[AttentionRecord] = []

    def add(self, record: AttentionRecord) -> None:
        self.records.append(record)
        print(
            f"[attention] layer={record.layer} step={record.step} "
            f"timestep={record.timestep} type=target→prefix/cross "
            f"shape={record.cross_shape} type=target→target/self "
            f"shape={record.self_shape} Q={record.query_shape} K={record.key_shape} "
            f"sampled_mass(prefix={record.prefix_mass:.4f}, target={record.target_mass:.4f})",
            flush=True,
        )
