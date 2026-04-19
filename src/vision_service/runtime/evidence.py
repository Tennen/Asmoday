from collections.abc import Sequence
from typing import TypeVar

from vision_service.contracts.callbacks import EvidencePhase

EVIDENCE_SAMPLES_PER_THRESHOLD = 3

T = TypeVar("T")


def evidence_capture_phase(*, index: int, total: int) -> EvidencePhase:
    if total <= 0:
        raise ValueError("total evidence samples must be positive")
    if index < 0 or index >= total:
        raise ValueError("evidence sample index is out of range")
    if index == 0:
        return "start"
    if index == total - 1:
        return "end"
    return f"sample_{index + 1:03d}"


def phased_evidence_items(items: Sequence[T]) -> list[tuple[EvidencePhase, T]]:
    total = len(items)
    return [
        (evidence_capture_phase(index=index, total=total), item)
        for index, item in enumerate(items)
    ]
