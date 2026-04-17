from dataclasses import dataclass, field
from datetime import datetime

from vision_service.contracts import VisionRule
from vision_service.runtime.dwell import EvidenceSample
from vision_service.vision.confidence import (
    ConfidenceAssessment,
    roi_signal_confidence,
    score_semantic_fallback,
)
from vision_service.vision.roi.models import ROIOccupancyObservation
from vision_service.vision.semantic import (
    SemanticCheckResult,
    SemanticChecker,
    SemanticVerdict,
)


POSITIVE_SEMANTIC_VERDICTS = {"有", "疑似有"}


@dataclass(slots=True, frozen=True)
class SemanticVoteSummary:
    attempts: int
    positive_votes: int
    verdicts: tuple[SemanticVerdict, ...]


@dataclass(slots=True, frozen=True)
class SemanticFallbackTransition:
    observed_at: datetime
    dwell_seconds: int
    semantic_result: SemanticCheckResult
    vote_summary: SemanticVoteSummary
    confidence: ConfidenceAssessment
    consecutive_yolo_failures: int
    yolo_support_confidence: float | None = None
    evidence_samples: tuple[EvidenceSample, ...] = ()


@dataclass(slots=True)
class _SemanticEpisode:
    entered_at: datetime
    last_seen_at: datetime
    strongest_roi_observation: ROIOccupancyObservation | None = None
    last_sampled_at: datetime | None = None
    last_checked_at: datetime | None = None
    semantic_result: SemanticCheckResult | None = None
    semantic_results: list[SemanticCheckResult] = field(default_factory=list)
    yolo_threshold_observed: bool = False
    consecutive_yolo_failures: int = 0
    best_yolo_confidence: float | None = None
    attempts: int = 0
    positive_votes: int = 0
    samples: list[EvidenceSample] = field(default_factory=list)


class SemanticFallbackTracker:
    def __init__(
        self,
        *,
        rule: VisionRule,
        threshold_seconds: int,
        sample_interval_seconds: float,
        max_samples: int,
        consecutive_yolo_failures: int,
        retry_cooldown_seconds: float,
        max_attempts_per_episode: int,
        checker: SemanticChecker,
    ) -> None:
        self._rule = rule.model_copy(deep=True)
        self._threshold_seconds = threshold_seconds
        self._sample_interval_seconds = sample_interval_seconds
        self._max_samples = max_samples
        self._consecutive_yolo_failures = consecutive_yolo_failures
        self._retry_cooldown_seconds = retry_cooldown_seconds
        self._max_attempts_per_episode = max_attempts_per_episode
        self._required_positive_votes = (
            1 if max_attempts_per_episode == 1 else 2
        )
        self._checker = checker
        self._episode: _SemanticEpisode | None = None

    @property
    def active(self) -> bool:
        return self._episode is not None

    async def observe(
        self,
        *,
        observed_at: datetime,
        roi_observation: ROIOccupancyObservation | None,
        evidence_image_bytes: bytes | None,
        semantic_image_bytes: bytes | None,
        yolo_confidence: float | None,
        yolo_threshold_observed: bool,
    ) -> SemanticFallbackTransition | None:
        if roi_observation is None or not roi_observation.presence_active:
            return self.force_clear(
                observed_at=observed_at,
                yolo_threshold_observed=yolo_threshold_observed,
            )

        episode = self._episode
        if episode is None:
            episode = _SemanticEpisode(
                entered_at=observed_at,
                last_seen_at=observed_at,
                strongest_roi_observation=roi_observation,
            )
            self._episode = episode

        episode.last_seen_at = observed_at
        episode.yolo_threshold_observed = (
            episode.yolo_threshold_observed or yolo_threshold_observed
        )
        if _is_stronger_roi_observation(
            candidate=roi_observation,
            current=episode.strongest_roi_observation,
        ):
            episode.strongest_roi_observation = roi_observation

        self._maybe_store_sample(
            episode=episode,
            observed_at=observed_at,
            image_bytes=evidence_image_bytes,
        )

        if yolo_confidence is not None:
            episode.best_yolo_confidence = max(
                yolo_confidence,
                episode.best_yolo_confidence or yolo_confidence,
            )
            episode.consecutive_yolo_failures = 0
        else:
            episode.consecutive_yolo_failures += 1

        if episode.yolo_threshold_observed:
            return None
        if episode.semantic_result is not None:
            return None
        if semantic_image_bytes is None:
            return None
        if episode.consecutive_yolo_failures < self._consecutive_yolo_failures:
            return None
        if episode.attempts >= self._max_attempts_per_episode:
            return None
        if (
            episode.last_checked_at is not None
            and (
                observed_at - episode.last_checked_at
            ).total_seconds()
            < self._effective_retry_cooldown_seconds()
        ):
            return None

        episode.last_checked_at = observed_at
        episode.attempts += 1
        result = await self._checker.check(
            image_bytes=semantic_image_bytes,
            rule=self._rule,
        )
        episode.semantic_results.append(result)
        if result.verdict in POSITIVE_SEMANTIC_VERDICTS:
            episode.positive_votes += 1
        if episode.positive_votes >= self._required_positive_votes:
            episode.semantic_result = _select_semantic_result(
                episode.semantic_results
            )
        return None

    def force_clear(
        self,
        *,
        observed_at: datetime,
        yolo_threshold_observed: bool = False,
    ) -> SemanticFallbackTransition | None:
        episode = self._episode
        self._episode = None
        if episode is None:
            return None
        if episode.yolo_threshold_observed or yolo_threshold_observed:
            return None
        if episode.semantic_result is None:
            return None

        dwell_seconds = max(
            0,
            int((episode.last_seen_at - episode.entered_at).total_seconds()),
        )
        if dwell_seconds < self._threshold_seconds:
            return None

        confidence = score_semantic_fallback(
            verdict=episode.semantic_result.verdict,
            roi_observation=episode.strongest_roi_observation,
            yolo_support_confidence=episode.best_yolo_confidence,
            consecutive_yolo_failures=episode.consecutive_yolo_failures,
        )
        return SemanticFallbackTransition(
            observed_at=observed_at,
            dwell_seconds=dwell_seconds,
            semantic_result=episode.semantic_result,
            vote_summary=SemanticVoteSummary(
                attempts=episode.attempts,
                positive_votes=episode.positive_votes,
                verdicts=tuple(
                    result.verdict for result in episode.semantic_results
                ),
            ),
            confidence=confidence,
            consecutive_yolo_failures=episode.consecutive_yolo_failures,
            yolo_support_confidence=episode.best_yolo_confidence,
            evidence_samples=self._select_evidence_samples(episode),
        )

    def _maybe_store_sample(
        self,
        *,
        episode: _SemanticEpisode,
        observed_at: datetime,
        image_bytes: bytes | None,
    ) -> None:
        if image_bytes is None:
            return
        if (
            episode.last_sampled_at is not None
            and (
                observed_at - episode.last_sampled_at
            ).total_seconds()
            < self._sample_interval_seconds
        ):
            return

        episode.samples.append(
            EvidenceSample(
                captured_at=observed_at,
                image_bytes=image_bytes,
            )
        )
        if len(episode.samples) > self._max_samples:
            episode.samples = self._rebalance_samples(
                samples=episode.samples,
                max_samples=self._max_samples,
            )
        episode.last_sampled_at = observed_at

    def _effective_retry_cooldown_seconds(self) -> float:
        threshold_spread_seconds = self._threshold_seconds / max(
            float(self._max_attempts_per_episode),
            1.0,
        )
        return max(self._retry_cooldown_seconds, threshold_spread_seconds)

    @staticmethod
    def _rebalance_samples(
        *,
        samples: list[EvidenceSample],
        max_samples: int,
    ) -> list[EvidenceSample]:
        if len(samples) <= max_samples:
            return samples
        if max_samples <= 1:
            return [samples[-1]]

        last_index = len(samples) - 1
        step = last_index / (max_samples - 1)
        chosen_indices: list[int] = []
        previous_index = -1
        for slot in range(max_samples):
            raw_index = round(slot * step)
            min_index = previous_index + 1
            max_index = last_index - (max_samples - slot - 1)
            index = min(max(raw_index, min_index), max_index)
            chosen_indices.append(index)
            previous_index = index
        return [samples[index] for index in chosen_indices]

    @staticmethod
    def _select_evidence_samples(
        episode: _SemanticEpisode,
    ) -> tuple[EvidenceSample, ...]:
        if not episode.samples:
            return ()

        start = episode.samples[0]
        middle = episode.samples[len(episode.samples) // 2]
        end = episode.samples[-1]
        return (start, middle, end)


def _is_stronger_roi_observation(
    *,
    candidate: ROIOccupancyObservation,
    current: ROIOccupancyObservation | None,
) -> bool:
    if current is None:
        return True
    return roi_signal_confidence(candidate) >= roi_signal_confidence(current)


def _select_semantic_result(
    results: list[SemanticCheckResult],
) -> SemanticCheckResult:
    positive_results = [
        result for result in results if result.verdict in POSITIVE_SEMANTIC_VERDICTS
    ]
    assert positive_results
    return max(
        positive_results,
        key=lambda result: (
            1 if result.verdict == "有" else 0,
            result.checked_at,
        ),
    )


def build_semantic_event_metadata(
    transition: SemanticFallbackTransition,
) -> dict[str, object]:
    return {
        "decision": {
            "source": transition.confidence.source,
            "confidence_score": transition.confidence.score,
            "confidence_breakdown": transition.confidence.breakdown,
            "semantic_check": {
                "verdict": transition.semantic_result.verdict,
                "raw_output": transition.semantic_result.raw_output,
                "model": transition.semantic_result.model_name,
                "checked_at": transition.semantic_result.checked_at.isoformat(),
                "attempts": transition.vote_summary.attempts,
                "positive_votes": transition.vote_summary.positive_votes,
                "verdicts": list(transition.vote_summary.verdicts),
                "consecutive_yolo_failures": transition.consecutive_yolo_failures,
                "yolo_support_confidence": transition.yolo_support_confidence,
            },
        }
    }


def semantic_evidence_metadata(
    rule: VisionRule,
    transition: SemanticFallbackTransition,
) -> dict[str, object]:
    return {
        "annotations": {
            "image_kind": "raw",
            "coordinate_space": "normalized_xywh",
            "source": "camera.frame",
            "detections": [],
            "semantic_roi": {
                "source": "configured_zone",
                "box": rule.zone.model_dump(mode="json"),
            },
        },
        "semantic_check": {
            "verdict": transition.semantic_result.verdict,
            "confidence_score": transition.confidence.score,
            "input_source": "roi.zone_crop",
            "input_region": "configured_zone",
        },
    }
