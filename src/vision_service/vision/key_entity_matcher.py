from base64 import b64encode
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from typing import Any, Protocol, Sequence
from urllib import error, request

from vision_service.contracts.callbacks import EvidencePhase
from vision_service.contracts.control import KeyEntityId, KeyEntityReference
from vision_service.runtime.dwell import EvidenceSample
from vision_service.settings import Settings
from vision_service.vision.key_entity_response import (
    KeyEntityMatchError,
    extract_message_text,
    parse_match_output,
)
from vision_service.vision.model_queue import (
    LocalModelRequestQueue,
    get_local_model_request_queue,
)


@dataclass(slots=True, frozen=True)
class KeyEntityFrameMatch:
    key_entity_id: KeyEntityId | None
    confidence: float | None
    reason: str | None
    raw_output: str
    model_name: str
    checked_at: datetime


@dataclass(slots=True, frozen=True)
class KeyEntityIdentification:
    key_entity_id: KeyEntityId | None
    metadata: dict[str, object]
    error_message: str | None = None


@dataclass(slots=True, frozen=True)
class KeyEntityCandidateFrameMatch:
    phase: EvidencePhase
    candidate_id: KeyEntityId
    match: KeyEntityFrameMatch


class KeyEntityMatcher(Protocol):
    async def match(
        self,
        *,
        image_bytes: bytes,
        key_entities: Sequence[KeyEntityReference],
    ) -> KeyEntityFrameMatch: ...


class OpenAICompatibleKeyEntityMatcher:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str | None,
        timeout_seconds: float,
        request_queue: LocalModelRequestQueue | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._request_queue = request_queue or get_local_model_request_queue()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        request_queue: LocalModelRequestQueue | None = None,
    ) -> "OpenAICompatibleKeyEntityMatcher | None":
        if not settings.semantic_checker_enabled:
            return None
        assert settings.semantic_checker_base_url is not None
        assert settings.semantic_checker_model_name is not None
        return cls(
            base_url=settings.semantic_checker_base_url,
            model_name=settings.semantic_checker_model_name,
            api_key=settings.semantic_checker_api_key,
            timeout_seconds=settings.semantic_checker_timeout_seconds,
            request_queue=request_queue,
        )

    async def match(
        self,
        *,
        image_bytes: bytes,
        key_entities: Sequence[KeyEntityReference],
    ) -> KeyEntityFrameMatch:
        payload = self._build_payload(
            image_bytes=image_bytes,
            key_entities=key_entities,
        )
        response_json = await self._request_queue.run(
            operation="key_entity_match",
            call=lambda: self._request_json(payload),
        )
        raw_output = extract_message_text(response_json)
        matched_id, confidence, reason = parse_match_output(
            text=raw_output,
            key_entities=key_entities,
        )
        return KeyEntityFrameMatch(
            key_entity_id=matched_id,
            confidence=confidence,
            reason=reason,
            raw_output=raw_output,
            model_name=self._model_name,
            checked_at=datetime.now(tz=UTC),
        )

    def _build_payload(
        self,
        *,
        image_bytes: bytes,
        key_entities: Sequence[KeyEntityReference],
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "你是监控事件关键实体匹配器。第一张图是由 YOLO 检测框裁切出的"
                    "待识别单个实体。"
                    "后续会给出候选关键实体，每个候选至少有参考图或文字描述。"
                    "候选可能只有一个；不要因为只有一个候选就强行选择。"
                    "只有当待识别实体与候选视觉特征或描述明确一致时，才返回该候选 id；"
                    "无法确认同一实体时返回 null。"
                    "只输出 JSON："
                    '{"key_entity_id": <候选id或null>, "confidence": <0到1的小数>, '
                    '"reason": "<不超过20字>"}。'
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,"
                    + b64encode(image_bytes).decode("ascii")
                },
            },
        ]
        for key_entity in key_entities:
            description = key_entity.description or "无文字描述"
            has_image = "有参考图" if key_entity.image is not None else "无参考图"
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"候选 id={key_entity.id}。"
                        f"描述：{description}。"
                        f"{has_image}。"
                    ),
                }
            )
            if key_entity.image is not None:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{key_entity.image.content_type};base64,"
                                f"{key_entity.image.base64}"
                            )
                        },
                    }
                )

        return {
            "model": self._model_name,
            "temperature": 0,
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }

    def _request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = (
            self._base_url
            if self._base_url.endswith("/chat/completions")
            else f"{self._base_url}/chat/completions"
        )
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise KeyEntityMatchError(
                f"key entity matcher request failed status={exc.code} body={details}"
            ) from exc
        except error.URLError as exc:
            raise KeyEntityMatchError(
                f"key entity matcher request failed reason={exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise KeyEntityMatchError(
                "key entity matcher request timed out "
                f"timeout_seconds={self._timeout_seconds}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise KeyEntityMatchError("key entity matcher returned invalid JSON") from exc


async def identify_key_entity(
    *,
    evidence_samples: Sequence[EvidenceSample],
    key_entities: Sequence[KeyEntityReference],
    matcher: KeyEntityMatcher | None,
) -> KeyEntityIdentification | None:
    if not key_entities:
        return None
    if matcher is None:
        message = "key entity matching is configured but semantic checker is not enabled"
        return KeyEntityIdentification(
            key_entity_id=None,
            metadata={
                "status": "unavailable",
                "reason": message,
            },
            error_message=message,
        )

    phases: tuple[EvidencePhase, ...] = ("start", "middle", "end")
    phased_samples = list(zip(phases, evidence_samples))
    usable_samples = [
        (phase, sample)
        for phase, sample in phased_samples
        if sample.crop_bytes is not None
    ]
    if not usable_samples:
        return KeyEntityIdentification(
            key_entity_id=None,
            metadata={
                "status": "skipped",
                "reason": "no target crops were captured in evidence samples",
            },
        )

    try:
        candidate_matches: list[KeyEntityCandidateFrameMatch] = []
        for phase, sample in usable_samples:
            for key_entity in key_entities:
                candidate_matches.append(
                    KeyEntityCandidateFrameMatch(
                        phase=phase,
                        candidate_id=key_entity.id,
                        match=await matcher.match(
                            image_bytes=sample.crop_bytes,  # type: ignore[arg-type]
                            key_entities=[key_entity],
                        ),
                    )
                )
        winner_id, votes, ambiguous = _aggregate_candidate_matches(
            key_entities=key_entities,
            candidate_matches=candidate_matches,
        )
    except KeyEntityMatchError as exc:
        return KeyEntityIdentification(
            key_entity_id=None,
            metadata={
                "status": "failed",
                "reason": str(exc),
            },
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        message = f"key entity matcher failed unexpectedly: {exc}"
        return KeyEntityIdentification(
            key_entity_id=None,
            metadata={
                "status": "failed",
                "reason": message,
            },
            error_message=message,
        )

    status = (
        "matched"
        if winner_id is not None
        else "ambiguous"
        if ambiguous
        else "no_match"
    )
    metadata = {
        "status": status,
        "winner_id": winner_id,
        "model": candidate_matches[0].match.model_name,
        "strategy": "pairwise_candidate_vote",
        "input_source": "yolo_detection_crop",
        "input_region": "yolo_detection_box",
        "frames": _build_candidate_frame_metadata(
            phased_samples=phased_samples,
            candidate_matches=candidate_matches,
        ),
        "votes": votes,
    }
    return KeyEntityIdentification(
        key_entity_id=winner_id,
        metadata=metadata,
    )


def _aggregate_candidate_matches(
    *,
    key_entities: Sequence[KeyEntityReference],
    candidate_matches: Sequence[KeyEntityCandidateFrameMatch],
) -> tuple[KeyEntityId | None, list[dict[str, object]], bool]:
    candidate_by_key = {str(entity.id): entity.id for entity in key_entities}
    vote_counts = {key: 0 for key in candidate_by_key}
    confidence_totals = {key: 0.0 for key in candidate_by_key}

    for candidate_match in candidate_matches:
        matched_id = candidate_match.match.key_entity_id
        if matched_id is None:
            continue
        candidate_key = str(candidate_match.candidate_id)
        matched_key = str(matched_id)
        if matched_key != candidate_key:
            raise KeyEntityMatchError(
                "key entity matcher returned an id outside the paired candidate: "
                f"candidate={candidate_match.candidate_id!r} matched={matched_id!r}"
            )
        vote_counts[candidate_key] += 1
        confidence_totals[candidate_key] += candidate_match.match.confidence or 0.0

    vote_rows = [
        {
            "key_entity_id": candidate_by_key[key],
            "votes": vote_counts[key],
            "confidence_total": round(confidence_totals[key], 3),
        }
        for key in candidate_by_key
        if vote_counts[key] > 0
    ]
    if not vote_rows:
        return None, vote_rows, False

    ranked_keys = sorted(
        (key for key in candidate_by_key if vote_counts[key] > 0),
        key=lambda key: (
            -vote_counts[key],
            -confidence_totals[key],
        ),
    )
    winner_key = ranked_keys[0]
    if len(ranked_keys) > 1:
        runner_up_key = ranked_keys[1]
        if (
            vote_counts[winner_key] == vote_counts[runner_up_key]
            and abs(confidence_totals[winner_key] - confidence_totals[runner_up_key])
            < 0.000001
        ):
            return None, vote_rows, True
    return candidate_by_key[winner_key], vote_rows, False


def _build_candidate_frame_metadata(
    *,
    phased_samples: Sequence[tuple[EvidencePhase, EvidenceSample]],
    candidate_matches: Sequence[KeyEntityCandidateFrameMatch],
) -> list[dict[str, object]]:
    matches_by_phase: dict[EvidencePhase, list[KeyEntityCandidateFrameMatch]] = {}
    for candidate_match in candidate_matches:
        matches_by_phase.setdefault(candidate_match.phase, []).append(candidate_match)

    frames: list[dict[str, object]] = []
    for phase, sample in phased_samples:
        phase_matches = matches_by_phase.get(phase, [])
        if not phase_matches:
            frames.append(
                {
                    "phase": phase,
                    "status": "skipped",
                    "reason": (
                        "missing_target_crop"
                        if sample.crop_bytes is None
                        else "sample_not_selected"
                    ),
                }
            )
            continue
        matched = any(match.match.key_entity_id is not None for match in phase_matches)
        frames.append(
            {
                "phase": phase,
                "status": "matched" if matched else "no_match",
                "input_source": "yolo_detection_crop",
                "input_region": "yolo_detection_box",
                "candidates": [
                    {
                        "candidate_id": candidate_match.candidate_id,
                        "status": (
                            "matched"
                            if candidate_match.match.key_entity_id is not None
                            else "no_match"
                        ),
                        "key_entity_id": candidate_match.match.key_entity_id,
                        "confidence": candidate_match.match.confidence,
                        "reason": candidate_match.match.reason,
                        "checked_at": candidate_match.match.checked_at.isoformat(),
                        "model": candidate_match.match.model_name,
                        "raw_output": candidate_match.match.raw_output,
                    }
                    for candidate_match in phase_matches
                ],
            }
        )
    return frames
