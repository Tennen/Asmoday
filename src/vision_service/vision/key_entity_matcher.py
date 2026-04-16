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
from vision_service.vision.model_queue import (
    LocalModelRequestQueue,
    get_local_model_request_queue,
)


class KeyEntityMatchError(RuntimeError):
    pass


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
        raw_output = _extract_message_text(response_json)
        matched_id, confidence, reason = _parse_match_output(
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
                    "你是监控事件关键实体匹配器。第一张图是待识别的单个实体裁切图。"
                    "后续会给出候选关键实体，每个候选至少有参考图或文字描述。"
                    "请只在候选 id 中选择一个最匹配的 id，或者返回 null。"
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
        frame_matches = []
        for _, sample in usable_samples:
            frame_matches.append(
                await matcher.match(
                    image_bytes=sample.crop_bytes,  # type: ignore[arg-type]
                    key_entities=key_entities,
                )
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

    winner_id, votes = _aggregate_frame_matches(
        key_entities=key_entities,
        frame_matches=frame_matches,
    )
    metadata = {
        "status": "matched" if winner_id is not None else "no_match",
        "winner_id": winner_id,
        "model": frame_matches[0].model_name,
        "frames": _build_frame_metadata(
            phased_samples=phased_samples,
            usable_samples=usable_samples,
            frame_matches=frame_matches,
        ),
        "votes": votes,
    }
    return KeyEntityIdentification(
        key_entity_id=winner_id,
        metadata=metadata,
    )


def _aggregate_frame_matches(
    *,
    key_entities: Sequence[KeyEntityReference],
    frame_matches: Sequence[KeyEntityFrameMatch],
) -> tuple[KeyEntityId | None, list[dict[str, object]]]:
    candidate_by_key = {str(entity.id): entity.id for entity in key_entities}
    candidate_order = {str(entity.id): index for index, entity in enumerate(key_entities)}
    vote_counts = {key: 0 for key in candidate_by_key}
    confidence_totals = {key: 0.0 for key in candidate_by_key}

    for match in frame_matches:
        if match.key_entity_id is None:
            continue
        candidate_key = str(match.key_entity_id)
        vote_counts[candidate_key] += 1
        confidence_totals[candidate_key] += match.confidence or 0.0

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
        return None, vote_rows

    winner_key = min(
        (key for key in candidate_by_key if vote_counts[key] > 0),
        key=lambda key: (
            -vote_counts[key],
            -confidence_totals[key],
            candidate_order[key],
        ),
    )
    return candidate_by_key[winner_key], vote_rows


def _build_frame_metadata(
    *,
    phased_samples: Sequence[tuple[EvidencePhase, EvidenceSample]],
    usable_samples: Sequence[tuple[EvidencePhase, EvidenceSample]],
    frame_matches: Sequence[KeyEntityFrameMatch],
) -> list[dict[str, object]]:
    match_by_phase = {
        phase: match
        for (phase, _), match in zip(usable_samples, frame_matches)
    }
    frames: list[dict[str, object]] = []
    for phase, sample in phased_samples:
        match = match_by_phase.get(phase)
        if match is None:
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
        frames.append(
            {
                "phase": phase,
                "status": "matched" if match.key_entity_id is not None else "no_match",
                "key_entity_id": match.key_entity_id,
                "confidence": match.confidence,
                "reason": match.reason,
                "checked_at": match.checked_at.isoformat(),
                "model": match.model_name,
            }
        )
    return frames


def _parse_match_output(
    *,
    text: str,
    key_entities: Sequence[KeyEntityReference],
) -> tuple[KeyEntityId | None, float | None, str | None]:
    parsed = _try_parse_json_object(text)
    candidate_by_key = {str(entity.id): entity.id for entity in key_entities}

    if parsed is None:
        normalized_id = _normalize_matched_id(
            value=text.strip(),
            candidate_by_key=candidate_by_key,
        )
        return normalized_id, None, None

    matched_id = _normalize_matched_id(
        value=parsed.get("key_entity_id"),
        candidate_by_key=candidate_by_key,
    )
    confidence = _parse_confidence(parsed.get("confidence"))
    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = None
    else:
        reason = reason.strip()
    return matched_id, confidence, reason


def _normalize_matched_id(
    *,
    value: object,
    candidate_by_key: dict[str, KeyEntityId],
) -> KeyEntityId | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"null", "none"}:
        return None
    if normalized not in candidate_by_key:
        raise KeyEntityMatchError(
            f"key entity matcher returned unsupported id: {value!r}"
        )
    return candidate_by_key[normalized]


def _parse_confidence(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        confidence = float(value)
        return max(0.0, min(1.0, confidence))
    return None


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        raise KeyEntityMatchError("key entity matcher response content is empty")

    candidate_json = stripped
    if candidate_json.startswith("```"):
        lines = [
            line
            for line in candidate_json.splitlines()
            if not line.strip().startswith("```")
        ]
        candidate_json = "\n".join(lines).strip()
    start = candidate_json.find("{")
    end = candidate_json.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(candidate_json[start : end + 1])
    except json.JSONDecodeError as exc:
        raise KeyEntityMatchError(
            f"key entity matcher returned invalid JSON payload: {text!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise KeyEntityMatchError("key entity matcher JSON payload must be an object")
    return parsed


def _extract_message_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise KeyEntityMatchError("key entity matcher response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise KeyEntityMatchError("key entity matcher response is missing message")
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text_value = item.get("text")
            if isinstance(text_value, str) and text_value.strip():
                text_parts.append(text_value.strip())
        text = "\n".join(text_parts).strip()
    else:
        text = ""
    if not text:
        raise KeyEntityMatchError("key entity matcher response content is empty")
    return text
