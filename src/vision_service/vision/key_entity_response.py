import json
from typing import Any, Sequence

from vision_service.contracts.control import KeyEntityId, KeyEntityReference


class KeyEntityMatchError(RuntimeError):
    pass


def parse_match_output(
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


def extract_message_text(response_json: dict[str, Any]) -> str:
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
