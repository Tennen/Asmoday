import asyncio
from base64 import b64encode
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from typing import Any, Literal, Protocol
from urllib import error, request

from vision_service.contracts import VisionRule
from vision_service.settings import Settings

SemanticVerdict = Literal["有", "疑似有", "无法确定"]


class SemanticCheckError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class SemanticCheckResult:
    verdict: SemanticVerdict
    raw_output: str
    model_name: str
    checked_at: datetime


class SemanticChecker(Protocol):
    async def check(
        self,
        *,
        image_bytes: bytes,
        rule: VisionRule,
    ) -> SemanticCheckResult: ...


def build_semantic_prompt(rule: VisionRule) -> str:
    entity = _humanize_entity(rule.entity_selector.value)
    behavior = rule.behavior
    if entity and behavior:
        question = f"这张图是监控目标区域截图。区域内是否有{entity}正在{behavior}？"
    elif entity:
        question = f"这张图是监控目标区域截图。区域内是否有{entity}？"
    elif behavior:
        question = f"这张图是监控目标区域截图。区域内是否有目标正在{behavior}？"
    else:
        question = "这张图是监控目标区域截图。区域内是否有明显的目标对象？"
    instructions = (
        "只有在画面中直接看见与目标相符的真实实体时，才回答“有”或“疑似有”。"
        "不要根据食盆、玩具、阴影、反光、纹理、静止器具或历史画面猜测。"
        "如果目标不存在、看不清，或无法确认，回答“无法确定”。"
    )
    return f"{question} {instructions} 仅回答：有、疑似有、无法确定。"


class OpenAICompatibleSemanticChecker:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str | None,
        timeout_seconds: float,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
    ) -> "OpenAICompatibleSemanticChecker | None":
        if not settings.semantic_checker_enabled:
            return None
        assert settings.semantic_checker_base_url is not None
        assert settings.semantic_checker_model_name is not None
        return cls(
            base_url=settings.semantic_checker_base_url,
            model_name=settings.semantic_checker_model_name,
            api_key=settings.semantic_checker_api_key,
            timeout_seconds=settings.semantic_checker_timeout_seconds,
        )

    async def check(
        self,
        *,
        image_bytes: bytes,
        rule: VisionRule,
    ) -> SemanticCheckResult:
        prompt = build_semantic_prompt(rule)
        payload = {
            "model": self._model_name,
            "temperature": 0,
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/jpeg;base64,"
                                    + b64encode(image_bytes).decode("ascii")
                                )
                            },
                        },
                    ],
                }
            ],
        }
        response_json = await asyncio.to_thread(self._request_json, payload)
        raw_output = _extract_message_text(response_json)
        verdict = parse_semantic_verdict(raw_output)
        return SemanticCheckResult(
            verdict=verdict,
            raw_output=raw_output,
            model_name=self._model_name,
            checked_at=datetime.now(tz=UTC),
        )

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
            raise SemanticCheckError(
                f"semantic checker request failed status={exc.code} body={details}"
            ) from exc
        except error.URLError as exc:
            raise SemanticCheckError(
                f"semantic checker request failed reason={exc.reason}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise SemanticCheckError("semantic checker returned invalid JSON") from exc


def parse_semantic_verdict(text: str) -> SemanticVerdict:
    normalized = "".join(text.split())
    if "疑似有" in normalized:
        return "疑似有"
    if "无法确定" in normalized:
        return "无法确定"
    if "有" in normalized:
        return "有"
    raise SemanticCheckError(f"semantic checker returned unsupported verdict: {text!r}")


def _extract_message_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SemanticCheckError("semantic checker response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise SemanticCheckError("semantic checker response is missing message")
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
        raise SemanticCheckError("semantic checker response content is empty")
    return text


def _humanize_entity(entity_value: str) -> str | None:
    stripped = entity_value.strip()
    if not stripped:
        return None
    return stripped.replace("_", " ")
