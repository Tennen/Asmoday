import asyncio
from threading import Lock
from time import sleep

import pytest

from vision_service.contracts import (
    CameraIdentity,
    EntitySelector,
    KeyEntityReference,
    RTSPSource,
    VisionRule,
    ZoneRect,
)
from vision_service.settings import Settings
from vision_service.vision.key_entity_matcher import OpenAICompatibleKeyEntityMatcher
from vision_service.vision.model_queue import LocalModelRequestQueue
from vision_service.vision.semantic import (
    OpenAICompatibleSemanticChecker,
    build_semantic_prompt,
    parse_semantic_verdict,
)


def build_rule(*, entity_value: str = "cat", behavior: str | None = None) -> VisionRule:
    return VisionRule(
        id="rule-1",
        name="Rule 1",
        enabled=True,
        camera=CameraIdentity(device_id="camera-1"),
        rtsp_source=RTSPSource(url="rtsp://camera/stream"),
        entity_selector=EntitySelector(value=entity_value),
        behavior=behavior,
        zone=ZoneRect(x=0.1, y=0.1, width=0.2, height=0.2),
        stay_threshold_seconds=5,
    )


def test_build_semantic_prompt_uses_entity_and_behavior() -> None:
    prompt = build_semantic_prompt(build_rule(entity_value="orange_cat", behavior="吃东西"))

    assert "orange cat" in prompt
    assert "吃东西" in prompt
    assert "不要根据食盆、玩具、阴影、反光、纹理、静止器具或历史画面猜测" in prompt
    assert prompt.endswith("仅回答：有、疑似有、无法确定。")


def test_parse_semantic_verdict_accepts_verbose_output() -> None:
    assert parse_semantic_verdict("结论：疑似有") == "疑似有"
    assert parse_semantic_verdict("无法确定，画面太糊") == "无法确定"
    assert parse_semantic_verdict("有。") == "有"


def test_semantic_checker_default_timeout_allows_local_vlm_latency() -> None:
    assert Settings().semantic_checker_timeout_seconds == 20.0


@pytest.mark.asyncio
async def test_local_model_requests_are_serialized_across_semantic_clients() -> None:
    request_queue = LocalModelRequestQueue()
    semantic_checker = OpenAICompatibleSemanticChecker(
        base_url="http://model.local/v1",
        model_name="mini-vlm",
        api_key=None,
        timeout_seconds=1.0,
        request_queue=request_queue,
    )
    key_entity_matcher = OpenAICompatibleKeyEntityMatcher(
        base_url="http://model.local/v1",
        model_name="mini-vlm",
        api_key=None,
        timeout_seconds=1.0,
        request_queue=request_queue,
    )
    state_lock = Lock()
    active_requests = 0
    max_active_requests = 0
    request_order: list[str] = []

    def request_json(label: str) -> dict[str, object]:
        nonlocal active_requests, max_active_requests
        with state_lock:
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
            request_order.append(f"start:{label}")
        sleep(0.01)
        with state_lock:
            active_requests -= 1
            request_order.append(f"end:{label}")
        if label == "semantic":
            return {"choices": [{"message": {"content": "有"}}]}
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"key_entity_id":"pet-1","confidence":0.8,'
                            '"reason":"match"}'
                        )
                    }
                }
            ]
        }

    semantic_checker._request_json = (  # type: ignore[method-assign]
        lambda payload: request_json("semantic")
    )
    key_entity_matcher._request_json = (  # type: ignore[method-assign]
        lambda payload: request_json("key_entity")
    )

    await asyncio.gather(
        semantic_checker.check(image_bytes=b"image", rule=build_rule()),
        key_entity_matcher.match(
            image_bytes=b"crop",
            key_entities=[KeyEntityReference(id="pet-1", description="orange cat")],
        ),
    )

    assert max_active_requests == 1
    assert request_order in (
        ["start:semantic", "end:semantic", "start:key_entity", "end:key_entity"],
        ["start:key_entity", "end:key_entity", "start:semantic", "end:semantic"],
    )
