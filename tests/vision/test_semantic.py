from vision_service.contracts import (
    CameraIdentity,
    EntitySelector,
    RTSPSource,
    VisionRule,
    ZoneRect,
)
from vision_service.vision.semantic import build_semantic_prompt, parse_semantic_verdict


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
    assert prompt.endswith("仅回答：有、疑似有、无法确定。")


def test_parse_semantic_verdict_accepts_verbose_output() -> None:
    assert parse_semantic_verdict("结论：疑似有") == "疑似有"
    assert parse_semantic_verdict("无法确定，画面太糊") == "无法确定"
    assert parse_semantic_verdict("有。") == "有"
