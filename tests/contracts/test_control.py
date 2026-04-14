import pytest

from vision_service.contracts import EntitySelector, VisionRule


def test_entity_selector_allows_empty_value_for_wildcard_rules() -> None:
    selector = EntitySelector(value="")

    assert selector.value == ""


def test_vision_rule_normalizes_blank_behavior_to_none() -> None:
    rule = VisionRule.model_validate(
        {
            "id": "rule-1",
            "name": "Rule 1",
            "enabled": True,
            "camera": {"device_id": "camera-1"},
            "rtsp_source": {"url": "rtsp://camera/stream"},
            "entity_selector": {"value": "cat"},
            "behavior": "   ",
            "zone": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
            "stay_threshold_seconds": 5,
        }
    )

    assert rule.behavior is None


def test_vision_rule_accepts_key_entity_with_image_only() -> None:
    rule = VisionRule.model_validate(
        {
            "id": "rule-1",
            "name": "Rule 1",
            "enabled": True,
            "camera": {"device_id": "camera-1"},
            "rtsp_source": {"url": "rtsp://camera/stream"},
            "entity_selector": {"value": "cat"},
            "key_entities": [
                {
                    "id": 1,
                    "image": {"base64": "aW1hZ2U="},
                }
            ],
            "zone": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
            "stay_threshold_seconds": 5,
        }
    )

    assert rule.key_entities[0].id == 1
    assert rule.key_entities[0].description is None


def test_vision_rule_rejects_key_entity_without_image_and_description() -> None:
    with pytest.raises(ValueError, match="image or description"):
        VisionRule.model_validate(
            {
                "id": "rule-1",
                "name": "Rule 1",
                "enabled": True,
                "camera": {"device_id": "camera-1"},
                "rtsp_source": {"url": "rtsp://camera/stream"},
                "entity_selector": {"value": "cat"},
                "key_entities": [{"id": 1}],
                "zone": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                "stay_threshold_seconds": 5,
            }
        )
