import asyncio
from datetime import UTC, datetime

import pytest

from vision_service.contracts import KeyEntityReference
from vision_service.runtime.dwell import EvidenceSample
from vision_service.vision.key_entity_matcher import (
    KeyEntityFrameMatch,
    KeyEntityMatchError,
    OpenAICompatibleKeyEntityMatcher,
    identify_key_entity,
)


@pytest.mark.asyncio
async def test_identify_key_entity_matches_evidence_samples_sequentially() -> None:
    class FakeMatcher:
        def __init__(self) -> None:
            self.active_requests = 0
            self.max_active_requests = 0
            self.image_bytes: list[bytes] = []

        async def match(self, *, image_bytes: bytes, key_entities):  # noqa: ANN001
            self.active_requests += 1
            self.max_active_requests = max(
                self.max_active_requests,
                self.active_requests,
            )
            self.image_bytes.append(image_bytes)
            await asyncio.sleep(0.01)
            self.active_requests -= 1
            return KeyEntityFrameMatch(
                key_entity_id="pet-1",
                confidence=0.8,
                reason="match",
                raw_output='{"key_entity_id":"pet-1"}',
                model_name="mini-vlm",
                checked_at=datetime(2026, 4, 16, 8, 0, tzinfo=UTC),
            )

    matcher = FakeMatcher()

    result = await identify_key_entity(
        evidence_samples=[
            EvidenceSample(
                captured_at=datetime(2026, 4, 16, 7, 59, 58, tzinfo=UTC),
                image_bytes=b"start",
                crop_bytes=b"crop-start",
            ),
            EvidenceSample(
                captured_at=datetime(2026, 4, 16, 8, 0, tzinfo=UTC),
                image_bytes=b"middle",
                crop_bytes=b"crop-middle",
            ),
            EvidenceSample(
                captured_at=datetime(2026, 4, 16, 8, 0, 2, tzinfo=UTC),
                image_bytes=b"end",
                crop_bytes=b"crop-end",
            ),
        ],
        key_entities=[KeyEntityReference(id="pet-1", description="orange cat")],
        matcher=matcher,
    )

    assert result is not None
    assert result.key_entity_id == "pet-1"
    assert matcher.max_active_requests == 1
    assert matcher.image_bytes == [b"crop-start", b"crop-middle", b"crop-end"]


@pytest.mark.asyncio
async def test_identify_key_entity_returns_degraded_result_for_unexpected_error() -> None:
    class FailingMatcher:
        async def match(self, *, image_bytes: bytes, key_entities):  # noqa: ANN001, ANN201
            raise RuntimeError("model server crashed")

    result = await identify_key_entity(
        evidence_samples=[
            EvidenceSample(
                captured_at=datetime(2026, 4, 16, 8, 0, tzinfo=UTC),
                image_bytes=b"frame",
                crop_bytes=b"crop",
            ),
        ],
        key_entities=[KeyEntityReference(id="pet-1", description="orange cat")],
        matcher=FailingMatcher(),
    )

    assert result is not None
    assert result.key_entity_id is None
    assert result.metadata["status"] == "failed"
    assert result.error_message is not None
    assert "model server crashed" in result.error_message


def test_key_entity_matcher_wraps_socket_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matcher = OpenAICompatibleKeyEntityMatcher(
        base_url="http://model.local/v1",
        model_name="mini-vlm",
        api_key=None,
        timeout_seconds=1.0,
    )

    def raise_timeout(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise TimeoutError("timed out")

    monkeypatch.setattr(
        "vision_service.vision.key_entity_matcher.request.urlopen",
        raise_timeout,
    )

    with pytest.raises(KeyEntityMatchError, match="timed out"):
        matcher._request_json({"messages": []})
