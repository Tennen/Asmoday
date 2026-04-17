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
            self.candidate_ids: list[str] = []

        async def match(self, *, image_bytes: bytes, key_entities):  # noqa: ANN001
            self.active_requests += 1
            self.max_active_requests = max(
                self.max_active_requests,
                self.active_requests,
            )
            self.image_bytes.append(image_bytes)
            self.candidate_ids.append(key_entities[0].id)
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
    assert matcher.candidate_ids == ["pet-1", "pet-1", "pet-1"]


@pytest.mark.asyncio
async def test_identify_key_entity_compares_each_candidate_and_records_raw_output() -> None:
    class FakeMatcher:
        def __init__(self) -> None:
            self.calls: list[tuple[bytes, str]] = []

        async def match(self, *, image_bytes: bytes, key_entities):  # noqa: ANN001
            assert len(key_entities) == 1
            candidate_id = key_entities[0].id
            self.calls.append((image_bytes, candidate_id))
            if candidate_id == "pet-2":
                return KeyEntityFrameMatch(
                    key_entity_id="pet-2",
                    confidence=0.93,
                    reason="纹理一致",
                    raw_output=(
                        '{"key_entity_id":"pet-2","confidence":0.93,'
                        '"reason":"纹理一致"}'
                    ),
                    model_name="mini-vlm",
                    checked_at=datetime(2026, 4, 16, 8, 0, tzinfo=UTC),
                )
            return KeyEntityFrameMatch(
                key_entity_id=None,
                confidence=0.2,
                reason="不像",
                raw_output='{"key_entity_id":null,"confidence":0.2,"reason":"不像"}',
                model_name="mini-vlm",
                checked_at=datetime(2026, 4, 16, 8, 0, tzinfo=UTC),
            )

    matcher = FakeMatcher()

    result = await identify_key_entity(
        evidence_samples=[
            EvidenceSample(
                captured_at=datetime(2026, 4, 16, 8, 0, tzinfo=UTC),
                image_bytes=b"frame",
                crop_bytes=b"crop",
            ),
        ],
        key_entities=[
            KeyEntityReference(id="pet-1", description="orange cat"),
            KeyEntityReference(id="pet-2", description="gray cat"),
        ],
        matcher=matcher,
    )

    assert result is not None
    assert result.key_entity_id == "pet-2"
    assert matcher.calls == [(b"crop", "pet-1"), (b"crop", "pet-2")]
    assert result.metadata["strategy"] == "pairwise_candidate_vote"
    assert result.metadata["input_source"] == "yolo_detection_crop"
    frame = result.metadata["frames"][0]  # type: ignore[index]
    assert frame["input_region"] == "yolo_detection_box"
    assert frame["candidates"][0]["raw_output"] == (  # type: ignore[index]
        '{"key_entity_id":null,"confidence":0.2,"reason":"不像"}'
    )


@pytest.mark.asyncio
async def test_identify_key_entity_returns_ambiguous_when_candidates_tie() -> None:
    class AlwaysMatches:
        async def match(self, *, image_bytes: bytes, key_entities):  # noqa: ANN001
            candidate_id = key_entities[0].id
            return KeyEntityFrameMatch(
                key_entity_id=candidate_id,
                confidence=0.8,
                reason="都像",
                raw_output=f'{{"key_entity_id":"{candidate_id}","confidence":0.8}}',
                model_name="mini-vlm",
                checked_at=datetime(2026, 4, 16, 8, 0, tzinfo=UTC),
            )

    result = await identify_key_entity(
        evidence_samples=[
            EvidenceSample(
                captured_at=datetime(2026, 4, 16, 8, 0, tzinfo=UTC),
                image_bytes=b"frame",
                crop_bytes=b"crop",
            ),
        ],
        key_entities=[
            KeyEntityReference(id="pet-1", description="orange cat"),
            KeyEntityReference(id="pet-2", description="gray cat"),
        ],
        matcher=AlwaysMatches(),
    )

    assert result is not None
    assert result.key_entity_id is None
    assert result.metadata["status"] == "ambiguous"
    assert result.metadata["winner_id"] is None


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
