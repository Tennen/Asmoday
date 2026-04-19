from datetime import UTC, datetime, timedelta

from vision_service.runtime.dwell import RuleDwellTracker, TrackEvidence


def test_completed_threshold_event_is_emitted_once_per_episode() -> None:
    tracker = RuleDwellTracker(
        threshold_seconds=5,
    )
    start = datetime(2026, 4, 9, 8, 0, 0, tzinfo=UTC)

    for second in range(8):
        transition = tracker.observe(
            observed_at=start + timedelta(seconds=second),
            visible_tracks={7: TrackEvidence(image_bytes=f"frame-{second}".encode())},
        )
        assert transition is None

    completed = tracker.observe(
        observed_at=start + timedelta(seconds=8),
        visible_tracks={},
    )
    assert completed is not None
    assert completed.status == "threshold_met"
    assert completed.track_id == 7
    assert completed.dwell_seconds == 7
    assert len(completed.evidence_samples) == 4


def test_force_clear_emits_completed_threshold_event_for_active_episode() -> None:
    tracker = RuleDwellTracker(
        threshold_seconds=2,
    )
    observed_at = datetime(2026, 4, 9, 8, 5, 0, tzinfo=UTC)

    assert tracker.force_clear(observed_at=observed_at) is None

    tracker.observe(
        observed_at=observed_at,
        visible_tracks={1: TrackEvidence(image_bytes=b"frame")},
    )
    tracker.observe(
        observed_at=observed_at + timedelta(seconds=2),
        visible_tracks={1: TrackEvidence(image_bytes=b"frame")},
    )

    cleared = tracker.force_clear(
        observed_at=observed_at + timedelta(seconds=3),
    )
    assert cleared is not None
    assert cleared.status == "threshold_met"
    assert cleared.track_id == 1
    assert cleared.dwell_seconds == 2


def test_evidence_samples_use_threshold_derived_interval() -> None:
    tracker = RuleDwellTracker(
        threshold_seconds=6,
        max_samples=20,
    )
    start = datetime(2026, 4, 9, 8, 10, 0, tzinfo=UTC)

    for second in range(13):
        transition = tracker.observe(
            observed_at=start + timedelta(seconds=second),
            visible_tracks={7: TrackEvidence(image_bytes=f"frame-{second}".encode())},
        )
        assert transition is None

    completed = tracker.observe(
        observed_at=start + timedelta(seconds=13),
        visible_tracks={},
    )
    assert completed is not None

    assert [
        sample.captured_at for sample in completed.evidence_samples
    ] == [
        start + timedelta(seconds=second)
        for second in (0, 2, 4, 6, 8, 10, 12)
    ]


def test_evidence_samples_stop_at_configured_buffer_limit() -> None:
    tracker = RuleDwellTracker(
        threshold_seconds=3,
        max_samples=4,
    )
    start = datetime(2026, 4, 9, 8, 20, 0, tzinfo=UTC)

    for second in range(8):
        tracker.observe(
            observed_at=start + timedelta(seconds=second),
            visible_tracks={7: TrackEvidence(image_bytes=f"frame-{second}".encode())},
        )

    completed = tracker.observe(
        observed_at=start + timedelta(seconds=8),
        visible_tracks={},
    )
    assert completed is not None
    assert len(completed.evidence_samples) == 4
    assert completed.evidence_samples[-1].captured_at == start + timedelta(seconds=3)
