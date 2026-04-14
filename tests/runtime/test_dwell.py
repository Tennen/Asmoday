from datetime import UTC, datetime, timedelta

from vision_service.runtime.dwell import RuleDwellTracker, TrackEvidence


def test_completed_threshold_event_is_emitted_once_per_episode() -> None:
    tracker = RuleDwellTracker(
        threshold_seconds=5,
        sample_interval_seconds=0.5,
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
    assert len(completed.evidence_samples) == 3


def test_force_clear_emits_completed_threshold_event_for_active_episode() -> None:
    tracker = RuleDwellTracker(
        threshold_seconds=2,
        sample_interval_seconds=0.1,
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


def test_evidence_samples_span_the_full_episode_with_middle_near_midpoint() -> None:
    tracker = RuleDwellTracker(
        threshold_seconds=2,
        sample_interval_seconds=1.0,
        max_samples=9,
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

    start_sample, middle_sample, end_sample = completed.evidence_samples
    assert start_sample.captured_at == start
    assert end_sample.captured_at == start + timedelta(seconds=12)
    expected_midpoint = start + timedelta(seconds=6)
    assert abs((middle_sample.captured_at - expected_midpoint).total_seconds()) <= 2
