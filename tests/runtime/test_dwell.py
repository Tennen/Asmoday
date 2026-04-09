from datetime import UTC, datetime, timedelta

from vision_service.runtime.dwell import RuleDwellTracker


def test_threshold_is_emitted_once_per_episode() -> None:
    tracker = RuleDwellTracker(
        threshold_seconds=5,
        sample_interval_seconds=0.5,
    )
    start = datetime(2026, 4, 9, 8, 0, 0, tzinfo=UTC)

    for second in range(5):
        transition = tracker.observe(
            observed_at=start + timedelta(seconds=second),
            visible_tracks={7: b"frame"},
        )
        assert transition is None

    threshold_met = tracker.observe(
        observed_at=start + timedelta(seconds=5),
        visible_tracks={7: b"frame"},
    )
    assert threshold_met is not None
    assert threshold_met.status == "threshold_met"
    assert threshold_met.track_id == 7
    assert threshold_met.dwell_seconds == 5
    assert len(threshold_met.evidence_samples) == 3

    still_active = tracker.observe(
        observed_at=start + timedelta(seconds=6),
        visible_tracks={7: b"frame"},
    )
    assert still_active is None

    cleared = tracker.observe(
        observed_at=start + timedelta(seconds=7),
        visible_tracks={},
    )
    assert cleared is not None
    assert cleared.status == "cleared"
    assert cleared.track_id == 7
    assert cleared.dwell_seconds >= 5


def test_force_clear_only_fires_when_active() -> None:
    tracker = RuleDwellTracker(
        threshold_seconds=2,
        sample_interval_seconds=0.1,
    )
    observed_at = datetime(2026, 4, 9, 8, 5, 0, tzinfo=UTC)

    assert tracker.force_clear(observed_at=observed_at) is None

    tracker.observe(
        observed_at=observed_at,
        visible_tracks={1: b"frame"},
    )
    tracker.observe(
        observed_at=observed_at + timedelta(seconds=2),
        visible_tracks={1: b"frame"},
    )

    cleared = tracker.force_clear(
        observed_at=observed_at + timedelta(seconds=3),
    )
    assert cleared is not None
    assert cleared.status == "cleared"
    assert cleared.track_id == 1
