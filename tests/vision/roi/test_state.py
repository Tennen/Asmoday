from datetime import UTC, datetime, timedelta

from vision_service.vision.roi.state import ROIOccupancyStateMachine


def test_roi_state_machine_holds_warming_up_until_background_warmup_expires() -> None:
    machine = ROIOccupancyStateMachine(
        stay_threshold_seconds=2,
        warmup_seconds=1.0,
        clear_hold_seconds=1.0,
    )
    start = datetime(2026, 4, 12, 8, 0, 0, tzinfo=UTC)

    warming = machine.observe(
        observed_at=start,
        frame_present=True,
        occupancy_ratio=0.2,
        largest_blob_area=100,
        roi_area_pixels=1000,
        foreground_pixels=200,
    )
    active = machine.observe(
        observed_at=start + timedelta(seconds=1),
        frame_present=True,
        occupancy_ratio=0.2,
        largest_blob_area=100,
        roi_area_pixels=1000,
        foreground_pixels=200,
    )

    assert warming.state == "warming_up"
    assert not warming.presence_active
    assert active.state == "candidate_occupied"
    assert active.presence_active


def test_roi_state_machine_keeps_presence_active_during_clear_hold_window() -> None:
    machine = ROIOccupancyStateMachine(
        stay_threshold_seconds=2,
        warmup_seconds=0.0,
        clear_hold_seconds=1.0,
    )
    start = datetime(2026, 4, 12, 8, 5, 0, tzinfo=UTC)

    machine.observe(
        observed_at=start,
        frame_present=True,
        occupancy_ratio=0.2,
        largest_blob_area=100,
        roi_area_pixels=1000,
        foreground_pixels=200,
    )
    occupied = machine.observe(
        observed_at=start + timedelta(seconds=2),
        frame_present=True,
        occupancy_ratio=0.2,
        largest_blob_area=100,
        roi_area_pixels=1000,
        foreground_pixels=200,
    )
    held = machine.observe(
        observed_at=start + timedelta(seconds=2, milliseconds=500),
        frame_present=False,
        occupancy_ratio=0.0,
        largest_blob_area=0,
        roi_area_pixels=1000,
        foreground_pixels=0,
    )
    cleared = machine.observe(
        observed_at=start + timedelta(seconds=4),
        frame_present=False,
        occupancy_ratio=0.0,
        largest_blob_area=0,
        roi_area_pixels=1000,
        foreground_pixels=0,
    )

    assert occupied.state == "occupied"
    assert held.state == "occupied"
    assert held.presence_active
    assert cleared.state == "empty"
    assert not cleared.presence_active
