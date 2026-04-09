from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Mapping


TransitionStatus = Literal["threshold_met", "cleared"]


@dataclass(slots=True, frozen=True)
class EvidenceSample:
    captured_at: datetime
    image_bytes: bytes


@dataclass(slots=True, frozen=True)
class DwellTransition:
    status: TransitionStatus
    observed_at: datetime
    dwell_seconds: int
    track_id: int | None = None
    evidence_samples: tuple[EvidenceSample, ...] = ()


@dataclass(slots=True)
class TrackEpisode:
    track_id: int
    entered_at: datetime
    last_seen_at: datetime
    last_sampled_at: datetime | None = None
    threshold_met: bool = False
    last_dwell_seconds: int = 0
    samples: list[EvidenceSample] = field(default_factory=list)


class RuleDwellTracker:
    def __init__(
        self,
        *,
        threshold_seconds: int,
        sample_interval_seconds: float,
        max_samples: int = 32,
    ) -> None:
        self._threshold_seconds = threshold_seconds
        self._sample_interval_seconds = sample_interval_seconds
        self._max_samples = max_samples
        self._tracks: dict[int, TrackEpisode] = {}
        self._active = False
        self._active_track_id: int | None = None
        self._last_active_dwell_seconds = 0

    @property
    def active(self) -> bool:
        return self._active

    def observe(
        self,
        *,
        observed_at: datetime,
        visible_tracks: Mapping[int, bytes | None],
    ) -> DwellTransition | None:
        current_ids = set(visible_tracks)
        for track_id in list(self._tracks):
            if track_id not in current_ids:
                del self._tracks[track_id]

        for track_id, image_bytes in visible_tracks.items():
            episode = self._tracks.get(track_id)
            if episode is None:
                episode = TrackEpisode(
                    track_id=track_id,
                    entered_at=observed_at,
                    last_seen_at=observed_at,
                )
                self._tracks[track_id] = episode

            episode.last_seen_at = observed_at
            episode.last_dwell_seconds = max(
                0,
                int((observed_at - episode.entered_at).total_seconds()),
            )

            crossed_threshold = (
                not episode.threshold_met
                and episode.last_dwell_seconds >= self._threshold_seconds
            )
            self._maybe_store_sample(
                episode=episode,
                observed_at=observed_at,
                image_bytes=image_bytes,
                force=crossed_threshold,
            )
            if crossed_threshold:
                episode.threshold_met = True

        qualifying = [
            episode
            for episode in self._tracks.values()
            if episode.threshold_met
        ]

        if not self._active and qualifying:
            chosen = min(
                qualifying,
                key=lambda episode: (episode.entered_at, episode.track_id),
            )
            self._active = True
            self._active_track_id = chosen.track_id
            self._last_active_dwell_seconds = max(
                episode.last_dwell_seconds for episode in qualifying
            )
            return DwellTransition(
                status="threshold_met",
                observed_at=observed_at,
                dwell_seconds=chosen.last_dwell_seconds,
                track_id=chosen.track_id,
                evidence_samples=self._select_evidence_samples(chosen),
            )

        if self._active and not qualifying:
            return self.force_clear(observed_at=observed_at)

        if self._active and qualifying:
            self._last_active_dwell_seconds = max(
                episode.last_dwell_seconds for episode in qualifying
            )
            if self._active_track_id not in {episode.track_id for episode in qualifying}:
                replacement = min(
                    qualifying,
                    key=lambda episode: (episode.entered_at, episode.track_id),
                )
                self._active_track_id = replacement.track_id

        return None

    def force_clear(self, *, observed_at: datetime) -> DwellTransition | None:
        if not self._active:
            return None

        transition = DwellTransition(
            status="cleared",
            observed_at=observed_at,
            dwell_seconds=self._last_active_dwell_seconds,
            track_id=self._active_track_id,
        )
        self._active = False
        self._active_track_id = None
        self._last_active_dwell_seconds = 0
        self._tracks.clear()
        return transition

    def _maybe_store_sample(
        self,
        *,
        episode: TrackEpisode,
        observed_at: datetime,
        image_bytes: bytes | None,
        force: bool,
    ) -> None:
        if image_bytes is None:
            return
        if (
            not force
            and episode.last_sampled_at is not None
            and (
                observed_at - episode.last_sampled_at
            ).total_seconds()
            < self._sample_interval_seconds
        ):
            return

        episode.samples.append(
            EvidenceSample(
                captured_at=observed_at,
                image_bytes=image_bytes,
            )
        )
        if len(episode.samples) > self._max_samples:
            episode.samples.pop(0)
        episode.last_sampled_at = observed_at

    @staticmethod
    def _select_evidence_samples(
        episode: TrackEpisode,
    ) -> tuple[EvidenceSample, ...]:
        if not episode.samples:
            return ()

        start = episode.samples[0]
        middle = episode.samples[len(episode.samples) // 2]
        end = episode.samples[-1]
        return (start, middle, end)
