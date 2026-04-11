from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Mapping


TransitionStatus = Literal["threshold_met"]


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
        removed_tracks = {
            track_id: episode
            for track_id, episode in list(self._tracks.items())
            if track_id not in current_ids
        }
        for track_id in removed_tracks:
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

        if qualifying:
            chosen = min(
                qualifying,
                key=lambda episode: (episode.entered_at, episode.track_id),
            )
            self._active = True
            self._active_track_id = chosen.track_id
            self._last_active_dwell_seconds = chosen.last_dwell_seconds
            return None

        return self._finalize_active_transition(
            observed_at=observed_at,
            removed_tracks=removed_tracks,
        )

    def force_clear(self, *, observed_at: datetime) -> DwellTransition | None:
        transition = self._finalize_active_transition(
            observed_at=observed_at,
            removed_tracks={},
        )
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
            episode.samples = self._rebalance_samples(
                samples=episode.samples,
                max_samples=self._max_samples,
            )
        episode.last_sampled_at = observed_at

    def _finalize_active_transition(
        self,
        *,
        observed_at: datetime,
        removed_tracks: Mapping[int, TrackEpisode],
    ) -> DwellTransition | None:
        if not self._active:
            return None

        episode = self._episode_for_active_track(removed_tracks=removed_tracks)
        transition = DwellTransition(
            status="threshold_met",
            observed_at=observed_at,
            dwell_seconds=(
                episode.last_dwell_seconds
                if episode is not None
                else self._last_active_dwell_seconds
            ),
            track_id=episode.track_id if episode is not None else self._active_track_id,
            evidence_samples=(
                self._select_evidence_samples(episode)
                if episode is not None
                else ()
            ),
        )
        self._active = False
        self._active_track_id = None
        self._last_active_dwell_seconds = 0
        return transition

    def _episode_for_active_track(
        self,
        *,
        removed_tracks: Mapping[int, TrackEpisode],
    ) -> TrackEpisode | None:
        if self._active_track_id is not None:
            episode = self._tracks.get(self._active_track_id)
            if episode is not None:
                return episode
            episode = removed_tracks.get(self._active_track_id)
            if episode is not None:
                return episode

        completed_tracks = [
            episode for episode in removed_tracks.values() if episode.threshold_met
        ]
        if not completed_tracks:
            return None
        return min(
            completed_tracks,
            key=lambda episode: (episode.entered_at, episode.track_id),
        )

    @staticmethod
    def _rebalance_samples(
        *,
        samples: list[EvidenceSample],
        max_samples: int,
    ) -> list[EvidenceSample]:
        if len(samples) <= max_samples:
            return samples
        if max_samples <= 1:
            return [samples[-1]]

        last_index = len(samples) - 1
        step = last_index / (max_samples - 1)
        chosen_indices: list[int] = []
        previous_index = -1
        for slot in range(max_samples):
            raw_index = round(slot * step)
            min_index = previous_index + 1
            max_index = last_index - (max_samples - slot - 1)
            index = min(max(raw_index, min_index), max_index)
            chosen_indices.append(index)
            previous_index = index
        return [samples[index] for index in chosen_indices]

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
