#!/usr/bin/env python

from lerobot.utils.critical_phase_tracker import CriticalPhaseTracker, EpisodeIntervalTracker


def test_critical_phase_tracker_serializes_episode_intervals():
    tracker = CriticalPhaseTracker()
    tracker.on_episode_start(3)
    tracker.toggle(10)
    tracker.mark_success(18)

    assert tracker.serialize_episode_intervals(3) == [
        {"start_frame": 10, "end_frame": 18, "outcome": "success"}
    ]


def test_episode_interval_tracker_serializes_and_discards_episode_intervals():
    tracker = EpisodeIntervalTracker(label="Human intervention")
    tracker.on_episode_start(1)
    tracker.start(5)
    tracker.stop(9)
    tracker.start(12)
    tracker.on_episode_end(16)

    assert tracker.serialize_episode_intervals(1) == [
        {"start_frame": 5, "end_frame": 9},
        {"start_frame": 12, "end_frame": 16},
    ]

    tracker.discard_episode(1)
    assert tracker.serialize_episode_intervals(1) == []
