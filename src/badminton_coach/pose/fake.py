from __future__ import annotations

from pathlib import Path

from badminton_coach.types import FramePoses


class FakePoseEstimator:
    def __init__(self, frames: list[FramePoses]):
        self._frames = frames

    def extract(self, video_path: Path) -> list[FramePoses]:
        return self._frames
