from __future__ import annotations

from pathlib import Path
from typing import Protocol

from badminton_coach.types import FramePoses


class PoseEstimator(Protocol):
    def extract(self, video_path: Path) -> list[FramePoses]: ...
