from pathlib import Path

import pytest

from badminton_coach.errors import CoachError
from badminton_coach.pose.fake import FakePoseEstimator
from badminton_coach.pose.mediapipe_backend import MediaPipePoseEstimator
from badminton_coach.types import FramePoses, Landmark, PersonPose


def test_fake_estimator_returns_injected_frames(tmp_path: Path):
    frames = [
        FramePoses(
            0,
            0.0,
            [PersonPose(0, [Landmark(0.1, 0.2, 0.0, 1.0) for _ in range(33)])],
        )
    ]
    est = FakePoseEstimator(frames)
    assert est.extract(tmp_path / "missing.mp4") is frames


def test_mediapipe_missing_model_raises(tmp_path: Path):
    est = MediaPipePoseEstimator(tmp_path / "nope.task")
    with pytest.raises(CoachError) as ei:
        est.extract(tmp_path / "video.mp4")
    assert ei.value.code == "MODEL_MISSING"
    assert "模型" in ei.value.user_message
