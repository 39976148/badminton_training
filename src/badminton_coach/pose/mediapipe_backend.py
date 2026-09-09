from __future__ import annotations

from pathlib import Path

import numpy as np

from badminton_coach.errors import CoachError
from badminton_coach.types import FramePoses, Landmark, PersonPose


class MediaPipePoseEstimator:
    def __init__(self, model_path: Path, max_people: int = 4):
        self.model_path = Path(model_path)
        self.max_people = max_people

    def extract(self, video_path: Path) -> list[FramePoses]:
        if not self.model_path.is_file():
            raise CoachError(
                "MODEL_MISSING",
                "找不到姿势模型，请将 pose_landmarker.task 放到 models 目录。",
            )
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise CoachError(
                "VIDEO_UNREADABLE",
                "视频打不开或编码不支持，请换成 MP4（H.264）。",
            )
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        landmarker = self._create_landmarker(use_gpu=True)
        frames: list[FramePoses] = []
        index = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                timestamp_ms = int(round(index * 1000.0 / fps))
                people = self._detect(landmarker, rgb, timestamp_ms)
                frames.append(
                    FramePoses(
                        frame_index=index,
                        timestamp_s=index / fps,
                        people=people,
                    )
                )
                index += 1
        finally:
            cap.release()
            closer = getattr(landmarker, "close", None)
            if callable(closer):
                closer()
        return frames

    def _create_landmarker(self, use_gpu: bool):
        try:
            return self._make(use_gpu=use_gpu)
        except Exception:
            if use_gpu:
                return self._make(use_gpu=False)
            raise

    def _make(self, use_gpu: bool):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        delegate = (
            BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU
        )
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(self.model_path),
                delegate=delegate,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=self.max_people,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def _detect(self, landmarker, rgb: np.ndarray, timestamp_ms: int) -> list[PersonPose]:
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(image, timestamp_ms)
        people: list[PersonPose] = []
        if not result.pose_landmarks:
            return people
        worlds = result.pose_world_landmarks or [None] * len(result.pose_landmarks)
        for pid, (lms, world) in enumerate(zip(result.pose_landmarks, worlds)):
            landmarks = []
            for i, lm in enumerate(lms):
                z = 0.0
                if world is not None and i < len(world):
                    z = world[i].z
                vis = getattr(lm, "visibility", 1.0) or 0.0
                landmarks.append(Landmark(lm.x, lm.y, z, vis))
            people.append(PersonPose(person_id=pid, landmarks=landmarks))
        return people
