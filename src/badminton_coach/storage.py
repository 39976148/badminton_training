from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path

from badminton_coach.types import (
    DemoDiff,
    FramePoses,
    Landmark,
    MetricScore,
    PersonPose,
    ScoredStroke,
    Stroke,
    StrokeScore,
)


def _landmark(d: dict) -> Landmark:
    return Landmark(**d)


def _person(d: dict) -> PersonPose:
    return PersonPose(
        person_id=d["person_id"],
        landmarks=[_landmark(x) for x in d["landmarks"]],
    )


def _frame(d: dict) -> FramePoses:
    return FramePoses(
        frame_index=d["frame_index"],
        timestamp_s=d["timestamp_s"],
        people=[_person(p) for p in d["people"]],
    )


def _stroke(d: dict) -> Stroke:
    return Stroke(**d)


def _score(d: dict) -> StrokeScore:
    return StrokeScore(
        total=d["total"],
        metrics=[MetricScore(**m) for m in d["metrics"]],
        findings=d["findings"],
        advice=d["advice"],
    )


def _scored(d: dict) -> ScoredStroke:
    return ScoredStroke(
        stroke=_stroke(d["stroke"]),
        score=_score(d["score"]),
        demo_diffs=[DemoDiff(**x) for x in d.get("demo_diffs", [])],
    )


class JobStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def create_job(
        self,
        video_bytes: bytes,
        stroke_type: str,
        handedness: str,
        demo_bytes: bytes | None = None,
    ) -> str:
        job_id = uuid.uuid4().hex[:12]
        d = self.job_dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "input.mp4").write_bytes(video_bytes)
        if demo_bytes is not None:
            (d / "demo.mp4").write_bytes(demo_bytes)
        job = {
            "job_id": job_id,
            "status": "queued",
            "stroke_type": stroke_type,
            "handedness": handedness,
            "camera": "side",
            "person_id": None,
            "error_code": None,
            "error_message": None,
            "demo_error": None,
        }
        self.save_job(job_id, job)
        return job_id

    def save_job(self, job_id: str, job: dict) -> None:
        path = self.job_dir(job_id) / "job.json"
        path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_job(self, job_id: str) -> dict:
        path = self.job_dir(job_id) / "job.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def list_jobs(self) -> list[dict]:
        if not self.root.exists():
            return []
        jobs = []
        for p in sorted(self.root.iterdir()):
            if (p / "job.json").exists():
                jobs.append(self.load_job(p.name))
        return jobs

    def save_poses(self, job_id: str, frames: list[FramePoses], name: str = "poses.jsonl") -> None:
        path = self.job_dir(job_id) / name
        with path.open("w", encoding="utf-8") as f:
            for fr in frames:
                f.write(json.dumps(asdict(fr), ensure_ascii=False) + "\n")

    def load_poses(self, job_id: str, name: str = "poses.jsonl") -> list[FramePoses]:
        path = self.job_dir(job_id) / name
        if not path.exists():
            return []
        frames = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    frames.append(_frame(json.loads(line)))
        return frames

    def poses_exist(self, job_id: str) -> bool:
        return (self.job_dir(job_id) / "poses.jsonl").exists()

    def save_strokes(self, job_id: str, strokes: list[ScoredStroke]) -> None:
        payload = [asdict(s) for s in strokes]
        path = self.job_dir(job_id) / "strokes.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_strokes(self, job_id: str) -> list[ScoredStroke]:
        path = self.job_dir(job_id) / "strokes.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [_scored(x) for x in data]
