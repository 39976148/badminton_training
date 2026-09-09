from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

STROKE_TYPES = ("forehand_clear", "smash", "drop", "backhand_clear")
StrokeType = Literal["forehand_clear", "smash", "drop", "backhand_clear"]
HANDEDNESS = ("right", "left")
Handedness = Literal["right", "left"]
JobStatus = Literal[
    "queued", "extracting_pose", "segmenting", "scoring", "done", "error"
]


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class PersonPose:
    person_id: int
    landmarks: list[Landmark]


@dataclass
class FramePoses:
    frame_index: int
    timestamp_s: float
    people: list[PersonPose]


@dataclass
class Stroke:
    stroke_id: str
    person_id: int
    start_frame: int
    impact_frame: int
    end_frame: int
    start_s: float
    impact_s: float
    end_s: float
    stroke_type: StrokeType
    segment_confidence: float
    handedness: Handedness


@dataclass
class MetricScore:
    name: str
    value: float
    score: float
    target_low: float
    target_high: float
    passed: bool


@dataclass
class StrokeScore:
    total: float
    metrics: list[MetricScore]
    findings: list[str]
    advice: list[str]


@dataclass
class DemoDiff:
    metric: str
    practice_value: float
    demo_value: float
    delta: float


@dataclass
class ScoredStroke:
    stroke: Stroke
    score: StrokeScore
    demo_diffs: list[DemoDiff] = field(default_factory=list)
