from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from badminton_coach.advice import TemplateAdviceGenerator
from badminton_coach.kinematics import (
    contact_height_ratio,
    elbow_angle,
    knee_angle,
    smooth,
    wrist_speed,
)
from badminton_coach.landmarks import LM
from badminton_coach.types import (
    FramePoses,
    MetricScore,
    PersonPose,
    Stroke,
    StrokeScore,
)

_ADVICE_PATH = Path(__file__).resolve().parents[2] / "config" / "advice.yaml"


def band_score(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 100.0
    width = max(high - low, 1e-6)
    if value < low:
        dist = (low - value) / width
    else:
        dist = (value - high) / width
    return float(max(0.0, 100.0 * (1.0 - dist)))


def _person_at(frame: FramePoses, person_id: int) -> PersonPose | None:
    return next((p for p in frame.people if p.person_id == person_id), None)


def _landmark_speed(
    frames: list[FramePoses], person_id: int, index: int
) -> np.ndarray:
    pts = []
    times = []
    for fr in frames:
        person = _person_at(fr, person_id)
        if person is None:
            pts.append(np.array([np.nan, np.nan]))
        else:
            p = person.landmarks[index]
            pts.append(np.array([p.x, p.y]))
        times.append(fr.timestamp_s)
    pts = np.vstack(pts)
    times = np.asarray(times)
    speed = np.zeros(len(frames))
    for i in range(1, len(frames)):
        dt = times[i] - times[i - 1]
        if dt <= 0 or np.any(np.isnan(pts[i])) or np.any(np.isnan(pts[i - 1])):
            speed[i] = 0.0
        else:
            speed[i] = float(np.linalg.norm(pts[i] - pts[i - 1]) / dt)
    return speed


def _chain_lag(window: list[FramePoses], person_id: int, handedness: str) -> float:
    shoulder = LM.RIGHT_SHOULDER if handedness == "right" else LM.LEFT_SHOULDER
    elbow = LM.RIGHT_ELBOW if handedness == "right" else LM.LEFT_ELBOW
    s = smooth(_landmark_speed(window, person_id, shoulder), 5)
    e = smooth(_landmark_speed(window, person_id, elbow), 5)
    w = smooth(wrist_speed(window, person_id, handedness), 5)  # type: ignore[arg-type]
    if float(np.max(s) + np.max(e) + np.max(w)) < 1e-6:
        return 1.0
    order = (int(np.argmax(s)), int(np.argmax(e)), int(np.argmax(w)))
    return 1.0 if order[0] <= order[1] <= order[2] else 0.0


def _load_rule(rules_dir: Path, stroke_type: str) -> dict:
    path = rules_dir / f"{stroke_type}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def score_stroke(
    frames: list[FramePoses],
    stroke: Stroke,
    rules_dir: Path,
    advice_path: Path | None = None,
) -> StrokeScore:
    spec = _load_rule(rules_dir, stroke.stroke_type)
    impact = frames[stroke.impact_frame]
    prep = frames[stroke.start_frame]
    pose_i = _person_at(impact, stroke.person_id)
    pose_p = _person_at(prep, stroke.person_id)
    if pose_i is None or pose_p is None:
        return StrokeScore(total=0.0, metrics=[], findings=[], advice=[])

    window = frames[stroke.start_frame : stroke.end_frame + 1]
    speeds = wrist_speed(window, stroke.person_id, stroke.handedness)
    values = {
        "elbow_at_impact": elbow_angle(pose_i, stroke.handedness),
        "knee_at_prep": knee_angle(pose_p, stroke.handedness),
        "contact_height": contact_height_ratio(pose_i, stroke.handedness),
        "wrist_peak_speed": float(np.max(speeds)) if len(speeds) else 0.0,
        "chain_lag": _chain_lag(window, stroke.person_id, stroke.handedness),
    }

    metrics: list[MetricScore] = []
    weighted = 0.0
    weight_sum = 0.0
    pending: list[tuple[float, str]] = []
    for name, cfg in spec["metrics"].items():
        low, high = cfg["target"]
        value = values[name]
        score = band_score(value, low, high)
        passed = low <= value <= high
        if cfg.get("invert_high") and value > high:
            pending.append((cfg["weight"], cfg["finding"]))
            passed = False
        elif not passed:
            pending.append((cfg["weight"], cfg["finding"]))
        metrics.append(
            MetricScore(name, float(value), score, float(low), float(high), passed)
        )
        weighted += score * cfg["weight"]
        weight_sum += cfg["weight"]

    total = weighted / weight_sum if weight_sum else 0.0
    pending.sort(key=lambda x: x[0], reverse=True)
    findings = [f for _, f in pending[:5]]
    generator = TemplateAdviceGenerator(advice_path or _ADVICE_PATH)
    advice = generator.generate(findings, metrics)
    return StrokeScore(total=float(total), metrics=metrics, findings=findings, advice=advice)
