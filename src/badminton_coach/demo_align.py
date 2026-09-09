from __future__ import annotations

import numpy as np

from badminton_coach.kinematics import elbow_angle
from badminton_coach.types import DemoDiff, FramePoses, Stroke


def pick_template_stroke(strokes: list[Stroke]) -> Stroke | None:
    if not strokes:
        return None
    return max(strokes, key=lambda s: s.segment_confidence)


def _dtw_path(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = abs(float(a[i - 1] - b[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    i, j = n, m
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        steps = (
            (cost[i - 1, j - 1], i - 1, j - 1),
            (cost[i - 1, j], i - 1, j),
            (cost[i, j - 1], i, j - 1),
        )
        _, i, j = min(steps, key=lambda t: t[0])
    path.reverse()
    return path


def _elbow_series(
    frames: list[FramePoses], stroke: Stroke
) -> np.ndarray:
    values = []
    for fr in frames[stroke.start_frame : stroke.end_frame + 1]:
        person = next((p for p in fr.people if p.person_id == stroke.person_id), None)
        if person is None:
            values.append(0.0)
        else:
            values.append(elbow_angle(person, stroke.handedness))
    return np.asarray(values, dtype=float)


def compare_to_demo(
    practice_frames: list[FramePoses],
    practice_stroke: Stroke,
    demo_frames: list[FramePoses],
    demo_stroke: Stroke,
) -> list[DemoDiff]:
    if not practice_frames or not demo_frames:
        return []
    a = _elbow_series(practice_frames, practice_stroke)
    b = _elbow_series(demo_frames, demo_stroke)
    if len(a) == 0 or len(b) == 0:
        return []
    path = _dtw_path(a, b)
    impact_local = practice_stroke.impact_frame - practice_stroke.start_frame
    impact_local = min(max(impact_local, 0), len(a) - 1)
    paired = [pb for pa, pb in path if pa == impact_local]
    demo_idx = paired[0] if paired else min(impact_local, len(b) - 1)
    pv = float(a[impact_local])
    dv = float(b[demo_idx])
    return [
        DemoDiff(
            metric="elbow_at_impact",
            practice_value=pv,
            demo_value=dv,
            delta=pv - dv,
        )
    ]
