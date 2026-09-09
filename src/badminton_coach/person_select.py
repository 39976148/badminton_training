from __future__ import annotations

import numpy as np

from badminton_coach.errors import CoachError
from badminton_coach.kinematics import wrist_speed
from badminton_coach.landmarks import LM
from badminton_coach.types import FramePoses, Handedness, PersonPose


def _bbox_area(pose: PersonPose) -> float:
    xs = [pose.landmarks[i].x for i in (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP, LM.LEFT_WRIST, LM.RIGHT_WRIST)]
    ys = [pose.landmarks[i].y for i in (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP, LM.LEFT_WRIST, LM.RIGHT_WRIST)]
    return max(max(xs) - min(xs), 1e-6) * max(max(ys) - min(ys), 1e-6)


def select_person_id(frames: list[FramePoses], handedness: Handedness) -> int:
    ids: set[int] = set()
    for fr in frames:
        for p in fr.people:
            ids.add(p.person_id)
    if not ids:
        raise CoachError("NO_PERSON", "全程检测不到人，请保证人完整入画、光线足够。")

    best_id = None
    best_score = -1.0
    for pid in ids:
        areas = []
        for fr in frames:
            person = next((p for p in fr.people if p.person_id == pid), None)
            if person is not None:
                areas.append(_bbox_area(person))
        mean_area = float(np.mean(areas)) if areas else 0.0
        speed = wrist_speed(frames, pid, handedness)
        peak = float(np.max(speed)) if len(speed) else 0.0
        score = mean_area * (peak + 1e-3)
        if score > best_score:
            best_score = score
            best_id = pid
    assert best_id is not None
    return best_id
