from __future__ import annotations

import math

import numpy as np

from badminton_coach.landmarks import LM
from badminton_coach.types import FramePoses, Handedness, Landmark, PersonPose


def _vec(p: Landmark) -> np.ndarray:
    return np.array([p.x, p.y, p.z], dtype=float)


def angle_deg(a: Landmark, b: Landmark, c: Landmark) -> float:
    ba = _vec(a) - _vec(b)
    bc = _vec(c) - _vec(b)
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _arm_ids(handedness: Handedness) -> tuple[int, int, int]:
    if handedness == "right":
        return LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST
    return LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST


def _leg_ids(handedness: Handedness) -> tuple[int, int, int]:
    if handedness == "right":
        return LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE
    return LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE


def elbow_angle(pose: PersonPose, handedness: Handedness) -> float:
    s, e, w = _arm_ids(handedness)
    return angle_deg(pose.landmarks[s], pose.landmarks[e], pose.landmarks[w])


def knee_angle(pose: PersonPose, handedness: Handedness) -> float:
    h, k, a = _leg_ids(handedness)
    return angle_deg(pose.landmarks[h], pose.landmarks[k], pose.landmarks[a])


def trunk_tilt_deg(pose: PersonPose) -> float:
    ls, rs = pose.landmarks[LM.LEFT_SHOULDER], pose.landmarks[LM.RIGHT_SHOULDER]
    lh, rh = pose.landmarks[LM.LEFT_HIP], pose.landmarks[LM.RIGHT_HIP]
    shoulder = np.array([(ls.x + rs.x) / 2, (ls.y + rs.y) / 2])
    hip = np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2])
    vec = shoulder - hip
    n = np.linalg.norm(vec)
    if n < 1e-9:
        return 0.0
    vec = vec / n
    return math.degrees(math.atan2(vec[0], -vec[1]))


def contact_height_ratio(pose: PersonPose, handedness: Handedness) -> float:
    s, _, w = _arm_ids(handedness)
    return pose.landmarks[s].y - pose.landmarks[w].y


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, window)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def wrist_speed(
    frames: list[FramePoses], person_id: int, handedness: Handedness
) -> np.ndarray:
    _, _, wid = _arm_ids(handedness)
    pts = []
    times = []
    for fr in frames:
        person = next((p for p in fr.people if p.person_id == person_id), None)
        if person is None:
            pts.append(np.array([np.nan, np.nan]))
        else:
            w = person.landmarks[wid]
            pts.append(np.array([w.x, w.y]))
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
