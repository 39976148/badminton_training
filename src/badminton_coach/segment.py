from __future__ import annotations

import uuid

import numpy as np
from scipy.signal import find_peaks

from badminton_coach.kinematics import smooth, wrist_speed
from badminton_coach.types import FramePoses, Handedness, Stroke, StrokeType

MIN_GAP_S = 0.7
PEAK_RATIO = 0.35
PRE_S = 0.45
POST_S = 0.55
MIN_LEN_S = 0.8
MAX_LEN_S = 1.5


def segment_strokes(
    frames: list[FramePoses],
    person_id: int,
    handedness: Handedness,
    stroke_type: StrokeType,
) -> list[Stroke]:
    if not frames:
        return []
    speed = smooth(wrist_speed(frames, person_id, handedness), window=5)
    times = np.array([f.timestamp_s for f in frames])
    duration = max(times[-1] - times[0], 1e-6)
    fps = (len(frames) - 1) / duration if len(frames) > 1 else 30.0
    min_distance = max(1, int(MIN_GAP_S * fps))
    max_speed = float(np.max(speed))
    height = max_speed * PEAK_RATIO if max_speed > 0 else 1e9
    peaks, _ = find_peaks(speed, height=height, distance=min_distance)
    strokes: list[Stroke] = []
    for peak in peaks:
        t = times[peak]
        start_t = t - PRE_S
        end_t = t + POST_S
        start_i = int(np.searchsorted(times, start_t, side="left"))
        end_i = int(np.searchsorted(times, end_t, side="right") - 1)
        start_i = max(0, start_i)
        end_i = min(len(frames) - 1, max(start_i + 1, end_i))
        length = times[end_i] - times[start_i]
        if length < MIN_LEN_S:
            extra = (MIN_LEN_S - length) / 2
            start_i = int(np.searchsorted(times, times[start_i] - extra, side="left"))
            end_i = int(np.searchsorted(times, times[end_i] + extra, side="right") - 1)
            start_i = max(0, start_i)
            end_i = min(len(frames) - 1, end_i)
        if times[end_i] - times[start_i] > MAX_LEN_S:
            start_i = int(np.searchsorted(times, t - MAX_LEN_S * 0.45, side="left"))
            end_i = int(np.searchsorted(times, t + MAX_LEN_S * 0.55, side="right") - 1)
            start_i = max(0, start_i)
            end_i = min(len(frames) - 1, end_i)
        conf = float(speed[peak] / max_speed) if max_speed > 0 else 0.0
        strokes.append(
            Stroke(
                stroke_id=str(uuid.uuid4())[:8],
                person_id=person_id,
                start_frame=start_i,
                impact_frame=int(peak),
                end_frame=end_i,
                start_s=float(times[start_i]),
                impact_s=float(times[peak]),
                end_s=float(times[end_i]),
                stroke_type=stroke_type,
                segment_confidence=conf,
                handedness=handedness,
            )
        )
    return strokes
