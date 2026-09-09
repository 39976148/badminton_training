import math

from badminton_coach.kinematics import angle_deg, elbow_angle, wrist_speed
from badminton_coach.landmarks import LM
from badminton_coach.types import FramePoses, Landmark, PersonPose


def _blank_landmarks():
    return [Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(33)]


def test_right_angle_is_90_degrees():
    a = Landmark(0.0, 0.0, 0.0, 1.0)
    b = Landmark(0.0, 1.0, 0.0, 1.0)
    c = Landmark(1.0, 1.0, 0.0, 1.0)
    assert abs(angle_deg(a, b, c) - 90.0) < 1e-6


def test_straight_elbow_is_180():
    lm = _blank_landmarks()
    lm[LM.RIGHT_SHOULDER] = Landmark(0.0, 0.0, 0.0, 1.0)
    lm[LM.RIGHT_ELBOW] = Landmark(0.0, 1.0, 0.0, 1.0)
    lm[LM.RIGHT_WRIST] = Landmark(0.0, 2.0, 0.0, 1.0)
    pose = PersonPose(0, lm)
    assert abs(elbow_angle(pose, "right") - 180.0) < 1e-4


def test_wrist_speed_peaks_when_wrist_jumps():
    frames = []
    for i, x in enumerate([0.0, 0.0, 0.0, 0.4, 0.4]):
        lm = _blank_landmarks()
        lm[LM.RIGHT_WRIST] = Landmark(x, 0.5, 0.0, 1.0)
        frames.append(FramePoses(i, i / 30.0, [PersonPose(0, lm)]))
    speed = wrist_speed(frames, person_id=0, handedness="right")
    assert speed.argmax() in (3, 4)
    assert speed[3] > speed[1]
