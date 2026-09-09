import pytest

from badminton_coach.errors import CoachError
from badminton_coach.landmarks import LM
from badminton_coach.person_select import select_person_id
from badminton_coach.types import FramePoses, Landmark, PersonPose


def test_picks_the_person_with_faster_racket_arm():
    frames = []
    for i in range(20):
        slow = [Landmark(0.2, 0.5, 0.0, 1.0) for _ in range(33)]
        slow[LM.RIGHT_SHOULDER] = Landmark(0.18, 0.40, 0.0, 1.0)
        slow[LM.RIGHT_HIP] = Landmark(0.18, 0.70, 0.0, 1.0)
        slow[LM.RIGHT_WRIST] = Landmark(0.20, 0.50, 0.0, 1.0)
        fast = [Landmark(0.8, 0.5, 0.0, 1.0) for _ in range(33)]
        fast[LM.RIGHT_SHOULDER] = Landmark(0.78, 0.40, 0.0, 1.0)
        fast[LM.RIGHT_HIP] = Landmark(0.78, 0.70, 0.0, 1.0)
        fast[LM.RIGHT_WRIST] = Landmark(0.50 + (0.30 if i == 10 else 0.0), 0.30, 0.0, 1.0)
        frames.append(
            FramePoses(i, i / 30.0, [PersonPose(0, slow), PersonPose(1, fast)])
        )
    assert select_person_id(frames, "right") == 1


def test_no_person_raises_chinese_error():
    with pytest.raises(CoachError) as ei:
        select_person_id([], "right")
    assert ei.value.code == "NO_PERSON"
    assert "检测不到人" in ei.value.user_message
