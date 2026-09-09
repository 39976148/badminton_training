from badminton_coach.errors import CoachError
from badminton_coach.landmarks import LM
from badminton_coach.types import (
    STROKE_TYPES,
    FramePoses,
    Landmark,
    PersonPose,
)


def test_stroke_types_are_the_four_rear_court_codes():
    assert STROKE_TYPES == (
        "forehand_clear",
        "smash",
        "drop",
        "backhand_clear",
    )


def test_landmark_count_for_one_person_is_33():
    people = [
        PersonPose(
            person_id=0,
            landmarks=[Landmark(0.1, 0.2, 0.0, 0.9) for _ in range(33)],
        )
    ]
    frame = FramePoses(frame_index=0, timestamp_s=0.0, people=people)
    assert len(frame.people[0].landmarks) == 33
    assert LM.RIGHT_WRIST == 16
    assert LM.LEFT_WRIST == 15


def test_coach_error_exposes_chinese_message():
    err = CoachError("NO_PERSON", "全程检测不到人，请保证人完整入画、光线足够。")
    assert err.code == "NO_PERSON"
    assert "检测不到人" in err.user_message
