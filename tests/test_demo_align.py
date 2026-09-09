from badminton_coach.demo_align import compare_to_demo, pick_template_stroke
from badminton_coach.landmarks import LM
from badminton_coach.types import FramePoses, Landmark, PersonPose, Stroke


def _pose(elbow_y: float) -> PersonPose:
    lm = [Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
    lm[LM.RIGHT_SHOULDER] = Landmark(0.40, 0.45, 0.0, 1.0)
    lm[LM.RIGHT_ELBOW] = Landmark(0.40, elbow_y, 0.0, 1.0)
    lm[LM.RIGHT_WRIST] = Landmark(0.40, 0.20, 0.0, 1.0)
    return PersonPose(0, lm)


def _frames(n=30, elbow_y=0.35):
    return [FramePoses(i, i / 30.0, [_pose(elbow_y)]) for i in range(n)]


def _stroke(sid="d"):
    return Stroke(sid, 0, 0, 10, 29, 0.0, 10 / 30.0, 29 / 30.0, "forehand_clear", 0.9, "right")


def test_identical_sequences_have_small_elbow_delta():
    frames = _frames()
    stroke = _stroke()
    diffs = compare_to_demo(frames, stroke, frames, stroke)
    assert diffs
    assert abs(diffs[0].delta) < 1.0


def test_empty_demo_returns_no_diffs():
    assert compare_to_demo(_frames(), _stroke(), [], _stroke("x")) == []


def test_pick_template_uses_highest_confidence():
    a = _stroke("a")
    a.segment_confidence = 0.4
    b = _stroke("b")
    b.segment_confidence = 0.9
    assert pick_template_stroke([a, b]).stroke_id == "b"
    assert pick_template_stroke([]) is None
