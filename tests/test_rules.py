from pathlib import Path

from badminton_coach.advice import TemplateAdviceGenerator
from badminton_coach.landmarks import LM
from badminton_coach.rules import score_stroke
from badminton_coach.types import FramePoses, Landmark, MetricScore, PersonPose, Stroke

ROOT = Path(__file__).resolve().parents[1] / "config" / "rules"
ADVICE = Path(__file__).resolve().parents[1] / "config" / "advice.yaml"


def _blank():
    return [Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(33)]


def _right_arm_pose(shoulder, elbow, wrist, knee_bend="straight"):
    lm = _blank()
    lm[LM.RIGHT_SHOULDER] = Landmark(*shoulder, 0.0, 1.0)
    lm[LM.RIGHT_ELBOW] = Landmark(*elbow, 0.0, 1.0)
    lm[LM.RIGHT_WRIST] = Landmark(*wrist, 0.0, 1.0)
    lm[LM.LEFT_SHOULDER] = Landmark(0.55, shoulder[1], 0.0, 1.0)
    lm[LM.LEFT_HIP] = Landmark(0.5, 0.70, 0.0, 1.0)
    lm[LM.RIGHT_HIP] = Landmark(0.45, 0.70, 0.0, 1.0)
    if knee_bend == "straight":
        lm[LM.RIGHT_KNEE] = Landmark(0.45, 0.82, 0.0, 1.0)
        lm[LM.RIGHT_ANKLE] = Landmark(0.45, 0.94, 0.0, 1.0)
    else:
        lm[LM.RIGHT_KNEE] = Landmark(0.52, 0.80, 0.0, 1.0)
        lm[LM.RIGHT_ANKLE] = Landmark(0.45, 0.94, 0.0, 1.0)
    return PersonPose(0, lm)


def _stroke():
    return Stroke("a", 0, 0, 20, 39, 0.0, 20 / 30.0, 39 / 30.0, "forehand_clear", 1.0, "right")


def _clip(pose: PersonPose, n=40):
    return [FramePoses(i, i / 30.0, [pose]) for i in range(n)]


def test_extended_clear_scores_higher_than_bent_elbow():
    # Vertical arm, elbow between shoulder and wrist (image y grows downward).
    good = _right_arm_pose((0.40, 0.45), (0.40, 0.38), (0.40, 0.28))
    bad = _right_arm_pose((0.40, 0.40), (0.40, 0.50), (0.55, 0.50))
    good_score = score_stroke(_clip(good), _stroke(), ROOT)
    bad_score = score_stroke(_clip(bad), _stroke(), ROOT)
    assert good_score.total > bad_score.total
    assert any("肘" in a for a in bad_score.advice)


def test_template_advice_is_chinese_and_capped():
    gen = TemplateAdviceGenerator(ADVICE)
    metrics = [
        MetricScore("elbow_at_impact", 90, 10, 140, 180, False),
        MetricScore("knee_at_prep", 170, 20, 120, 165, False),
    ]
    lines = gen.generate(["elbow_not_extended", "no_leg_drive", "missing_key"], metrics)
    assert 1 <= len(lines) <= 5
    assert any("肘" in line for line in lines)
    assert all("missing_key" not in line for line in lines)
