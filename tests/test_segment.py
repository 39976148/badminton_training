from badminton_coach.landmarks import LM
from badminton_coach.segment import segment_strokes
from badminton_coach.types import FramePoses, Landmark, PersonPose


def _frames_with_wrist_x(xs, fps=30.0):
    out = []
    for i, x in enumerate(xs):
        lm = [Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
        lm[LM.RIGHT_WRIST] = Landmark(x, 0.4, 0.0, 1.0)
        out.append(FramePoses(i, i / fps, [PersonPose(0, lm)]))
    return out


def test_two_well_separated_peaks_become_two_strokes():
    xs = [0.2] * 30 + [0.6, 0.2] + [0.2] * 40 + [0.7, 0.2] + [0.2] * 20
    strokes = segment_strokes(_frames_with_wrist_x(xs), 0, "right", "forehand_clear")
    assert len(strokes) == 2
    assert strokes[0].impact_s < strokes[1].impact_s
    assert strokes[0].end_s <= strokes[1].start_s + 1e-6


def test_peaks_closer_than_0_7s_merge_to_higher():
    xs = [0.2] * 10 + [0.5] + [0.2] * 8 + [0.9] + [0.2] * 20
    strokes = segment_strokes(_frames_with_wrist_x(xs), 0, "right", "smash")
    assert len(strokes) == 1
    assert strokes[0].stroke_type == "smash"
