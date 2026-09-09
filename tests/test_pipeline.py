from pathlib import Path

from badminton_coach.landmarks import LM
from badminton_coach.pipeline import JobStore, delete_stroke, rescore_stroke, run_job
from badminton_coach.pose.fake import FakePoseEstimator
from badminton_coach.types import FramePoses, Landmark, PersonPose

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "config" / "rules"
ADVICE = ROOT / "config" / "advice.yaml"


def _frames_two_peaks():
    xs = [0.2] * 30 + [0.6, 0.2] + [0.2] * 40 + [0.7, 0.2] + [0.2] * 20
    out = []
    for i, x in enumerate(xs):
        lm = [Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
        lm[LM.RIGHT_SHOULDER] = Landmark(0.40, 0.45, 0.0, 1.0)
        lm[LM.RIGHT_ELBOW] = Landmark(0.40, 0.38, 0.0, 1.0)
        lm[LM.RIGHT_WRIST] = Landmark(x, 0.28, 0.0, 1.0)
        lm[LM.RIGHT_HIP] = Landmark(0.45, 0.70, 0.0, 1.0)
        lm[LM.RIGHT_KNEE] = Landmark(0.45, 0.82, 0.0, 1.0)
        lm[LM.RIGHT_ANKLE] = Landmark(0.45, 0.94, 0.0, 1.0)
        lm[LM.LEFT_SHOULDER] = Landmark(0.50, 0.45, 0.0, 1.0)
        lm[LM.LEFT_HIP] = Landmark(0.50, 0.70, 0.0, 1.0)
        out.append(FramePoses(i, i / 30.0, [PersonPose(0, lm)]))
    return out


def test_run_job_segments_and_scores(tmp_path: Path):
    store = JobStore(tmp_path)
    job_id = store.create_job(
        video_bytes=b"fake",
        stroke_type="forehand_clear",
        handedness="right",
    )
    est = FakePoseEstimator(_frames_two_peaks())
    run_job(store, job_id, est, RULES, ADVICE)
    job = store.load_job(job_id)
    assert job["status"] == "done"
    strokes = store.load_strokes(job_id)
    assert len(strokes) == 2
    assert strokes[0].score.advice is not None
    assert est.extract(tmp_path / "x") is not None  # still callable


def test_rescore_does_not_need_new_pose(tmp_path: Path):
    store = JobStore(tmp_path)
    job_id = store.create_job(b"fake", "forehand_clear", "right")

    class Counting(FakePoseEstimator):
        def __init__(self, frames):
            super().__init__(frames)
            self.calls = 0

        def extract(self, video_path):
            self.calls += 1
            return super().extract(video_path)

    est = Counting(_frames_two_peaks())
    run_job(store, job_id, est, RULES, ADVICE)
    assert est.calls == 1
    strokes = store.load_strokes(job_id)
    rescore_stroke(store, job_id, strokes[0].stroke.stroke_id, "smash", RULES, ADVICE)
    assert est.calls == 1
    updated = store.load_strokes(job_id)
    assert updated[0].stroke.stroke_type == "smash"


def test_empty_pose_is_error(tmp_path: Path):
    store = JobStore(tmp_path)
    job_id = store.create_job(b"fake", "drop", "right")
    run_job(store, job_id, FakePoseEstimator([]), RULES, ADVICE)
    job = store.load_job(job_id)
    assert job["status"] == "error"
    assert job["error_code"] == "NO_PERSON"


def test_missing_demo_still_done(tmp_path: Path):
    store = JobStore(tmp_path)
    job_id = store.create_job(b"fake", "forehand_clear", "right", demo_bytes=None)
    run_job(store, job_id, FakePoseEstimator(_frames_two_peaks()), RULES, ADVICE)
    assert store.load_job(job_id)["status"] == "done"


def test_delete_stroke(tmp_path: Path):
    store = JobStore(tmp_path)
    job_id = store.create_job(b"fake", "forehand_clear", "right")
    run_job(store, job_id, FakePoseEstimator(_frames_two_peaks()), RULES, ADVICE)
    sid = store.load_strokes(job_id)[0].stroke.stroke_id
    delete_stroke(store, job_id, sid)
    assert len(store.load_strokes(job_id)) == 1
