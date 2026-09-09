from pathlib import Path

from fastapi.testclient import TestClient

from badminton_coach.landmarks import LM
from badminton_coach.pose.fake import FakePoseEstimator
from badminton_coach.storage import JobStore
from badminton_coach.types import FramePoses, Landmark, PersonPose
from web.app import create_app


def _frames():
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


def _client(tmp_path: Path) -> TestClient:
    app = create_app(
        store=JobStore(tmp_path),
        estimator=FakePoseEstimator(_frames()),
    )
    return TestClient(app)


def test_create_and_get_job(tmp_path: Path):
    client = _client(tmp_path)
    res = client.post(
        "/jobs",
        data={"stroke_type": "forehand_clear", "handedness": "right"},
        files={"video": ("a.mp4", b"fake-bytes", "video/mp4")},
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    got = client.get(f"/jobs/{job_id}")
    assert got.status_code == 200
    body = got.json()
    assert body["status"] == "done"
    assert len(body["strokes"]) == 2


def test_change_type_and_delete(tmp_path: Path):
    client = _client(tmp_path)
    job_id = client.post(
        "/jobs",
        data={"stroke_type": "forehand_clear"},
        files={"video": ("a.mp4", b"x", "video/mp4")},
    ).json()["job_id"]
    sid = client.get(f"/jobs/{job_id}").json()["strokes"][0]["stroke"]["stroke_id"]
    ch = client.post(f"/jobs/{job_id}/strokes/{sid}/type", json={"stroke_type": "drop"})
    assert ch.status_code == 200
    assert client.get(f"/jobs/{job_id}").json()["strokes"][0]["stroke"]["stroke_type"] == "drop"
    deleted = client.delete(f"/jobs/{job_id}/strokes/{sid}")
    assert deleted.status_code == 200
    assert len(client.get(f"/jobs/{job_id}").json()["strokes"]) == 1


def test_demo_video_missing_is_404_but_job_ok(tmp_path: Path):
    client = _client(tmp_path)
    job_id = client.post(
        "/jobs",
        data={"stroke_type": "smash"},
        files={"video": ("a.mp4", b"x", "video/mp4")},
    ).json()["job_id"]
    assert client.get(f"/jobs/{job_id}").json()["status"] == "done"
    assert client.get(f"/jobs/{job_id}/demo-video").status_code == 404
