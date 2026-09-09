from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from badminton_coach.errors import CoachError
from badminton_coach.pipeline import delete_stroke, resegment, rescore_stroke, retry, run_job
from badminton_coach.pose.mediapipe_backend import MediaPipePoseEstimator
from badminton_coach.storage import JobStore
from badminton_coach.types import STROKE_TYPES

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "config" / "rules"
ADVICE = ROOT / "config" / "advice.yaml"
STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 1_000_000_000
MODEL = ROOT / "models" / "pose_landmarker.task"


def _estimator_default():
    return MediaPipePoseEstimator(MODEL)


def create_app(store: JobStore | None = None, estimator=None) -> FastAPI:
    app = FastAPI(title="羽毛球姿势教练")
    app.state.store = store or JobStore(ROOT / "data" / "jobs")
    app.state.estimator = estimator

    def estimator_for():
        return app.state.estimator or _estimator_default()

    @app.post("/jobs")
    async def create_job(
        background: BackgroundTasks,
        video: UploadFile = File(...),
        demo: UploadFile | None = File(None),
        stroke_type: str = Form("forehand_clear"),
        handedness: str = Form("right"),
        camera: str = Form("side"),
    ):
        del camera
        if stroke_type not in STROKE_TYPES:
            raise HTTPException(400, "不支持的动作类型")
        if handedness not in ("right", "left"):
            raise HTTPException(400, "持拍手只能是 right 或 left")
        video_bytes = await video.read()
        if len(video_bytes) > MAX_UPLOAD:
            raise HTTPException(400, "视频超过 1GB，请压缩后再传")
        demo_bytes = None
        if demo is not None and demo.filename:
            demo_bytes = await demo.read()
        if demo_bytes is not None and len(demo_bytes) > MAX_UPLOAD:
            raise HTTPException(400, "示范视频超过 1GB")
        job_id = app.state.store.create_job(
            video_bytes, stroke_type, handedness, demo_bytes
        )
        background.add_task(
            run_job, app.state.store, job_id, estimator_for(), RULES, ADVICE
        )
        return {"job_id": job_id, "status": "queued"}

    @app.get("/jobs")
    def list_jobs():
        return app.state.store.list_jobs()

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            job = app.state.store.load_job(job_id)
        except FileNotFoundError:
            raise HTTPException(404, "任务不存在")
        strokes = [asdict(s) for s in app.state.store.load_strokes(job_id)]
        job["strokes"] = strokes
        return job

    @app.get("/jobs/{job_id}/video")
    def get_video(job_id: str):
        path = app.state.store.job_dir(job_id) / "input.mp4"
        if not path.exists():
            raise HTTPException(404, "视频不存在")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/jobs/{job_id}/demo-video")
    def get_demo(job_id: str):
        path = app.state.store.job_dir(job_id) / "demo.mp4"
        if not path.exists():
            raise HTTPException(404, "没有示范视频")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/jobs/{job_id}/poses")
    def get_poses(job_id: str, from_frame: int = 0, to_frame: int = 10**9):
        frames = app.state.store.load_poses(job_id)
        sliced = [asdict(f) for f in frames if from_frame <= f.frame_index <= to_frame]
        return sliced

    @app.post("/jobs/{job_id}/person")
    def change_person(job_id: str, payload: dict):
        try:
            resegment(
                app.state.store,
                job_id,
                int(payload["person_id"]),
                RULES,
                ADVICE,
            )
        except CoachError as exc:
            raise HTTPException(400, exc.user_message)
        return get_job(job_id)

    @app.post("/jobs/{job_id}/strokes/{stroke_id}/type")
    def change_type(job_id: str, stroke_id: str, payload: dict):
        stroke_type = payload.get("stroke_type")
        if stroke_type not in STROKE_TYPES:
            raise HTTPException(400, "不支持的动作类型")
        rescore_stroke(
            app.state.store, job_id, stroke_id, stroke_type, RULES, ADVICE
        )
        return get_job(job_id)

    @app.delete("/jobs/{job_id}/strokes/{stroke_id}")
    def remove_stroke(job_id: str, stroke_id: str):
        delete_stroke(app.state.store, job_id, stroke_id)
        return get_job(job_id)

    @app.post("/jobs/{job_id}/retry")
    def retry_job(job_id: str, background: BackgroundTasks):
        background.add_task(
            retry, app.state.store, job_id, estimator_for(), RULES, ADVICE
        )
        return {"job_id": job_id, "status": "queued"}

    STATIC.mkdir(exist_ok=True)
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


app = create_app()
