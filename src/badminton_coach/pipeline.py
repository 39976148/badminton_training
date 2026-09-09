from __future__ import annotations

from pathlib import Path

from badminton_coach.demo_align import compare_to_demo, pick_template_stroke
from badminton_coach.errors import CoachError
from badminton_coach.person_select import select_person_id
from badminton_coach.pose.protocol import PoseEstimator
from badminton_coach.rules import score_stroke
from badminton_coach.segment import segment_strokes
from badminton_coach.storage import JobStore
from badminton_coach.types import ScoredStroke, StrokeType

__all__ = [
    "JobStore",
    "run_job",
    "rescore_stroke",
    "delete_stroke",
    "resegment",
    "retry",
]


def _set_status(store: JobStore, job_id: str, **fields) -> dict:
    job = store.load_job(job_id)
    job.update(fields)
    store.save_job(job_id, job)
    return job


def _score_all(
    store: JobStore,
    job_id: str,
    frames,
    strokes,
    rules_dir: Path,
    advice_path: Path,
) -> list[ScoredStroke]:
    demo_frames = store.load_poses(job_id, "demo_poses.jsonl")
    demo_template = None
    job = store.load_job(job_id)
    if demo_frames:
        try:
            demo_strokes = segment_strokes(
                demo_frames,
                select_person_id(demo_frames, job["handedness"]),
                job["handedness"],
                job["stroke_type"],
            )
            demo_template = pick_template_stroke(demo_strokes)
        except CoachError as exc:
            job["demo_error"] = exc.user_message
            store.save_job(job_id, job)
    scored: list[ScoredStroke] = []
    for stroke in strokes:
        score = score_stroke(frames, stroke, rules_dir, advice_path)
        diffs = []
        if demo_template is not None and demo_frames:
            diffs = compare_to_demo(frames, stroke, demo_frames, demo_template)
        scored.append(ScoredStroke(stroke=stroke, score=score, demo_diffs=diffs))
    return scored


def run_job(
    store: JobStore,
    job_id: str,
    estimator: PoseEstimator,
    rules_dir: Path,
    advice_path: Path,
) -> None:
    job = store.load_job(job_id)
    try:
        _set_status(store, job_id, status="extracting_pose")
        if store.poses_exist(job_id):
            frames = store.load_poses(job_id)
        else:
            frames = estimator.extract(store.job_dir(job_id) / "input.mp4")
            store.save_poses(job_id, frames)
        if not any(fr.people for fr in frames):
            raise CoachError("NO_PERSON", "全程检测不到人，请保证人完整入画、光线足够。")

        demo_path = store.job_dir(job_id) / "demo.mp4"
        if demo_path.exists() and not (store.job_dir(job_id) / "demo_poses.jsonl").exists():
            try:
                demo_frames = estimator.extract(demo_path)
                store.save_poses(job_id, demo_frames, "demo_poses.jsonl")
            except CoachError as exc:
                job = store.load_job(job_id)
                job["demo_error"] = exc.user_message
                store.save_job(job_id, job)

        _set_status(store, job_id, status="segmenting")
        job = store.load_job(job_id)
        person_id = job.get("person_id")
        if person_id is None:
            person_id = select_person_id(frames, job["handedness"])
            job["person_id"] = person_id
            store.save_job(job_id, job)

        _set_status(store, job_id, status="scoring")
        job = store.load_job(job_id)
        strokes = segment_strokes(
            frames, int(job["person_id"]), job["handedness"], job["stroke_type"]
        )
        scored = _score_all(store, job_id, frames, strokes, rules_dir, advice_path)
        store.save_strokes(job_id, scored)
        _set_status(store, job_id, status="done", error_code=None, error_message=None)
    except CoachError as exc:
        _set_status(
            store,
            job_id,
            status="error",
            error_code=exc.code,
            error_message=exc.user_message,
        )


def rescore_stroke(
    store: JobStore,
    job_id: str,
    stroke_id: str,
    stroke_type: StrokeType,
    rules_dir: Path,
    advice_path: Path,
) -> None:
    frames = store.load_poses(job_id)
    scored = store.load_strokes(job_id)
    for i, item in enumerate(scored):
        if item.stroke.stroke_id == stroke_id:
            item.stroke.stroke_type = stroke_type
            item.score = score_stroke(frames, item.stroke, rules_dir, advice_path)
            demo_frames = store.load_poses(job_id, "demo_poses.jsonl")
            if demo_frames:
                demo_strokes = segment_strokes(
                    demo_frames,
                    item.stroke.person_id,
                    item.stroke.handedness,
                    stroke_type,
                )
                template = pick_template_stroke(demo_strokes)
                item.demo_diffs = (
                    compare_to_demo(frames, item.stroke, demo_frames, template)
                    if template
                    else []
                )
            scored[i] = item
            break
    store.save_strokes(job_id, scored)


def delete_stroke(store: JobStore, job_id: str, stroke_id: str) -> None:
    scored = [s for s in store.load_strokes(job_id) if s.stroke.stroke_id != stroke_id]
    store.save_strokes(job_id, scored)


def resegment(
    store: JobStore,
    job_id: str,
    person_id: int,
    rules_dir: Path,
    advice_path: Path,
) -> None:
    job = store.load_job(job_id)
    job["person_id"] = person_id
    store.save_job(job_id, job)
    frames = store.load_poses(job_id)
    strokes = segment_strokes(frames, person_id, job["handedness"], job["stroke_type"])
    scored = _score_all(store, job_id, frames, strokes, rules_dir, advice_path)
    store.save_strokes(job_id, scored)


def retry(
    store: JobStore,
    job_id: str,
    estimator: PoseEstimator,
    rules_dir: Path,
    advice_path: Path,
) -> None:
    run_job(store, job_id, estimator, rules_dir, advice_path)
