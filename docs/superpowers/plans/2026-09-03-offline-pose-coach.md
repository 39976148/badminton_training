# 本机离线姿势教练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在本机网页上传侧面后场练习视频，自动切击球、按规则评分并给出中文模板建议，可选示范视频骨架对比。

**Architecture:** FastAPI 提供异步分析任务与静态页；分析库 `badminton_coach` 内姿态/选人/切割/规则/建议/DTW 均为可替换单元。骨架只抽一次，改选人/改类型读缓存重算。前端按帧关键点画骨架。

**Tech Stack:** Python 3.11、FastAPI、Uvicorn、MediaPipe Pose Landmarker、OpenCV、NumPy、SciPy、PyYAML、pytest、原生 HTML/JS（无 Node）。

## Global Constraints

- Python 版本地板：3.11（本机同时有 3.12；虚拟环境必须 3.11，因 MediaPipe 更稳）。
- 第一期机位只实现 `side`；动作类型只实现 `forehand_clear` / `smash` / `drop` / `backhand_clear`。
- 用户可见文案必须是中文；日志可以英文。
- 依赖安装优先 `https://pypi.tuna.tsinghua.edu.cn/simple`。
- 视频与 MediaPipe 模型不进 git：`data/jobs/`、`models/*.task` 已在 `.gitignore`。
- 工作目录：`D:\cursor_work\badminton_training`。提交前若 `git config user.name` 仍为空，停止提交并告知用户自行设置身份（不要由代理执行 `git config`）。
- 实现必须 TDD：先写失败测试，再写最少量实现代码。

## File Structure

- `pyproject.toml` — 包元数据、依赖、pytest 配置
- `src/badminton_coach/types.py` — Landmark / FramePoses / Stroke / 评分等数据结构
- `src/badminton_coach/landmarks.py` — MediaPipe 33 点索引常量
- `src/badminton_coach/kinematics.py` — 关节角、手腕速度
- `src/badminton_coach/segment.py` — 击球切割
- `src/badminton_coach/person_select.py` — 默认选人
- `src/badminton_coach/rules.py` — YAML 规则打分
- `src/badminton_coach/advice.py` — 模板建议（`AdviceGenerator` 协议）
- `src/badminton_coach/demo_align.py` — DTW 关节角对比
- `src/badminton_coach/pose/protocol.py` — `PoseEstimator` 协议
- `src/badminton_coach/pose/fake.py` — 测试用假后端
- `src/badminton_coach/pose/mediapipe_backend.py` — 真后端
- `src/badminton_coach/storage.py` — `data/jobs/<id>/` 读写
- `src/badminton_coach/pipeline.py` — 任务状态机
- `src/badminton_coach/errors.py` — 可映射到中文的错误码
- `config/rules/*.yaml` — 四种动作阈值
- `config/advice.yaml` — 建议模板
- `web/app.py` — FastAPI
- `web/static/index.html` `app.js` `style.css` — 两页单显示器 UI
- `tests/` — 与上面对应

---

### Task 1: 工程骨架与核心类型

**Files:**
- Create: `pyproject.toml`
- Create: `src/badminton_coach/__init__.py`
- Create: `src/badminton_coach/types.py`
- Create: `src/badminton_coach/landmarks.py`
- Create: `src/badminton_coach/errors.py`
- Create: `tests/test_types.py`
- Create: `README.md`

**Interfaces:**
- Consumes: 无
- Produces: `Landmark`, `PersonPose`, `FramePoses`, `Stroke`, `MetricScore`, `StrokeScore`, `DemoDiff`, `JobStatus`, `CoachError`, `LM` 点位常量，`STROKE_TYPES`，`HANDEDNESS`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_types.py
from badminton_coach.errors import CoachError
from badminton_coach.landmarks import LM
from badminton_coach.types import (
    HANDEDNESS,
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `py -3.11 -m pytest tests/test_types.py -v`  
Expected: FAIL，提示找不到 `badminton_coach` 或模块不存在。

- [ ] **Step 3: 最小实现**

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "badminton-coach"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "scipy>=1.11",
  "pyyaml>=6.0",
  "opencv-python-headless>=4.8",
  "mediapipe>=0.10.14",
  "fastapi>=0.110",
  "uvicorn[standard]>=0.27",
  "python-multipart>=0.0.9",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

`src/badminton_coach/landmarks.py`:

```python
class LM:
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LANDMARK_COUNT = 33
```

`src/badminton_coach/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

STROKE_TYPES = ("forehand_clear", "smash", "drop", "backhand_clear")
StrokeType = Literal["forehand_clear", "smash", "drop", "backhand_clear"]
HANDEDNESS = ("right", "left")
Handedness = Literal["right", "left"]
JobStatus = Literal[
    "queued", "extracting_pose", "segmenting", "scoring", "done", "error"
]


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class PersonPose:
    person_id: int
    landmarks: list[Landmark]


@dataclass
class FramePoses:
    frame_index: int
    timestamp_s: float
    people: list[PersonPose]


@dataclass
class Stroke:
    stroke_id: str
    person_id: int
    start_frame: int
    impact_frame: int
    end_frame: int
    start_s: float
    impact_s: float
    end_s: float
    stroke_type: StrokeType
    segment_confidence: float
    handedness: Handedness


@dataclass
class MetricScore:
    name: str
    value: float
    score: float
    target_low: float
    target_high: float
    passed: bool


@dataclass
class StrokeScore:
    total: float
    metrics: list[MetricScore]
    findings: list[str]
    advice: list[str]


@dataclass
class DemoDiff:
    metric: str
    practice_value: float
    demo_value: float
    delta: float


@dataclass
class ScoredStroke:
    stroke: Stroke
    score: StrokeScore
    demo_diffs: list[DemoDiff] = field(default_factory=list)
```

`src/badminton_coach/errors.py`:

```python
class CoachError(Exception):
    def __init__(self, code: str, user_message: str):
        super().__init__(code)
        self.code = code
        self.user_message = user_message
```

`src/badminton_coach/__init__.py` 可为空。

`README.md` 写清：Python 3.11、清华镜像创建 venv、`pip install -e ".[dev]"`、`uvicorn web.app:app --reload`、模型放到 `models/`。

- [ ] **Step 4: 建 venv 并跑测试**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
.\.venv\Scripts\python -m pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
.\.venv\Scripts\python -m pytest tests/test_types.py -v
```

Expected: PASS。若 MediaPipe 安装失败，先不装 mediapipe，把该依赖从 `pyproject.toml` 挪到可选 extra `pose`，保证 Task 1–6 的纯数值测试能跑；Task 7 再装。

- [ ] **Step 5: Commit**

```powershell
git add pyproject.toml src/badminton_coach tests/test_types.py README.md
git commit -m "feat: add core types and project scaffold"
```

若因缺少 git user 失败：停下提交，继续后续任务，最后再补提交。

---

### Task 2: 运动学（角度与手腕速度）

**Files:**
- Create: `src/badminton_coach/kinematics.py`
- Create: `tests/test_kinematics.py`

**Interfaces:**
- Consumes: `Landmark`, `PersonPose`, `LM`
- Produces:
  - `angle_deg(a, b, c) -> float` 点 b 处夹角
  - `elbow_angle(pose, handedness) -> float`
  - `knee_angle(pose, handedness) -> float`
  - `trunk_tilt_deg(pose) -> float` 肩中点相对髋中点相对图像竖直方向的前倾角，后仰为正
  - `contact_height_ratio(pose, handedness) -> float` `(1-wrist.y) - (1-shoulder.y)`，越大表示手腕相对肩膀越高（图像 y 向下）
  - `wrist_speed(frames, person_id, handedness) -> np.ndarray` 长度 = 帧数，单位：归一化坐标 / 秒
  - `smooth(values, window=5) -> np.ndarray`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_kinematics.py
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.\.venv\Scripts\python -m pytest tests/test_kinematics.py -v`  
Expected: FAIL `ModuleNotFoundError: kinematics`

- [ ] **Step 3: 最小实现**

```python
# src/badminton_coach/kinematics.py
from __future__ import annotations

import math
import numpy as np
from badminton_coach.landmarks import LM
from badminton_coach.types import FramePoses, Landmark, PersonPose, Handedness


def _vec(p: Landmark) -> np.ndarray:
    return np.array([p.x, p.y, p.z], dtype=float)


def angle_deg(a: Landmark, b: Landmark, c: Landmark) -> float:
    ba = _vec(a) - _vec(b)
    bc = _vec(c) - _vec(b)
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _arm_ids(handedness: Handedness) -> tuple[int, int, int]:
    if handedness == "right":
        return LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST
    return LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST


def _leg_ids(handedness: Handedness) -> tuple[int, int, int]:
    if handedness == "right":
        return LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE
    return LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE


def elbow_angle(pose: PersonPose, handedness: Handedness) -> float:
    s, e, w = _arm_ids(handedness)
    return angle_deg(pose.landmarks[s], pose.landmarks[e], pose.landmarks[w])


def knee_angle(pose: PersonPose, handedness: Handedness) -> float:
    h, k, a = _leg_ids(handedness)
    return angle_deg(pose.landmarks[h], pose.landmarks[k], pose.landmarks[a])


def trunk_tilt_deg(pose: PersonPose) -> float:
    ls, rs = pose.landmarks[LM.LEFT_SHOULDER], pose.landmarks[LM.RIGHT_SHOULDER]
    lh, rh = pose.landmarks[LM.LEFT_HIP], pose.landmarks[LM.RIGHT_HIP]
    shoulder = np.array([(ls.x + rs.x) / 2, (ls.y + rs.y) / 2])
    hip = np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2])
    vec = shoulder - hip
    # image y down: vec.y < 0 means shoulders above hips (upright).
    # tilt: deviation from vertical (0, -1). positive = lean back (shoulders +x or -x? use y).
    vertical = np.array([0.0, -1.0])
    n = np.linalg.norm(vec)
    if n < 1e-9:
        return 0.0
    vec = vec / n
    angle = math.degrees(math.atan2(vec[0], -vec[1]))
    return angle


def contact_height_ratio(pose: PersonPose, handedness: Handedness) -> float:
    _, _, w = _arm_ids(handedness)
    s, _, _ = _arm_ids(handedness)
    return pose.landmarks[s].y - pose.landmarks[w].y


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, window)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def wrist_speed(
    frames: list[FramePoses], person_id: int, handedness: Handedness
) -> np.ndarray:
    _, _, wid = _arm_ids(handedness)
    pts = []
    times = []
    for fr in frames:
        person = next((p for p in fr.people if p.person_id == person_id), None)
        if person is None:
            pts.append(np.array([np.nan, np.nan]))
        else:
            w = person.landmarks[wid]
            pts.append(np.array([w.x, w.y]))
        times.append(fr.timestamp_s)
    pts = np.vstack(pts)
    times = np.asarray(times)
    speed = np.zeros(len(frames))
    for i in range(1, len(frames)):
        dt = times[i] - times[i - 1]
        if dt <= 0 or np.any(np.isnan(pts[i])) or np.any(np.isnan(pts[i - 1])):
            speed[i] = 0.0
        else:
            speed[i] = float(np.linalg.norm(pts[i] - pts[i - 1]) / dt)
    return speed
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.\.venv\Scripts\python -m pytest tests/test_kinematics.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```powershell
git add src/badminton_coach/kinematics.py tests/test_kinematics.py
git commit -m "feat: add joint angles and wrist speed"
```

---

### Task 3: 击球切割

**Files:**
- Create: `src/badminton_coach/segment.py`
- Create: `tests/test_segment.py`

**Interfaces:**
- Consumes: `wrist_speed`, `smooth`, `FramePoses`, `Stroke`
- Produces: `segment_strokes(frames, person_id, handedness, stroke_type, fps) -> list[Stroke]`
  - 峰值间隔 < 0.7s 只留更高者
  - 过低峰丢弃（低于全局最大峰的 35%）
  - 窗：impact 前 0.45s 到后 0.55s，再用速度谷收缩，夹在 0.8–1.5s
  - `segment_confidence`：该峰 / 最大峰，夹到 `[0,1]`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_segment.py
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.\.venv\Scripts\python -m pytest tests/test_segment.py -v`  
Expected: FAIL 找不到 `segment`

- [ ] **Step 3: 最小实现**

```python
# src/badminton_coach/segment.py
from __future__ import annotations

import uuid
import numpy as np
from scipy.signal import find_peaks
from badminton_coach.kinematics import smooth, wrist_speed
from badminton_coach.types import FramePoses, Handedness, Stroke, StrokeType

MIN_GAP_S = 0.7
PEAK_RATIO = 0.35
PRE_S = 0.45
POST_S = 0.55
MIN_LEN_S = 0.8
MAX_LEN_S = 1.5


def segment_strokes(
    frames: list[FramePoses],
    person_id: int,
    handedness: Handedness,
    stroke_type: StrokeType,
) -> list[Stroke]:
    if not frames:
        return []
    speed = smooth(wrist_speed(frames, person_id, handedness), window=5)
    times = np.array([f.timestamp_s for f in frames])
    duration = max(times[-1] - times[0], 1e-6)
    fps = (len(frames) - 1) / duration if len(frames) > 1 else 30.0
    min_distance = max(1, int(MIN_GAP_S * fps))
    height = float(np.max(speed) * PEAK_RATIO) if np.max(speed) > 0 else 1e9
    peaks, _ = find_peaks(speed, height=height, distance=min_distance)
    strokes: list[Stroke] = []
    for peak in peaks:
        t = times[peak]
        start_t = t - PRE_S
        end_t = t + POST_S
        start_i = int(np.searchsorted(times, start_t, side="left"))
        end_i = int(np.searchsorted(times, end_t, side="right") - 1)
        start_i = max(0, start_i)
        end_i = min(len(frames) - 1, max(start_i + 1, end_i))
        length = times[end_i] - times[start_i]
        if length < MIN_LEN_S:
            extra = (MIN_LEN_S - length) / 2
            start_i = int(np.searchsorted(times, times[start_i] - extra, side="left"))
            end_i = int(np.searchsorted(times, times[end_i] + extra, side="right") - 1)
            start_i = max(0, start_i)
            end_i = min(len(frames) - 1, end_i)
        if times[end_i] - times[start_i] > MAX_LEN_S:
            start_i = int(np.searchsorted(times, t - MAX_LEN_S * 0.45, side="left"))
            end_i = int(np.searchsorted(times, t + MAX_LEN_S * 0.55, side="right") - 1)
            start_i = max(0, start_i)
            end_i = min(len(frames) - 1, end_i)
        conf = float(speed[peak] / np.max(speed)) if np.max(speed) > 0 else 0.0
        strokes.append(
            Stroke(
                stroke_id=str(uuid.uuid4())[:8],
                person_id=person_id,
                start_frame=start_i,
                impact_frame=int(peak),
                end_frame=end_i,
                start_s=float(times[start_i]),
                impact_s=float(times[peak]),
                end_s=float(times[end_i]),
                stroke_type=stroke_type,
                segment_confidence=conf,
                handedness=handedness,
            )
        )
    return strokes
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.\.venv\Scripts\python -m pytest tests/test_segment.py -v`  
Expected: PASS。若两次峰值测例因平滑把峰糊掉，加大测试里手腕位移或略降 `PEAK_RATIO`，以测试意图为准（两次分离峰 → 两击）。

- [ ] **Step 5: Commit**

```powershell
git add src/badminton_coach/segment.py tests/test_segment.py
git commit -m "feat: segment strokes from wrist-speed peaks"
```

---

### Task 4: 规则引擎与中文模板建议

**Files:**
- Create: `config/rules/forehand_clear.yaml`
- Create: `config/rules/smash.yaml`
- Create: `config/rules/drop.yaml`
- Create: `config/rules/backhand_clear.yaml`
- Create: `config/advice.yaml`
- Create: `src/badminton_coach/rules.py`
- Create: `src/badminton_coach/advice.py`
- Create: `tests/test_rules.py`

**Interfaces:**
- Consumes: `Stroke`, `FramePoses`, kinematics
- Produces:
  - `score_stroke(frames, stroke, rules_dir) -> StrokeScore`
  - `TemplateAdviceGenerator.generate(findings, metrics) -> list[str]` 最多 5 条
  - 指标名固定：`elbow_at_impact`, `knee_at_prep`, `contact_height`, `wrist_peak_speed`, `chain_lag`（肩峰值帧早于肘、肘早于腕为好，好则高分）

每种 YAML 结构：

```yaml
camera: side
metrics:
  elbow_at_impact:
    target: [140, 180]
    weight: 1.0
    finding: elbow_not_extended
  knee_at_prep:
    target: [120, 165]
    weight: 0.8
    finding: no_leg_drive
  contact_height:
    target: [0.02, 0.25]
    weight: 1.0
    finding: contact_too_low
  wrist_peak_speed:
    target: [1.2, 8.0]
    weight: 0.7
    finding: swing_too_slow
  chain_lag:
    target: [0.5, 1.01]
    weight: 0.8
    finding: arm_only
```

`drop.yaml` 的 `wrist_peak_speed` target 更低，如 `[0.4, 2.0]`，超高触发 `swing_too_fast`。  
`smash.yaml` 的 `wrist_peak_speed` 下限更高，如 `[1.8, 10]`。  
`backhand_clear.yaml` 的 `contact_height` 可略低，`elbow_at_impact` 仍要打开。

`config/advice.yaml`:

```yaml
elbow_not_extended: "击球时肘关节伸展不足，建议在击球瞬间把肘打开，再送拍。"
no_leg_drive: "下肢参与不够，建议先屈膝蹬地，再带动手臂。"
contact_too_low: "击球点偏低，建议在头侧上方击球，而不是在胸前够球。"
swing_too_slow: "挥拍速度偏慢，建议完整引拍后加速，避免只靠手臂轻推。"
swing_too_fast: "这一拍更像抽或杀，吊球应缩短随挥、降低击球段手腕速度。"
arm_only: "发力顺序偏手臂先行，建议蹬地转体后再甩臂。"
```

线性分数：值在区间内 100；每超出区间宽度 100% 扣到 0。加权平均为 `total`。未通过的 finding 按权重从高到低最多 5 条交给建议器。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_rules.py
from pathlib import Path
from badminton_coach.advice import TemplateAdviceGenerator
from badminton_coach.landmarks import LM
from badminton_coach.rules import score_stroke
from badminton_coach.types import FramePoses, Landmark, PersonPose, Stroke

ROOT = Path(__file__).resolve().parents[1] / "config" / "rules"


def _pose(elbow=170, wrist_y=0.30, shoulder_y=0.40, knee=150):
    lm = [Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
    lm[LM.RIGHT_SHOULDER] = Landmark(0.4, shoulder_y, 0.0, 1.0)
    lm[LM.RIGHT_ELBOW] = Landmark(0.4, shoulder_y + 0.1, 0.0, 1.0)
    # place wrist so elbow angle ~ elbow degrees in 2D: keep shoulder-elbow vertical, wrist offset
    import math
    rad = math.radians(180 - elbow)
    lm[LM.RIGHT_WRIST] = Landmark(0.4 + 0.1 * math.sin(rad), shoulder_y + 0.1 + 0.1 * math.cos(rad), 0.0, 1.0)
    lm[LM.RIGHT_HIP] = Landmark(0.45, 0.6, 0.0, 1.0)
    lm[LM.RIGHT_KNEE] = Landmark(0.45, 0.75, 0.0, 1.0)
    lm[LM.RIGHT_ANKLE] = Landmark(0.45, 0.9, 0.0, 1.0)
    lm[LM.LEFT_SHOULDER] = Landmark(0.5, shoulder_y, 0.0, 1.0)
    lm[LM.LEFT_HIP] = Landmark(0.5, 0.6, 0.0, 1.0)
    return PersonPose(0, lm)


def test_extended_clear_scores_higher_than_bent_elbow():
    fps = 30.0
    good_frames, bad_frames = [], []
    for i in range(40):
        t = i / fps
        good_frames.append(FramePoses(i, t, [_pose(elbow=175, wrist_y=0.25)]))
        bad_frames.append(FramePoses(i, t, [_pose(elbow=90, wrist_y=0.55)]))
    stroke = Stroke("a", 0, 0, 20, 39, 0, 20 / fps, 39 / fps, "forehand_clear", 1.0, "right")
    good = score_stroke(good_frames, stroke, ROOT)
    bad = score_stroke(bad_frames, stroke, ROOT)
    assert good.total > bad.total
    assert any("肘" in a for a in bad.advice)
```

测试里若几何构造导致肘角不准，改为直接单测 `_metric_score` 或在测试中断言 `bad.metrics` 里 `elbow_at_impact.score < good`。以实现后能稳定复现为准：坏样本必须打出含「肘」的建议。

- [ ] **Step 2: 跑测试确认失败**

Run: `.\.venv\Scripts\python -m pytest tests/test_rules.py -v`  
Expected: FAIL 找不到 `rules`

- [ ] **Step 3: 实现 `band_score`、加载 YAML、`score_stroke`、`TemplateAdviceGenerator`**

`advice.py` 定义：

```python
class AdviceGenerator(Protocol):
    def generate(self, findings: list[str], metrics: list[MetricScore]) -> list[str]: ...
```

`TemplateAdviceGenerator.__init__(self, path: Path)` 读 `advice.yaml`。找不到 finding 键则跳过。

- [ ] **Step 4: 跑测试确认通过**

Run: `.\.venv\Scripts\python -m pytest tests/test_rules.py tests/test_segment.py tests/test_kinematics.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```powershell
git add config src/badminton_coach/rules.py src/badminton_coach/advice.py tests/test_rules.py
git commit -m "feat: score rear-court strokes with YAML rules and Chinese advice"
```

---

### Task 5: 选人

**Files:**
- Create: `src/badminton_coach/person_select.py`
- Create: `tests/test_person_select.py`

**Interfaces:**
- Produces: `select_person_id(frames, handedness) -> int`
  - 对每个 `person_id` 计算：平均包围盒面积（肩髋腕）× 持拍手最大手腕速度
  - 返回乘积最大者；无人则抛 `CoachError("NO_PERSON", "全程检测不到人，请保证人完整入画、光线足够。")`

- [ ] **Step 1: 写失败测试**

```python
from badminton_coach.errors import CoachError
from badminton_coach.landmarks import LM
from badminton_coach.person_select import select_person_id
from badminton_coach.types import FramePoses, Landmark, PersonPose
import pytest

def _person(pid, scale, wrist_x_series):
    frames = []
    for i, wx in enumerate(wrist_x_series):
        lm = [Landmark(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
        lm[LM.RIGHT_SHOULDER] = Landmark(0.4, 0.4 * scale, 0, 1)
        lm[LM.RIGHT_HIP] = Landmark(0.4, 0.7 * scale, 0, 1)
        lm[LM.RIGHT_WRIST] = Landmark(wx, 0.3, 0, 1)
        frames.append((i, pid, lm))
    return frames

def test_picks_the_person_with_faster_racket_arm():
    frames = []
    for i in range(20):
        slow = [Landmark(0.2, 0.5, 0, 1) for _ in range(33)]
        slow[LM.RIGHT_WRIST] = Landmark(0.2, 0.5, 0, 1)
        fast = [Landmark(0.8, 0.5, 0, 1) for _ in range(33)]
        fast[LM.RIGHT_WRIST] = Landmark(0.5 + (0.3 if i == 10 else 0.0), 0.3, 0, 1)
        frames.append(FramePoses(i, i / 30, [PersonPose(0, slow), PersonPose(1, fast)]))
    assert select_person_id(frames, "right") == 1

def test_no_person_raises_chinese_error():
    with pytest.raises(CoachError) as ei:
        select_person_id([], "right")
    assert ei.value.code == "NO_PERSON"
```

- [ ] **Step 2–4:** 失败 → 实现 → PASS  
- [ ] **Step 5: Commit** `feat: select hitter by size and wrist speed`

---

### Task 6: 示范 DTW 对比

**Files:**
- Create: `src/badminton_coach/demo_align.py`
- Create: `tests/test_demo_align.py`

**Interfaces:**
- Produces: `pick_template_stroke(strokes) -> Stroke | None`（置信度最高）
- Produces: `compare_to_demo(practice_frames, practice_stroke, demo_frames, demo_stroke) -> list[DemoDiff]`
  - 两边在各自 `[start_frame, end_frame]` 上取 `elbow_angle` 序列
  - SciPy/`numpy` 实现简易 DTW，在对准后的击球时刻附近比较肘角，输出一条 `DemoDiff(metric="elbow_at_impact", ...)`
  - 示范 0 次击球：返回 `[]`，不抛错

- [ ] **Step 1:** 测试：同一骨架序列对比时 `abs(delta) < 1`；空示范返回 `[]`  
- [ ] **Step 2–4:** 失败 → 实现 → PASS  
- [ ] **Step 5: Commit** `feat: align practice strokes to demo via DTW`

---

### Task 7: 姿态后端协议 + Fake + MediaPipe

**Files:**
- Create: `src/badminton_coach/pose/__init__.py`
- Create: `src/badminton_coach/pose/protocol.py`
- Create: `src/badminton_coach/pose/fake.py`
- Create: `src/badminton_coach/pose/mediapipe_backend.py`
- Create: `tests/test_pose_fake.py`
- Create: `models/README.md`

**Interfaces:**
- Produces:

```python
class PoseEstimator(Protocol):
    def extract(self, video_path: Path) -> list[FramePoses]: ...
```

`FakePoseEstimator(frames: list[FramePoses])` 忽略视频路径，返回注入的帧（供 pipeline 测试）。

`MediaPipePoseEstimator(model_path: Path, max_people: int = 4)`：OpenCV 读视频；模型不存在抛 `CoachError("MODEL_MISSING", "找不到姿势模型，请将 pose_landmarker.task 放到 models 目录。")`；解码失败抛 `CoachError("VIDEO_UNREADABLE", "视频打不开或编码不支持，请换成 MP4（H.264）。")`；优先 GPU，失败则 CPU 再试一次。每帧最多 4 人。`person_id` 为该帧检测索引（第一期不做跨帧 ID 跟踪；选人按整段聚合，改选人用整段里出现过的 id）。

`models/README.md` 写下载地址：

`https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task`

保存为 `models/pose_landmarker.task`。第一期用 lite，1070 上更稳。

- [ ] **Step 1:** `tests/test_pose_fake.py` 断言 Fake 原样返回；`MediaPipePoseEstimator` 在缺失模型时抛 `MODEL_MISSING`（不要求本任务跑通真视频）。  
- [ ] **Step 2–4:** 失败 → 实现 → PASS  
- [ ] **Step 5: Commit** `feat: add pose estimator protocol and mediapipe backend`

---

### Task 8: 任务存储与 pipeline

**Files:**
- Create: `src/badminton_coach/storage.py`
- Create: `src/badminton_coach/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- `JobStore(root: Path)`：`create_job` 生成 id、拷贝 `input.mp4`（及可选 `demo.mp4`）、写 `job.json`
- `job.json` 字段：`status`, `stroke_type`, `handedness`, `camera=side`, `person_id`, `error_code`, `error_message`
- `save_poses` / `load_poses`：jsonl，每行一个 `FramePoses`（用 `asdict`）
- `save_strokes` / `load_strokes`：`strokes.json` 存 `list[ScoredStroke]`
- `run_job(job_id, estimator, rules_dir, advice_yaml)` 状态机：
  1. `extracting_pose`：若 `poses.jsonl` 已存在则跳过
  2. 若全程无人 → `error` / `NO_PERSON`
  3. `segmenting`：`select_person_id`（若 job 已有 person_id 则尊重）→ `segment_strokes`
  4. `scoring`：逐击 `score_stroke`；若有 demo poses 则切示范、取模板、写 `demo_diffs`
  5. `done`；0 次击球仍为 `done`，strokes 空列表
  6. 示范失败只记 `demo_error` 字段，主流程继续
- `resegment(job_id, person_id)`、`rescore_stroke(job_id, stroke_id, stroke_type)`、`delete_stroke(job_id, stroke_id)`、`retry(job_id)`

测试用 `FakePoseEstimator` + 临时目录，不读真实 mp4（`create_job` 可写一个空文件当 input）。

- [ ] **Step 1:** 测试：假骨架两峰 → done 且 2 击且有 advice；改类型不调用 estimator；空帧 → NO_PERSON；缺 demo 仍 done  
- [ ] **Step 2–4:** 失败 → 实现 → PASS  
- [ ] **Step 5: Commit** `feat: add job storage and analysis pipeline`

---

### Task 9: FastAPI

**Files:**
- Create: `web/app.py`
- Create: `web/__init__.py`
- Create: `tests/test_api.py`

**Interfaces:**
- `POST /jobs` multipart: `video`, 可选 `demo`, `stroke_type`, `handedness`（默认 `right`），`camera` 忽略并强制 `side`
- 立即 200 `{job_id, status: queued}`，`BackgroundTasks` 调 `run_job`
- `GET /jobs` 列表
- `GET /jobs/{id}` 状态 + scored strokes（无整段 poses）
- `GET /jobs/{id}/video` `FileResponse` input.mp4
- `GET /jobs/{id}/demo-video` 无则 404
- `GET /jobs/{id}/poses?from_frame&to_frame`
- `POST /jobs/{id}/person` JSON `{person_id}`
- `POST /jobs/{id}/strokes/{stroke_id}/type` JSON `{stroke_type}`
- `DELETE /jobs/{id}/strokes/{stroke_id}`
- `POST /jobs/{id}/retry`
- 静态：`web/static` 挂在 `/`
- 错误：`CoachError` → 400，`user_message` 中文

测试用 `TestClient` + Fake estimator 依赖注入（`app.dependency_overrides` 或 `web.app` 里 `get_estimator()` 可替换）。不要在 API 测试里跑 MediaPipe。

- [ ] **Step 1–4:** TDD 覆盖创建、查状态、改类型、删除、示范缺失仍 200  
- [ ] **Step 5: Commit** `feat: add local FastAPI job endpoints`

---

### Task 10: 单显示器两页前端

**Files:**
- Create: `web/static/index.html`
- Create: `web/static/app.js`
- Create: `web/static/style.css`

**Interfaces:**
- 页面 1：上半新建（文件、类型、持拍手、机位只读「侧面」、可选示范），下半 `GET /jobs` 历史
- 页面 2：`#/jobs/{id}` 左列表、中 `<video>` + canvas 骨架、右分数建议；低置信度 `<0.5` 标记；改类型 POST；删误切 DELETE；多人下拉改 person_id；有示范则并排第二个 video，按 `impact_s` 对齐当前击；轮询直到 `done`/`error`；error 显示 `error_message`；重试按钮打 `POST retry`
- 骨架：当前 `video.currentTime` 对应帧，从 `/poses?from_frame&to_frame` 取附近帧画 33 点及肩-肘-腕、髋-膝-踝连线

无前端单测。手工验收见 spec 第 11 节。实现后用浏览器打开 `http://127.0.0.1:8000`：无视频时至少能看到页面 1、上传控件、历史空列表。

- [ ] **Step 1:** 实现三文件，保证 `GET /` 返回 html  
- [ ] **Step 2:** `.\.venv\Scripts\python -m uvicorn web.app:app --host 127.0.0.1 --port 8000` 能启动  
- [ ] **Step 3: Commit** `feat: add local two-page coaching UI`

---

### Task 11: 全量测试与运行说明

**Files:**
- Modify: `README.md`
- Create: `models/README.md`（若 Task 7 已写则只补命令）

- [ ] **Step 1:** `.\.venv\Scripts\python -m pytest -v` 全绿  
- [ ] **Step 2:** README 含：建 venv、清华镜像、下载 lite 模型、`uvicorn web.app:app --host 127.0.0.1 --port 8000`、用侧面后场视频验收清单（与 spec §11 一致）  
- [ ] **Step 3: Commit** `docs: document how to run the local coach`

---

## Spec coverage（自审）

| Spec 条目 | 任务 |
|-----------|------|
| 整段视频自动切击球 | Task 3, 8 |
| 四类后场 + 整段类型 + 单条改正 | Task 4, 8, 9, 10 |
| 规则 + 可选示范 DTW | Task 4, 6, 8 |
| 多人自动选 + 手选 | Task 5, 8, 9, 10 |
| 中文模板建议，预留 AdviceGenerator | Task 4 |
| 异步任务与骨架缓存 | Task 8, 9 |
| 机位 side only | Task 8, 9, 10 |
| 失败中文、示范失败不影响主评分、CPU 回退、retry | Task 7, 8, 9 |
| poses 按帧范围、前端画骨架 | Task 9, 10 |
| 1GB 建议限制 | Task 9 上传校验 1_000_000_000 字节 |
| 自动测试 | Tasks 1–9 |
| 不做实时/球/App | 无对应任务 |

类型名全局一致：`StrokeType` 四码、`JobStatus` 六态、`CoachError.code` 使用 `NO_PERSON` / `VIDEO_UNREADABLE` / `MODEL_MISSING`。
