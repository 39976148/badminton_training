# 本机羽毛球姿势教练（第一期）

侧面后场练习视频：自动切击球、规则评分、中文模板建议。可选示范视频骨架对比。

## 环境

- Windows，Python **3.11**
- 本仓库 `.venv` 已按 3.11 创建时可直接用

```powershell
cd D:\cursor_work\badminton_training
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
.\.venv\Scripts\python -m pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 姿势模型

下载 Lite 模型到 `models/pose_landmarker.task`（见 `models/README.md`）：

```powershell
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile "models\pose_landmarker.task"
```

## 运行

```powershell
.\.venv\Scripts\python -m uvicorn web.app:app --host 127.0.0.1 --port 8000
```

浏览器打开 http://127.0.0.1:8000 （一块显示器即可：先上传页，再结果页）。

上传侧面后场练习视频，选择动作类型（正手高远 / 杀 / 吊 / 反手高远）和持拍手。可附示范视频。

## 测试

```powershell
.\.venv\Scripts\python -m pytest -v
```

## 验收（有真实视频时）

- 上传整段练习，进度走到完成
- 列表出现多次击球；点开能跳转、看到骨架、分数和中文建议
- 改某一次类型后分数更新，不必重抽骨架
- 多人画面可改选人
- 无示范或示范损坏时规则结果仍在
- 关掉浏览器再打开同一任务，结果还在
