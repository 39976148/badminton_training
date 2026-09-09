# 姿势模型

从 Google 存储下载 Lite 模型（GTX 1070 更稳），保存为本目录下的 `pose_landmarker.task`：

https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

```powershell
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile "models\pose_landmarker.task"
```

该 `.task` 文件已被 git 忽略。
