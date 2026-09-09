from __future__ import annotations

from pathlib import Path
from typing import Protocol

import yaml

from badminton_coach.types import MetricScore


class AdviceGenerator(Protocol):
    def generate(self, findings: list[str], metrics: list[MetricScore]) -> list[str]: ...


class TemplateAdviceGenerator:
    def __init__(self, path: Path):
        with path.open(encoding="utf-8") as f:
            self._templates: dict[str, str] = yaml.safe_load(f) or {}

    def generate(self, findings: list[str], metrics: list[MetricScore]) -> list[str]:
        lines: list[str] = []
        for key in findings:
            text = self._templates.get(key)
            if text:
                lines.append(text)
            if len(lines) >= 5:
                break
        return lines
