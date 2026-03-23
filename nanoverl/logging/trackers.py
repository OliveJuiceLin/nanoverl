"""Minimal tracker integrations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class Tracker:
    def log(self, data: Dict[str, float], step: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return


class ConsoleTracker(Tracker):
    def log(self, data: Dict[str, float], step: int) -> None:
        print("[step=%s] %s" % (step, json.dumps(data, sort_keys=True)))


class FileTracker(Tracker):
    def __init__(self, project_name: str, experiment_name: str):
        root = Path("logs") / project_name
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / ("%s.jsonl" % experiment_name)
        self._handle = self.path.open("a", encoding="utf-8")

    def log(self, data: Dict[str, float], step: int) -> None:
        self._handle.write(json.dumps({"step": step, "data": data}, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class WandbTracker(Tracker):
    def __init__(self, project_name: str, experiment_name: str, config):
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - only exercised when wandb is installed
            raise RuntimeError("WandB tracking requires wandb to be installed.") from exc
        self._wandb = wandb
        self._wandb.init(project=project_name, name=experiment_name, config=config)

    def log(self, data: Dict[str, float], step: int) -> None:
        self._wandb.log(data, step=step)

    def close(self) -> None:
        self._wandb.finish()


class TrackingManager:
    """Fan-out tracker wrapper."""

    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        backends: Iterable[str],
        config: Optional[dict] = None,
    ):
        self.trackers: List[Tracker] = []
        for backend in backends:
            if backend == "console":
                self.trackers.append(ConsoleTracker())
            elif backend == "file":
                self.trackers.append(FileTracker(project_name, experiment_name))
            elif backend == "wandb":
                self.trackers.append(WandbTracker(project_name, experiment_name, config))
            else:
                raise ValueError("Unsupported tracker backend: %s" % backend)

    def log(self, data: Dict[str, float], step: int) -> None:
        for tracker in self.trackers:
            tracker.log(data, step)

    def close(self) -> None:
        for tracker in self.trackers:
            tracker.close()


__all__ = ["TrackingManager"]
