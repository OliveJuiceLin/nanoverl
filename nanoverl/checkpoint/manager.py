"""Local filesystem checkpoint manager."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional


TRACKER_FILE = "latest_checkpointed_iteration.txt"


def find_latest_checkpoint(root_dir: str | Path) -> Optional[Path]:
    root = Path(root_dir)
    tracker = root / TRACKER_FILE
    if not tracker.exists():
        return None
    step = tracker.read_text(encoding="utf-8").strip()
    if not step:
        return None
    checkpoint_dir = root / ("global_step_%s" % step)
    if not checkpoint_dir.exists():
        return None
    return checkpoint_dir


class CheckpointManager:
    """Saves trainer state and checkpointable component state as a single pickle payload."""

    def __init__(self, root_dir: str | Path, max_to_keep: Optional[int] = None):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep

    def save(self, global_step: int, payload: Dict[str, Any]) -> Path:
        checkpoint_dir = self.root_dir / ("global_step_%s" % global_step)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state_path = checkpoint_dir / "state.pkl"
        with state_path.open("wb") as file_obj:
            pickle.dump(payload, file_obj)
        (self.root_dir / TRACKER_FILE).write_text(str(global_step), encoding="utf-8")
        self._enforce_retention()
        return checkpoint_dir

    def load_latest(self) -> Optional[Dict[str, Any]]:
        checkpoint_dir = find_latest_checkpoint(self.root_dir)
        if checkpoint_dir is None:
            return None
        return self.load(checkpoint_dir)

    def load(self, checkpoint_dir: str | Path) -> Dict[str, Any]:
        state_path = Path(checkpoint_dir) / "state.pkl"
        with state_path.open("rb") as file_obj:
            return pickle.load(file_obj)

    def _enforce_retention(self) -> None:
        if not self.max_to_keep or self.max_to_keep <= 0:
            return
        checkpoints = sorted(
            (path for path in self.root_dir.iterdir() if path.is_dir() and path.name.startswith("global_step_")),
            key=lambda path: int(path.name.rsplit("_", 1)[-1]),
        )
        while len(checkpoints) > self.max_to_keep:
            stale = checkpoints.pop(0)
            for child in stale.glob("**/*"):
                if child.is_file():
                    child.unlink()
            for child in sorted(stale.glob("**/*"), reverse=True):
                if child.is_dir():
                    child.rmdir()
            stale.rmdir()


__all__ = ["CheckpointManager", "find_latest_checkpoint"]
