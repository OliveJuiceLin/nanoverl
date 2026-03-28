"""Lightweight rollout and validation dumps for experiment debugging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from nanoverl.core.batch import RLBatch


def build_batch_preview_rows(
    batch: RLBatch,
    max_rows: int,
    reward_scores: Sequence[Sequence[float]] | None = None,
    reward_extras: Mapping[str, Sequence[object]] | None = None,
) -> List[Dict[str, Any]]:
    # This function is new in Phase 2 because metrics alone are not enough to debug
    # reward functions or rollout behavior. We keep one small, readable row preview
    # instead of adding a large metric surface or full dataset dumps.
    preview_rows: List[Dict[str, Any]] = []
    prompt_texts = batch.non_tensor.get("prompt_text") or batch.non_tensor.get("prompt") or []
    response_texts = batch.non_tensor.get("response_text") or []
    token_level_scores = reward_scores if reward_scores is not None else batch.batch.get("token_level_scores", [])
    extra_columns = reward_extras if reward_extras is not None else {
        key: values
        for key, values in batch.non_tensor.items()
        if key not in {"prompt", "prompt_text", "response_text", "uid", "data_source", "rollout_index"}
    }

    for row_index in range(min(len(batch), max_rows)):
        row: Dict[str, Any] = {
            "uid": batch.non_tensor.get("uid", [None] * len(batch))[row_index],
            "data_source": batch.non_tensor.get("data_source", ["unknown"] * len(batch))[row_index],
            "rollout_index": batch.non_tensor.get("rollout_index", [0] * len(batch))[row_index],
            "prompt_text": prompt_texts[row_index] if row_index < len(prompt_texts) else "",
            "response_text": response_texts[row_index] if row_index < len(response_texts) else "",
            "prompt_length": len(batch.batch.get("prompts", [])[row_index]) if "prompts" in batch.batch else 0,
            "response_length": sum(batch.batch.get("response_mask", [])[row_index]) if "response_mask" in batch.batch else 0,
        }
        if row_index < len(token_level_scores):
            row["reward_score"] = float(sum(token_level_scores[row_index]))
        for key, values in extra_columns.items():
            if row_index < len(values):
                row[key] = values[row_index]
        preview_rows.append(row)
    return preview_rows


class ArtifactWriter:
    """Writes small JSON snapshots for the batches a researcher usually inspects first."""

    def __init__(self, root_dir: str | Path, experiment_name: str):
        self.root_dir = Path(root_dir) / "artifacts" / experiment_name
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def write_train_preview(self, global_step: int, rows: Sequence[Mapping[str, Any]]) -> Path:
        # This method is new in Phase 2 because Phase 1 had metrics and checkpoints,
        # but no easy way to inspect the actual rollout rows that produced them.
        return self._write_payload(
            "train_step_%06d.json" % global_step,
            {"kind": "train", "global_step": global_step, "rows": list(rows)},
        )

    def write_validation_preview(
        self,
        global_step: int,
        metrics: Mapping[str, float],
        rows: Sequence[Mapping[str, Any]],
    ) -> Path:
        # This method is new in Phase 2 because validation needed one compact artifact
        # that ties summary metrics back to concrete prompt/response examples.
        return self._write_payload(
            "validation_step_%06d.json" % global_step,
            {"kind": "validation", "global_step": global_step, "metrics": dict(metrics), "rows": list(rows)},
        )

    def _write_payload(self, filename: str, payload: Mapping[str, Any]) -> Path:
        path = self.root_dir / filename
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


__all__ = ["ArtifactWriter", "build_batch_preview_rows"]
