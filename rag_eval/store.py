"""Provide store utilities and tests."""

import json
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from .core import Run
from .dto import SavedRunInput, validation_issue


class RunStore:
    """Represent run store data."""

    def __init__(self, root: str | Path = "artifacts/runs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, run: Run) -> Path:
        """Save."""
        try:
            SavedRunInput.model_validate(run)
        except ValidationError as exc:
            if any(error["loc"] == ("id",) for error in exc.errors()):
                raise ValueError("Invalid run ID") from exc
            raise ValueError(f"Invalid run: {validation_issue(exc)}") from exc
        target = self.root / f"{run['id']}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(run, indent=2), encoding="utf-8")
        temporary.replace(target)
        return target

    def list(self) -> list[Run]:
        """List."""
        runs: list[Run] = []
        for path in sorted(self.root.glob("run-*.json"), reverse=True):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                SavedRunInput.model_validate(raw)
                runs.append(cast(Run, raw))
            except (OSError, json.JSONDecodeError, ValidationError):
                continue
        return runs
