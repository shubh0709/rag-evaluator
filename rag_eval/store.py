import json
from pathlib import Path
from typing import cast

from .core import Run


class RunStore:
    def __init__(self, root: str | Path = "artifacts/runs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, run: Run) -> Path:
        if run.get("schema_version") != 2:
            raise ValueError("Expected run schema version 2")
        run_id = run.get("id")
        if not isinstance(run_id, str) or not run_id.startswith("run-") or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for char in run_id
        ):
            raise ValueError("Invalid run ID")
        if not isinstance(run.get("results"), list) or not isinstance(run.get("summary"), dict):
            raise ValueError("Run results and summary are required")
        target = self.root / f"{run['id']}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(run, indent=2), encoding="utf-8")
        temporary.replace(target)
        return target

    def list(self) -> list[Run]:
        runs: list[Run] = []
        for path in sorted(self.root.glob("run-*.json"), reverse=True):
            try:
                runs.append(cast(Run, json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError):
                continue
        return runs
