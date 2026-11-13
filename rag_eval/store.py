import json
from pathlib import Path
from typing import Any


class RunStore:
    def __init__(self, root: str | Path = "artifacts/runs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, run: dict[str, Any]) -> Path:
        target = self.root / f"{run['id']}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(run, indent=2), encoding="utf-8")
        temporary.replace(target)
        return target

    def list(self) -> list[dict[str, Any]]:
        runs = []
        for path in sorted(self.root.glob("run-*.json"), reverse=True):
            try:
                runs.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return runs
