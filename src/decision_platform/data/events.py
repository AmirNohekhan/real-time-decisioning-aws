import json
from pathlib import Path
from threading import Lock

from decision_platform.contracts import InteractionEvent


class LocalEventStore:
    """Append-only JSONL event sink with process-local idempotency."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._seen: set[str] = set()
        self._lock = Lock()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    self._seen.add(str(json.loads(line)["event_id"]))

    def put(self, event: InteractionEvent) -> bool:
        key = str(event.event_id)
        with self._lock:
            if key in self._seen:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")
            self._seen.add(key)
            return True
