from __future__ import annotations

from typing import Any, Dict, Optional
import json
import os
import sys
from datetime import datetime


class FallbackLogger:
    def __init__(self, session_path: str) -> None:
        self.session_path = session_path
        os.makedirs(self.session_path, exist_ok=True)

    def info(self, **event: Any) -> None:
        try:
            event = {**event, "ts": datetime.utcnow().isoformat() + "Z"}
            with open(os.path.join(self.session_path, "events.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            pass


def get_logger(session_path: str, session_id: str, level: str = "normal", redact: bool = True):
    try:
        import structlog  # type: ignore

        def redact_processor(logger, method_name, event_dict):  # noqa: ANN001
            if redact:
                # Basic redaction hook (extend as needed)
                for k in list(event_dict.keys()):
                    if k.lower() in {"api_key", "authorization"}:
                        event_dict[k] = "[REDACTED]"
            return event_dict

        os.makedirs(session_path, exist_ok=True)
        structlog.configure(
            processors=[
                redact_processor,
                structlog.processors.TimeStamper(fmt="iso", key="ts"),
                structlog.processors.JSONRenderer(ensure_ascii=False),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
            context_class=dict,
            cache_logger_on_first_use=True,
        )
        logger = structlog.get_logger().bind(session_id=session_id)

        class StructlogFileWriter:
            def info(self, **event: Any) -> None:
                # Write to events.jsonl to keep a single artifact
                line = logger.new(**event).msg if hasattr(logger, "new") else json.dumps(event, ensure_ascii=False)
                try:
                    with open(os.path.join(session_path, "events.jsonl"), "a", encoding="utf-8") as f:
                        if isinstance(line, str):
                            f.write(line + "\n")
                        else:
                            f.write(json.dumps(line, ensure_ascii=False) + "\n")
                except Exception:
                    pass

        return StructlogFileWriter()
    except Exception:
        return FallbackLogger(session_path)


