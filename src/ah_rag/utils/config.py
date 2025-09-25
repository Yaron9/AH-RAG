from __future__ import annotations

from typing import Any, Dict
import os

# Auto-load .env at import so all modules can read keys via os.getenv
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()  # search for .env in project root
except Exception:
    pass


def load_config(defaults: Dict[str, Any] | None = None, path: str = "config/ah_rag.yaml") -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if defaults:
        cfg.update(defaults)
    # file
    try:
        import yaml  # type: ignore
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                file_cfg = yaml.safe_load(f) or {}
                if isinstance(file_cfg, dict):
                    cfg.update(file_cfg)
    except Exception:
        pass
    # env (flat keys only)
    env_map = {
        "LOG_LEVEL": ("logging.log_level", None),
        "REDACT": ("logging.redact", lambda v: v.lower() in {"1", "true", "yes"}),
    }
    for env_key, (dst_key, caster) in env_map.items():
        val = os.getenv(env_key)
        if val is None:
            continue
        if caster:
            try:
                val = caster(val)
            except Exception:
                continue
        _set_nested(cfg, dst_key, val)
    return cfg


def _set_nested(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


