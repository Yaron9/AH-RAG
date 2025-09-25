from __future__ import annotations

from typing import Any, Dict, List, Tuple
import re

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore


def _safe_token_len(text: str, model: str | None = None) -> int:
    if not text:
        return 0
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base") if not model else tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            pass
    # Fallback heuristic: ~4 chars per token
    return max(1, len(text) // 4)


def _normalize_float(x: Any, scale: float = 10.0, default: float = 0.0) -> float:
    try:
        v = float(x)
        return max(0.0, min(1.0, v / scale))
    except Exception:
        return default


def _layer_weight(level: int | None) -> float:
    if level is None:
        return 0.5
    # L2 > L1 > L0
    return {2: 1.0, 1: 0.7, 0: 0.4}.get(level, 0.5)


def _extract_kept_spans(text: str) -> List[str]:
    spans: List[str] = []
    # numbers and dates
    spans += re.findall(r"\b\d{4}[-/.年]?(?:\d{1,2}[-/.月]?)?(?:\d{1,2}日)?\b", text)
    spans += re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
    # simple negation words (zh/en)
    neg = ["不", "未", "无", "否", "not", "no", "never", "without"]
    for n in neg:
        if n in text:
            spans.append(n)
    # dedup
    uniq = []
    seen = set()
    for s in spans:
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


class ContextProcessor:
    """
    Build a high-fidelity context string under a strict token budget.
    Input evidence is a dict with keys: {"summaries": [{node_id,...}], "entities": [...]}
    Requires `hg` (HierarchicalGraph-like) to fetch node metadata and texts.
    """

    def __init__(self, model_for_budget: str | None = None) -> None:
        self.model_for_budget = model_for_budget

    def build_context(
        self,
        evidence: Dict[str, Any],
        hg: Any,
        token_budget: int,
        config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        cfg = config or {}
        skeleton_ratio = float(cfg.get("skeleton_ratio", 0.2))
        reserve_ratio = float(cfg.get("reserve_ratio", 0.1))
        enable_kept_spans = bool(cfg.get("enable_kept_spans", True))
        enable_cache = bool(cfg.get("enable_cache", True))
        summarizer_model = cfg.get("summarizer_model") or None
        summarizer_max_tokens = int(cfg.get("summarizer_max_tokens", 256))

        # 1) gather candidate nodes
        nodes: List[str] = []
        for k in ("summaries", "entities"):
            for item in (evidence.get(k) or []):
                nid = item.get("node_id")
                if nid:
                    nodes.append(nid)
        # dedup while preserving order
        seen = set()
        ordered_nodes: List[str] = []
        for nid in nodes:
            if nid not in seen:
                seen.add(nid)
                ordered_nodes.append(nid)

        # 2) score and sort (normalized components)
        scored: List[Tuple[str, float]] = []
        for nid in ordered_nodes:
            d = hg.G.nodes.get(nid, {}) if hasattr(hg, "G") else {}
            judge = _normalize_float((d.get("judge_overall") or d.get("judge", {}).get("overall")))
            conf = _normalize_float(d.get("confidence") or d.get("confidence_score"))
            level = d.get("level")
            layer_w = _layer_weight(level)
            # weights can be tuned via cfg
            w = cfg.get("rank_weights") or {"judge": 0.4, "conf": 0.2, "layer": 0.4}
            score = w["judge"] * judge + w["conf"] * conf + w["layer"] * layer_w
            scored.append((nid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_nodes = [nid for nid, _ in scored]

        # 3) skeleton assembly
        budget_total = int(token_budget)
        budget_skeleton = int(budget_total * skeleton_ratio)
        budget_reserve = int(budget_total * reserve_ratio)
        tokens_used = 0
        skeleton_lines: List[str] = []
        details_lines: List[str] = []
        used_nodes: List[str] = []
        kept_spans: Dict[str, List[str]] = {}
        per_node_mode: Dict[str, str] = {}

        for nid in ranked_nodes:
            d = hg.G.nodes.get(nid, {}) if hasattr(hg, "G") else {}
            title = d.get("title") or d.get("name") or ""
            summary = d.get("summary_text") or d.get("summary") or d.get("description") or ""
            node_type = d.get("node_type") or ""
            line = f"- [{nid}] ({node_type}) {title} :: {summary[:160]}".strip()
            t = _safe_token_len(line, self.model_for_budget)
            if tokens_used + t <= budget_skeleton:
                skeleton_lines.append(line)
                tokens_used += t
                used_nodes.append(nid)
                if enable_kept_spans:
                    kept_spans[nid] = _extract_kept_spans(summary)
                per_node_mode[nid] = "skeleton"

        # 4) details assembly with compression if needed
        def compress_text(nid: str, text: str, target_tokens: int) -> str:
            # Minimal heuristic compression when no LLM available
            if target_tokens <= 0 or not text:
                return ""
            # try to shrink to approx target tokens using naive sentence trim
            parts = re.split(r"(?<=[。！？.!?])\s+", text)
            acc = []
            cur = 0
            for p in parts:
                if not p:
                    continue
                pt = _safe_token_len(p, self.model_for_budget)
                if cur + pt > target_tokens:
                    break
                acc.append(p)
                cur += pt
            out = " ".join(acc).strip()
            return out if out else text[: max(1, target_tokens * 4)]

        for nid in ranked_nodes:
            d = hg.G.nodes.get(nid, {}) if hasattr(hg, "G") else {}
            # prefer longer text if available
            raw = d.get("source_text") or d.get("source_text_ref") or d.get("summary_text") or d.get("description") or ""
            if not raw:
                continue
            # remaining budget excluding reserve
            remaining = max(0, budget_total - budget_reserve - tokens_used)
            if remaining <= 0:
                break
            raw_tokens = _safe_token_len(raw, self.model_for_budget)
            if raw_tokens <= remaining:
                details_lines.append(f"[DETAIL:{nid}]\n{raw.strip()}\n")
                tokens_used += raw_tokens
                per_node_mode[nid] = per_node_mode.get(nid, "detail_full")
            else:
                # compress to fit remaining
                target = min(remaining, summarizer_max_tokens)
                comp = compress_text(nid, raw, target)
                comp_tokens = _safe_token_len(comp, self.model_for_budget)
                if comp and comp_tokens <= remaining:
                    # enforce kept spans presence when enabled
                    if enable_kept_spans and kept_spans.get(nid):
                        for span in kept_spans[nid]:
                            if span and span not in comp and span in raw:
                                # append span contextually if missing
                                comp = (comp + f"\n[KEEP:{span}]").strip()
                                comp_tokens = _safe_token_len(comp, self.model_for_budget)
                                if comp_tokens > remaining:
                                    break
                    details_lines.append(f"[DETAIL:{nid}]\n{comp.strip()}\n")
                    tokens_used += comp_tokens
                    per_node_mode[nid] = per_node_mode.get(nid, "detail_compressed")
                else:
                    per_node_mode[nid] = per_node_mode.get(nid, "detail_dropped")

        context_text = (
            "# Evidence Skeleton\n" + "\n".join(skeleton_lines) + "\n\n# Evidence Details\n" + "\n".join(details_lines)
        ).strip()

        stats = {
            "budget_total": budget_total,
            "tokens_used": _safe_token_len(context_text, self.model_for_budget),
            "skeleton_tokens": _safe_token_len("\n".join(skeleton_lines), self.model_for_budget),
            "detail_tokens": _safe_token_len("\n".join(details_lines), self.model_for_budget),
            "compression_rate": 1.0 if not details_lines else min(1.0, tokens_used / max(1, budget_total)),
            "per_node_mode": per_node_mode,
            "kept_spans": kept_spans,
        }

        return {
            "context_text": context_text,
            "used_nodes": used_nodes,
            "stats": stats,
        }


