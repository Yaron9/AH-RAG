from __future__ import annotations

from typing import Any, Dict, List
import json
import re
import unicodedata
import os

try:
    import evaluate  # type: ignore
except Exception:  # pragma: no cover
    evaluate = None  # type: ignore


class AnswerEvaluator:
    """
    Universal RAG evaluation framework based on Claude.md diagnosis theory.
    Core principle: RAG quality = Retriever × Generator (multiplicative relationship)

    Supports both system-specific and universal evaluation modes.
    """

    def __init__(self, system_type: str = "ah_rag"):
        """
        Args:
            system_type: RAG system type ("ah_rag", "naive_rag", "graph_rag", etc.)
        """
        self.system_type = system_type

    def evaluate(self, answer_obj: Dict[str, Any], graph: Any, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Main evaluation entry point.
        Returns unified evaluation results with diagnosis.
        """
        cfg = config or {}

        # Extract session data if available
        session_data = answer_obj.get('session_data', {})
        question = answer_obj.get('query', '')

        # Level 1: Universal RAG evaluation (retriever + generator)
        universal_metrics = self.evaluate_universal(answer_obj, question, session_data, cfg)

        # Level 2: System-specific metrics
        specific_metrics = self.evaluate_system_specific(session_data, graph)

        # Level 3: Apply Claude.md diagnosis formula
        diagnosis = self.apply_diagnosis_formula(universal_metrics)

        return {
            "scores": {
                "overall": self._calculate_overall_score(universal_metrics),
                **universal_metrics
            },
            "details": {
                "universal_metrics": universal_metrics,
                "specific_metrics": specific_metrics,
                "diagnosis": diagnosis,
                "system_type": self.system_type
            }
        }

    def evaluate_universal(self, answer_obj: Dict[str, Any], question: str, session_data: Dict, config: Dict) -> Dict[str, float]:
        """
        Universal metrics applicable to all RAG systems.
        Based on Claude.md theory: separate retriever and generator evaluation.
        """
        # Retriever metrics
        retriever_metrics = self.evaluate_retriever(session_data, question, answer_obj, config)

        # Generator metrics
        generator_metrics = self.evaluate_generator(answer_obj, session_data, question, config)

        # Combine with current quantitative/qualitative/efficiency
        quantitative = self.evaluate_quantitative(
            answer_obj.get('answer', ''),
            answer_obj.get('gold_answers', [])
        )
        qualitative = self.evaluate_qualitative(answer_obj, question, config)

        return {
            # Retriever metrics (Claude.md core)
            'contextual_relevancy': retriever_metrics.get('contextual_relevancy', 0.0),
            'contextual_recall': retriever_metrics.get('contextual_recall', 0.0),
            'contextual_precision': retriever_metrics.get('contextual_precision', 0.0),

            # Generator metrics (Claude.md core)
            'faithfulness': generator_metrics.get('faithfulness', 0.0),
            'answer_relevancy': generator_metrics.get('answer_relevancy', 0.0),

            # Existing metrics (backward compatibility)
            'f1': quantitative.get('f1', 0.0),
            'em': quantitative.get('em', 0.0),
            'judge_overall': qualitative.get('overall', 0.0)
        }

    def evaluate_retriever(self, session_data: Dict, question: str, answer_obj: Dict, config: Dict) -> Dict[str, float]:
        """
        Evaluate retriever performance based on Claude.md framework.
        Key insight: Contextual recall is most critical for avoiding hallucination.
        """
        # First try to get evidence from answer_obj (new format)
        evidence = answer_obj.get('evidence', {})
        retrieved_nodes = answer_obj.get('retrieved_nodes', [])
        context = answer_obj.get('context', {})

        # Fallback to session_data format
        if not retrieved_nodes:
            actions = session_data.get('stats', {}).get('actions', [])
            if not actions:
                return {'contextual_relevancy': 0.0, 'contextual_recall': 0.0, 'contextual_precision': 0.0}
            retrieved_nodes = self._extract_retrieved_nodes(session_data)

        # 1. Contextual Relevancy: How relevant is retrieved content?
        contextual_relevancy = self._calculate_contextual_relevancy(retrieved_nodes, question, evidence)

        # 2. Contextual Recall: Did we find all key information needed?
        contextual_recall = self._calculate_contextual_recall(retrieved_nodes, answer_obj, evidence, context)

        # 3. Contextual Precision: Are relevant items ranked higher?
        contextual_precision = self._calculate_contextual_precision(retrieved_nodes, question)

        return {
            'contextual_relevancy': contextual_relevancy,
            'contextual_recall': contextual_recall,
            'contextual_precision': contextual_precision
        }

    def evaluate_generator(self, answer_obj: Dict, session_data: Dict, question: str, config: Dict) -> Dict[str, float]:
        """
        Evaluate generator performance based on Claude.md framework.
        Key metrics: faithfulness and answer relevancy.
        """
        # 1. Faithfulness: Is output consistent with retrieved info?
        faithfulness = self._calculate_faithfulness(answer_obj, session_data, config)

        # 2. Answer Relevancy: Does answer address the question?
        answer_relevancy = self._calculate_answer_relevancy(answer_obj, question, config)

        return {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy
        }

    def apply_diagnosis_formula(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Apply Claude.md diagnosis formula to identify failure points.

        Formula:
        - High faithfulness + Low relevancy → Retriever problem
        - Low faithfulness + High relevancy → Generator problem
        - Both low → System problem
        - Both high → Edge case investigation
        """
        faithfulness = metrics.get('faithfulness', 0.0)
        answer_relevancy = metrics.get('answer_relevancy', 0.0)
        contextual_recall = metrics.get('contextual_recall', 0.0)

        # Apply diagnosis thresholds
        high_threshold = 0.7
        low_threshold = 0.5

        if faithfulness > high_threshold and answer_relevancy < low_threshold:
            primary_issue = 'retriever'
            reason = '检索内容不相关，生成器无从下手'
            confidence = 0.8
        elif faithfulness < low_threshold and answer_relevancy > high_threshold:
            primary_issue = 'generator'
            reason = '检索内容没问题，但生成器没用好'
            confidence = 0.8
        elif faithfulness < low_threshold and answer_relevancy < low_threshold:
            primary_issue = 'both'
            reason = '整个系统都有问题，需全面检查'
            confidence = 0.9
        elif contextual_recall < low_threshold:
            primary_issue = 'retriever'
            reason = '召回率低导致生成器脑补（Claude.md关键洞察）'
            confidence = 0.85
        else:
            primary_issue = 'edge_case'
            reason = '系统整体正常，需排查边缘情况'
            confidence = 0.3

        return {
            'primary_issue': primary_issue,
            'reason': reason,
            'confidence': confidence,
            'metrics_snapshot': {
                'faithfulness': faithfulness,
                'answer_relevancy': answer_relevancy,
                'contextual_recall': contextual_recall
            }
        }

    def evaluate_system_specific(self, session_data: Dict, graph: Any) -> Dict[str, Any]:
        """
        System-specific evaluation metrics.
        Can be extended for different RAG systems.
        """
        if self.system_type == 'ah_rag':
            return self._evaluate_ahrag_specific(session_data, graph)
        elif self.system_type == 'naive_rag':
            return self._evaluate_naive_specific(session_data)
        else:
            return {}

    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall score emphasizing Claude.md key insight:
        contextual_recall is most critical for quality.
        """
        weights = {
            'contextual_recall': 0.3,    # Most important per Claude.md
            'faithfulness': 0.25,
            'contextual_relevancy': 0.2,
            'answer_relevancy': 0.15,
            'contextual_precision': 0.1
        }

        score = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight
                total_weight += weight

        return score / max(total_weight, 1e-9)

    # Helper methods for metric calculations
    def _extract_retrieved_nodes(self, session_data: Dict) -> List[str]:
        """Extract retrieved node IDs from session data."""
        nodes = []
        actions = session_data.get('stats', {}).get('actions', [])
        for action in actions:
            if action.get('action') == 'semantic_anchor':
                # For semantic_anchor, we need to extract the returned nodes
                # This would need to be enhanced based on actual data structure
                nodes.extend(action.get('returned_nodes', []))
            elif action.get('action') in ['expand_parents', 'expand_children', 'expand_related']:
                inputs = action.get('inputs', [])
                nodes.extend(inputs)
        return list(set(nodes))  # Remove duplicates

    def _calculate_contextual_relevancy(self, retrieved_nodes: List[str], question: str, evidence: Dict = None) -> float:
        """
        Calculate how relevant retrieved content is to the question.
        Enhanced implementation using evidence data.
        """
        if not retrieved_nodes:
            return 0.0

        evidence = evidence or {}

        # Check if we have evidence content to analyze
        summaries = evidence.get('summaries', [])
        entities = evidence.get('entities', [])

        if not summaries and not entities:
            # Fallback: assume moderate relevancy
            return 0.7

        # Analyze evidence content for relevancy
        question_lower = question.lower()
        relevant_items = 0
        total_items = len(summaries) + len(entities)

        # Check summaries for relevancy
        for summary in summaries:
            title = (summary.get('title') or '').lower()
            summary_text = (summary.get('summary') or '').lower()
            if any(keyword in title or keyword in summary_text
                   for keyword in question_lower.split() if len(keyword) > 3):
                relevant_items += 1

        # Check entities for relevancy
        for entity in entities:
            title = (entity.get('title') or '').lower()
            summary_text = (entity.get('summary') or '').lower()
            if any(keyword in title or keyword in summary_text
                   for keyword in question_lower.split() if len(keyword) > 3):
                relevant_items += 1

        # Calculate relevancy ratio
        if total_items > 0:
            relevancy = relevant_items / total_items
            return min(1.0, relevancy * 1.1)  # Slight boost to account for partial matches

        return 0.7  # Default moderate relevancy

    def _calculate_contextual_recall(self, retrieved_nodes: List[str], answer_obj: Dict, evidence: Dict = None, context: Dict = None) -> float:
        """
        Calculate if we retrieved all key information needed.
        This is the CRITICAL metric per Claude.md.
        """
        if not retrieved_nodes:
            return 0.0

        # Enhanced recall calculation using evidence data
        evidence = evidence or {}
        context = context or {}

        # Count available evidence
        summaries = evidence.get('summaries', [])
        entities = evidence.get('entities', [])
        total_evidence = len(summaries) + len(entities)

        # If we have processed context, use context statistics
        if context and context.get('stats'):
            used_nodes = context.get('used_nodes', [])
            context_stats = context.get('stats', {})

            # Good recall if we used substantial portion of retrieved evidence
            if total_evidence > 0:
                usage_ratio = len(used_nodes) / total_evidence
                return min(1.0, usage_ratio * 1.2)  # Allow exceeding 1.0 then cap

        # Fallback: check citations vs retrieved nodes
        citations = answer_obj.get('citations', [])
        if citations and retrieved_nodes:
            citation_coverage = len(citations) / max(len(retrieved_nodes), 1)
            return min(1.0, citation_coverage)

        # If we have evidence but no clear usage stats, moderate recall
        if total_evidence > 0:
            return 0.7

        # No evidence available
        return 0.0

    def _calculate_contextual_precision(self, retrieved_nodes: List[str], question: str) -> float:
        """
        Calculate if relevant content is ranked higher.
        """
        if not retrieved_nodes:
            return 0.0
        # Simplified: assume reasonable precision
        return 0.65

    def _calculate_faithfulness(self, answer_obj: Dict, session_data: Dict, config: Dict) -> float:
        """
        Calculate faithfulness using LLM-as-a-Judge if available,
        otherwise use qualitative scores.
        """
        # Try to use existing qualitative evaluation
        qual_scores = self.evaluate_qualitative(answer_obj, answer_obj.get('query', ''), config)
        correctness = qual_scores.get('correctness', 0.0)

        # Normalize to 0-1 range (qualitative scores are 1-10)
        return correctness / 10.0 if correctness > 0 else 0.5

    def _calculate_answer_relevancy(self, answer_obj: Dict, question: str, config: Dict) -> float:
        """
        Calculate how well the answer addresses the question.
        """
        # Use qualitative coverage as proxy for relevancy
        qual_scores = self.evaluate_qualitative(answer_obj, question, config)
        coverage = qual_scores.get('coverage', 0.0)

        # Normalize to 0-1 range
        return coverage / 10.0 if coverage > 0 else 0.5

    def _evaluate_ahrag_specific(self, session_data: Dict, graph: Any) -> Dict[str, Any]:
        """AH-RAG specific metrics."""
        stats = session_data.get('stats', {}).get('cumulative', {})
        actions = session_data.get('stats', {}).get('actions', [])

        # Layer utilization analysis
        layer_usage = {'L0': 0, 'L1': 0, 'L2': 0}
        for action in actions:
            inputs = action.get('inputs', [])
            for node_id in inputs:
                if node_id.startswith('ent:'):
                    layer_usage['L0'] += 1
                elif node_id.startswith('sum:'):
                    # Would need graph to determine L1 vs L2
                    layer_usage['L1'] += 1

        # Calculate graph coverage correctly (flatten lists and make hashable)
        all_inputs = []
        for action in actions:
            inputs = action.get('inputs', [])
            if inputs:
                all_inputs.extend(inputs)

        return {
            'reasoning_steps': stats.get('steps', 0),
            'layer_utilization': layer_usage,
            'graph_coverage': len(set(all_inputs))  # Unique nodes accessed
        }

    def _evaluate_naive_specific(self, session_data: Dict) -> Dict[str, Any]:
        """Naive RAG specific metrics."""
        stats = session_data.get('stats', {}).get('cumulative', {})
        return {
            'retrieval_efficiency': stats.get('time_s', 0.0),
            'context_utilization': 1.0  # Assumes full context usage
        }

    # 3.4.2 Level 1: quantitative F1/EM
    def evaluate_quantitative(self, pred_text: str, gold_texts: List[str]) -> Dict[str, float]:
        pred = _normalize_text(pred_text)
        refs = [_normalize_text(x) for x in (gold_texts or []) if x]
        if not refs:
            return {"f1": 0.0, "em": 0.0}
        if evaluate is None:
            # simple char-level F1/EM fallback
            def f1_char(a: str, b: str) -> float:
                if not a or not b:
                    return 0.0
                import collections
                ca = collections.Counter(a)
                cb = collections.Counter(b)
                overlap = sum((ca & cb).values())
                if overlap == 0:
                    return 0.0
                precision = overlap / max(1, len(a))
                recall = overlap / max(1, len(b))
                return 2 * precision * recall / max(1e-9, (precision + recall))
            f1s = [f1_char(pred, r) for r in refs]
            ems = [1.0 if pred == r else 0.0 for r in refs]
            return {"f1": max(f1s), "em": max(ems)}
        # use evaluate squad_v2 style
        squad = evaluate.load("squad_v2") if evaluate is not None else None
        best_f1 = 0.0
        best_em = 0.0
        for r in refs:
            res = squad.compute(predictions=[{"id": "0", "prediction_text": pred, "no_answer_probability": 0.0}],
                                references=[{"id": "0", "answers": {"text": [r], "answer_start": [0]}}])
            best_f1 = max(best_f1, float(res.get("f1", 0.0)))
            best_em = max(best_em, float(res.get("exact", 0.0)))
        return {"f1": best_f1, "em": best_em}

    # 3.4.2 Level 2: qualitative LLM-Judge
    def evaluate_qualitative(self, answer_json: Dict[str, Any], question: str, config: Dict[str, Any] | None = None) -> Dict[str, float]:
        cfg = config or {}
        judge = (cfg.get("evaluation") or {}).get("judge") or {}
        use_llm = bool(judge.get("use_llm", False))
        if not use_llm:
            return {"correctness": 0.0, "coverage": 0.0, "clarity": 0.0, "overall": 0.0}
        try:
            from ah_rag.utils.llm_client import LLMModule, create_chat_completion, is_llm_enabled
            if not is_llm_enabled(LLMModule.EVALUATION_JUDGE):
                return {"correctness": 0.0, "coverage": 0.0, "clarity": 0.0, "overall": 0.0}
        except Exception:  # pragma: no cover
            return {"correctness": 0.0, "coverage": 0.0, "clarity": 0.0, "overall": 0.0}
        # Model configuration is now handled by unified LLM client
        prompt = _build_judge_prompt(question, answer_json)
        retries = int(judge.get("max_retries", 1))
        for i in range(retries + 1):
            try:
                resp = create_chat_completion(
                    LLMModule.EVALUATION_JUDGE,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300
                )
                txt = resp.choices[0].message.content or ""
                obj = _extract_json(txt)
                if obj:
                    return {
                        "correctness": float(obj.get("correctness", 0.0)),
                        "coverage": float(obj.get("coverage", 0.0)),
                        "clarity": float(obj.get("clarity", 0.0)),
                        "overall": float(obj.get("overall", 0.0)),
                    }
            except Exception:
                continue
        return {"correctness": 0.0, "coverage": 0.0, "clarity": 0.0, "overall": 0.0}

    # 3.4.2 Level 3: efficiency
    def evaluate_efficiency(self, summary_json_path: str) -> Dict[str, float]:
        try:
            with open(summary_json_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            stats = obj.get("stats", {}).get("cumulative", {})
            out = {
                "steps": float(stats.get("steps", 0)),
                "nodes_expanded": float(stats.get("expansions", 0)),
                "latency_s": float(stats.get("time_s", 0.0)),
            }
            # optional token metrics if available
            if "tokens_total" in stats:
                out["tokens_total"] = float(stats.get("tokens_total", 0))
            return out
        except Exception:
            return {"steps": 0.0, "nodes_expanded": 0.0, "latency_s": 0.0}


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # remove punctuation using Unicode categories (avoid \p escapes unsupported by re)
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))
    # basic number/date normalization hooks can be added here if needed
    return s


def _build_judge_prompt(question: str, ans: Dict[str, Any]) -> str:
    schema = {"correctness": 0, "coverage": 0, "clarity": 0, "overall": 0}
    return (
        f"You are a strict QA judge. Score the answer on 1-10 for each dimension.\n"
        f"Question: {question}\n"
        f"Answer JSON: {json.dumps(ans, ensure_ascii=False)}\n"
        f"Dimensions: correctness (factual alignment), coverage (evidence completeness), clarity (conciseness & coherence).\n"
        f"Return only a JSON: {json.dumps(schema, ensure_ascii=False)}"
    )


def _extract_json(text: str) -> Dict[str, Any] | None:
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

