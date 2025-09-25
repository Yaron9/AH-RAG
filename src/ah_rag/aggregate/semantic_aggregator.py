from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import numpy as np
from ah_rag.extract.hypergraph_schema import HypergraphExtraction, Entity
import os
import json

# New imports for clustering
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

# New imports for summarization
from ah_rag.utils.llm_client import LLMModule, create_chat_completion, is_llm_enabled
from pydantic import BaseModel, ValidationError, TypeAdapter
import re
import networkx as nx
import random

ARTIFACT_DIR = "artifacts"

class TopicSummary(BaseModel):
    topic_id: int
    title: str
    summary: str
    confidence: float

class JudgeScore(BaseModel):
    id: int
    consistency: float
    accuracy: float
    informativeness: float
    overall: float
    comments: str

class SemanticAggregator:
    """
    Handles the hierarchical aggregation of knowledge from L0 to L1.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the aggregator with a sentence transformer model.

        Args:
            model_name: The name of the sentence-transformer model to use.
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.entities_map: Dict[str, Entity] = {}
        self.entity_embeddings: np.ndarray = None
        self.entity_names: List[str] = []

    def embed_l0_entities(self, l0_extractions: List[HypergraphExtraction]):
        """
        Generates vector embeddings for all unique L0 entities.
        """
        for extraction in l0_extractions:
            for entity in extraction.entities:
                if entity.name not in self.entities_map:
                    self.entities_map[entity.name] = entity
        
        unique_entities = list(self.entities_map.values())
        self.entity_names = [e.name for e in unique_entities]
        
        if not unique_entities:
            print("No unique entities found to embed.")
            return

        sentences_to_encode = [entity.description or entity.name for entity in unique_entities]

        print(f"Generating embeddings for {len(sentences_to_encode)} unique entities...")
        self.entity_embeddings = self.embedding_model.encode(
            sentences_to_encode,
            show_progress_bar=True
        )
        print("Embeddings generated successfully.\nShape of embeddings matrix:", self.entity_embeddings.shape)
        
        # persist embeddings
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        np.save(os.path.join(ARTIFACT_DIR, "embeddings.npy"), self.entity_embeddings)

    def cluster_with_bertopic(self,
                              prob_threshold: float = 0.10,
                              max_parents: int = 2,
                              umap_neighbors: int = 15,
                              umap_min_dist: float = 0.0,
                              min_topic_size: int = 2,
                              random_state: int = 42,
                              metric: str = 'euclidean') -> Dict[str, Any]:
        """
        Run BERTopic with UMAP+HDBSCAN, produce soft assignments and L1 topic nodes.
        Returns a dict with entity->parents mapping and topic node metadata.
        """
        if self.entity_embeddings is None or len(self.entity_names) == 0:
            raise RuntimeError("Embeddings not available. Run embed_l0_entities first.")

        # UMAP + HDBSCAN configuration
        num_points = len(self.entity_names)
        adaptive_neighbors = max(2, min(umap_neighbors, max(2, num_points - 1)))
        # Adapt min_topic_size for small datasets
        adaptive_min_topic = max(2, min(min_topic_size, num_points))

        umap_model = UMAP(n_neighbors=adaptive_neighbors,
                          n_components=2,
                          min_dist=umap_min_dist,
                          metric=metric,
                          random_state=random_state)
        hdbscan_model = HDBSCAN(min_cluster_size=adaptive_min_topic,
                                min_samples=None,
                                metric=metric,
                                prediction_data=True)

        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=True,
            min_topic_size=adaptive_min_topic
        )

        # Dummy docs are entity names/descriptions; embeddings are precomputed
        dummy_docs = self.entity_names
        topics, probs = topic_model.fit_transform(dummy_docs, embeddings=self.entity_embeddings)

        # Build topic -> members
        topic_to_members: Dict[int, List[int]] = {}
        for idx, topic_id in enumerate(topics):
            topic_to_members.setdefault(topic_id, []).append(idx)

        # Build entity -> parent topics with probabilities
        entity_to_parents: Dict[str, List[Dict[str, Any]]] = {}
        # Determine topic id ordering to match probs columns
        topic_info = topic_model.get_topic_info()
        ordered_topic_ids = [int(tid) for tid in topic_info.Topic.tolist() if int(tid) != -1]

        for i, name in enumerate(self.entity_names):
            # Default: fallback to assigned topic with prob 1.0 when probabilities are unusable
            fallback_topic = int(topics[i]) if i < len(topics) else -1

            if probs is None:
                if fallback_topic != -1:
                    entity_to_parents[name] = [{"topic_id": fallback_topic, "prob": 1.0}]
                else:
                    entity_to_parents[name] = []
                continue

            row = probs[i]
            # Some small-N settings may yield a scalar or 1D list
            try:
                row_arr = np.asarray(row)
            except Exception:
                row_arr = None

            if row_arr is None or row_arr.ndim == 0:
                if fallback_topic != -1:
                    entity_to_parents[name] = [{"topic_id": fallback_topic, "prob": 1.0}]
                else:
                    entity_to_parents[name] = []
                continue

            if row_arr.ndim > 1:
                row_arr = row_arr.ravel()

            # Align length with ordered topics
            cols = min(len(ordered_topic_ids), len(row_arr))
            if cols == 0:
                if fallback_topic != -1:
                    entity_to_parents[name] = [{"topic_id": fallback_topic, "prob": 1.0}]
                else:
                    entity_to_parents[name] = []
                continue

            topic_probs = list(zip(ordered_topic_ids[:cols], [float(x) for x in row_arr[:cols]]))
            topic_probs.sort(key=lambda x: x[1], reverse=True)
            selected = [
                {"topic_id": int(tid), "prob": float(p)}
                for tid, p in topic_probs if p >= prob_threshold
            ][:max_parents]

            # Ensure at least fallback assignment if nothing passes threshold
            if not selected and fallback_topic != -1:
                selected = [{"topic_id": fallback_topic, "prob": 1.0}]

            entity_to_parents[name] = selected

        # L1 topic nodes metadata
        l1_nodes: List[Dict[str, Any]] = []
        all_topics = topic_model.get_topics()
        for tid, words in all_topics.items():
            if tid == -1:
                continue
            member_indices = topic_to_members.get(tid, [])
            member_names = [self.entity_names[i] for i in member_indices]
            centroid = None
            if member_indices:
                centroid = np.mean(self.entity_embeddings[member_indices], axis=0).tolist()
            l1_nodes.append({
                "topic_id": int(tid),
                "top_words": [w for w, _ in words[:10]] if isinstance(words, list) else [],
                "members": member_names,
                "centroid": centroid
            })

        # persist artifacts
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        with open(os.path.join(ARTIFACT_DIR, "topics.json"), "w", encoding="utf-8") as f:
            json.dump({
                "entity_to_parents": entity_to_parents,
                "l1_nodes": l1_nodes
            }, f, ensure_ascii=False, indent=2)

        with open(os.path.join(ARTIFACT_DIR, "l1_nodes.json"), "w", encoding="utf-8") as f:
            json.dump(l1_nodes, f, ensure_ascii=False, indent=2)

        return {
            "entity_to_parents": entity_to_parents,
            "l1_nodes": l1_nodes,
            "model": topic_model
        }

    def _check_llm_enabled(self) -> bool:
        """Check if LLM is enabled for semantic aggregation."""
        return is_llm_enabled(LLMModule.SEMANTIC_AGGREGATION)

    def summarize_topics(self,
                         l1_nodes: List[Dict[str, Any]],
                         max_members_per_topic: int = 20,
                         member_snippet_len: int = 160,
                         temperature: float = 0.2,
                         language: str = "zh") -> List[TopicSummary]:
        """
        Generate title/summary/confidence for each L1 topic using Kimi.
        """
        if not l1_nodes:
            return []

        if not self._check_llm_enabled():
            print("LLM semantic aggregation disabled, skipping summarization")
            return []
        adapter = TypeAdapter(List[TopicSummary])
        out: List[TopicSummary] = []

        def build_prompt(node: Dict[str, Any]) -> str:
            topic_id = node.get("topic_id")
            top_words = node.get("top_words", [])
            members = node.get("members", [])
            # Collect member descriptions
            texts = []
            for name in members[:max_members_per_topic]:
                ent = self.entities_map.get(name)
                if ent is None:
                    continue
                desc = (ent.description or ent.name or "").strip()
                if desc:
                    texts.append(desc[:member_snippet_len])
            top_words_str = ", ".join(top_words)
            members_block = "\n- " + "\n- ".join(texts) if texts else "(no member descriptions)"
            schema_example = (
                "{\n"
                f"  \"topic_id\": {topic_id},\n"
                "  \"title\": \"一句话标题（不超过20字）\",\n"
                "  \"summary\": \"两三句话的摘要，聚合共性，避免枚举\",\n"
                "  \"confidence\": 7.5\n"
                "}"
            )
            return (
                f"你是一个高精度知识聚合助手，请用{language}输出规范 JSON，对以下主题簇生成摘要。\n"
                f"- 主题ID: {topic_id}\n"
                f"- 关键词: {top_words_str}\n"
                f"- 成员描述（截断）:\n{members_block}\n\n"
                f"严格只返回一个 JSON 对象，字段为 topic_id/title/summary/confidence。示例：\n{schema_example}\n"
                f"不要输出任何额外文本或注释。"
            )

        for idx, node in enumerate(l1_nodes):
            # Print initial info for human inspection
            print("\n--- Topic Initial Info ---")
            print(json.dumps({
                "topic_id": node.get("topic_id"),
                "top_words": node.get("top_words", []),
                "members": node.get("members", [])[:max_members_per_topic]
            }, ensure_ascii=False, indent=2))
            prompt = build_prompt(node)
            try:
                resp = create_chat_completion(
                    LLMModule.SEMANTIC_AGGREGATION,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = resp.choices[0].message.content
                # Extract JSON block
                m = re.search(r"\{[\s\S]*\}", content)
                if not m:
                    raise ValueError("No JSON object found in model response")
                cleaned = m.group(0)
                item = TypeAdapter(TopicSummary).validate_json(cleaned)
                out.append(item)
                # Print generated summary for human inspection
                print("--- Topic Summary ---")
                print(json.dumps(item.model_dump(), ensure_ascii=False, indent=2))
            except Exception as e:
                # Fallback: heuristic summary
                print(f"Summarization failed for topic {node.get('topic_id')}: {e}")
                tw = node.get("top_words", [])
                title = " / ".join(tw[:3]) or f"Topic {node.get('topic_id')}"
                # Build a naive summary from member descriptions
                members = node.get("members", [])
                snippets = []
                for name in members[:max_members_per_topic]:
                    ent = self.entities_map.get(name)
                    if ent and (ent.description or ent.name):
                        snippets.append((ent.description or ent.name)[:member_snippet_len])
                summary_text = " ".join(snippets[:3]) or "该主题由若干相关实体构成，语义相近。"
                fallback = TopicSummary(
                    topic_id=int(node.get("topic_id")),
                    title=title,
                    summary=summary_text,
                    confidence=5.0
                )
                out.append(fallback)
                print("--- Topic Summary (Fallback) ---")
                print(json.dumps(fallback.model_dump(), ensure_ascii=False, indent=2))

        # Persist summaries
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        with open(os.path.join(ARTIFACT_DIR, "l1_summaries.json"), "w", encoding="utf-8") as f:
            json.dump([s.model_dump() for s in out], f, ensure_ascii=False, indent=2)

        # Merge back into l1_nodes
        tid_to_summary = {s.topic_id: s for s in out}
        for n in l1_nodes:
            s = tid_to_summary.get(int(n.get("topic_id")))
            if s:
                n["title"] = s.title
                n["summary"] = s.summary
                n["confidence"] = s.confidence
        with open(os.path.join(ARTIFACT_DIR, "l1_nodes.json"), "w", encoding="utf-8") as f:
            json.dump(l1_nodes, f, ensure_ascii=False, indent=2)

        return out

    def judge_level_nodes(self,
                          nodes: List[Dict[str, Any]],
                          node_sample_size: int = 2,
                          out_path: str = os.path.join(ARTIFACT_DIR, "l2_judge_nodes.json"),
                          language: str = "zh") -> List[Dict[str, Any]]:
        if not nodes:
            return []
        if not self._check_llm_enabled():
            print("LLM semantic aggregation disabled, returning empty results")
            return []
        import random
        samples = random.sample(nodes, k=min(node_sample_size, len(nodes)))
        results: List[Dict[str, Any]] = []

        def build_node_prompt(n: Dict[str, Any]) -> str:
            meta = {
                "topic_id": int(n.get("topic_id")),
                "title": n.get("title"),
                "summary": n.get("summary") or n.get("summary_text"),
                "top_words": n.get("top_words", [])[:10],
                "members": n.get("members", [])[:10],
            }
            example = (
                "{\n"
                f"  \"id\": {meta['topic_id']},\n"
                "  \"consistency\": 8.0,\n"
                "  \"accuracy\": 7.5,\n"
                "  \"informativeness\": 7.0,\n"
                "  \"overall\": 7.5,\n"
                "  \"comments\": \"简要说明优缺点与改进点\"\n"
                "}"
            )
            return (
                f"你是严格的评审专家，请用{language}对以下L2主题节点进行打分（1-10，允许小数）。\n"
                f"元信息:\n{json.dumps(meta, ensure_ascii=False, indent=2)}\n"
                f"只返回一个JSON对象，字段为 id/consistency/accuracy/informativeness/overall/comments，示例：\n{example}\n"
                f"不要输出任何其他内容。"
            )

        for n in samples:
            prompt = build_node_prompt(n)
            try:
                resp = create_chat_completion(
                    LLMModule.SEMANTIC_AGGREGATION,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = resp.choices[0].message.content
                m = re.search(r"\{[\s\S]*\}", content)
                if not m:
                    raise ValueError("no-json")
                obj = json.loads(m.group(0))
                results.append(obj)
            except Exception:
                results.append({
                    "id": int(n.get("topic_id")),
                    "consistency": 6.0,
                    "accuracy": 6.0,
                    "informativeness": 6.0,
                    "overall": 6.0,
                    "comments": "fallback"
                })

        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return results

    def compute_escalation_metrics(self,
                                   l1_nodes: List[Dict[str, Any]],
                                   l2_nodes: List[Dict[str, Any]],
                                   l1_to_l2_path: str = os.path.join(ARTIFACT_DIR, "l1_to_l2.json"),
                                   l1_judge_path: str = os.path.join(ARTIFACT_DIR, "l1_judge_nodes.json"),
                                   l2_judge_path: str = os.path.join(ARTIFACT_DIR, "l2_judge_nodes.json"),
                                   thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        thresholds = thresholds or {"compression": 1.5, "improvement": 0.2, "coverage": 0.9}
        # Compression
        c_ratio = (len(l1_nodes) / max(1, len(l2_nodes))) if l2_nodes else 0.0
        # Coverage
        covered = 0
        total = len(l1_nodes)
        l1_to_l2 = {}
        if os.path.exists(l1_to_l2_path):
            with open(l1_to_l2_path, "r", encoding="utf-8") as f:
                l1_to_l2 = json.load(f)
        covered = sum(1 for n in l1_nodes if str(int(n.get("topic_id"))) in l1_to_l2)
        coverage = (covered / total) if total > 0 else 0.0
        # Judge means
        def mean_overall(path: str) -> float:
            if not os.path.exists(path):
                return None
            try:
                with open(path, "r", encoding="utf-8") as f:
                    arr = json.load(f)
                vals = [float(x.get("overall")) for x in arr if isinstance(x.get("overall"), (int, float))]
                return (sum(vals) / len(vals)) if vals else None
            except Exception:
                return None
        mean_l1 = mean_overall(l1_judge_path)
        mean_l2 = mean_overall(l2_judge_path)
        improvement = (mean_l2 - mean_l1) if (mean_l1 is not None and mean_l2 is not None) else None
        should_stop = False
        if len(l2_nodes) == 0:
            should_stop = True
        else:
            tests = []
            tests.append(c_ratio >= thresholds["compression"])
            tests.append((improvement is not None) and (improvement >= thresholds["improvement"]))
            tests.append(coverage >= thresholds["coverage"])
            should_stop = not all(tests)

        metrics = {
            "compression_ratio_l1_over_l2": round(c_ratio, 4),
            "coverage_l1_to_l2": round(coverage, 4),
            "mean_judge_overall_l1": mean_l1,
            "mean_judge_overall_l2": mean_l2,
            "improvement_overall": None if improvement is None else round(improvement, 4),
            "thresholds": thresholds,
            "should_stop_escalation": should_stop,
        }
        with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        return metrics

    def aggregate_level2_via_communities(self,
                                         l1_nodes: List[Dict[str, Any]],
                                         l1_edges_path: str = os.path.join(ARTIFACT_DIR, "l1_edges.json"),
                                         min_comm_size: int = 3,
                                         temperature: float = 0.2,
                                         language: str = "zh") -> List[Dict[str, Any]]:
        """
        Create L2 summaries by running community detection on L1 summary graph,
        then summarizing each community and mapping L1->L2 (belongs_to).
        """
        l2_nodes_path = os.path.join(ARTIFACT_DIR, "l2_nodes.json")
        l1_to_l2_path = os.path.join(ARTIFACT_DIR, "l1_to_l2.json")

        # Build graph of L1 summaries
        G = nx.Graph()
        tid_to_node = {int(n.get("topic_id")): n for n in l1_nodes}
        for tid in tid_to_node.keys():
            G.add_node(int(tid))
        if os.path.exists(l1_edges_path):
            with open(l1_edges_path, "r", encoding="utf-8") as f:
                edges = json.load(f)
            for e in edges:
                a = int(e.get("source")); b = int(e.get("target"))
                w = float(e.get("weight", 0.0))
                if a in G and b in G and w >= 0.15:
                    G.add_edge(a, b, weight=w)

        # Community detection (greedy modularity as built-in)
        comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight='weight')) if G.number_of_edges() > 0 else [set(G.nodes())]
        # Filter small communities
        comms = [list(c) for c in comms if len(c) >= min_comm_size]
        if not comms:
            os.makedirs(ARTIFACT_DIR, exist_ok=True)
            with open(l2_nodes_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            with open(l1_to_l2_path, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
            return []

        # Summarize each community
        if not self._check_llm_enabled():
            print("LLM semantic aggregation disabled, returning empty results")
            return []
        l2_nodes: List[Dict[str, Any]] = []
        l1_to_l2: Dict[str, int] = {}

        def build_comm_prompt(cid: int, tids: List[int]) -> str:
            texts = []
            for tid in tids[:30]:
                n = tid_to_node.get(int(tid))
                if not n:
                    continue
                title = n.get("title") or ""
                summ = n.get("summary") or n.get("summary_text") or ""
                texts.append(f"- {title}: {summ}")
            body = "\n".join(texts)
            example = (
                "{\n"
                f"  \"topic_id\": {cid},\n"
                "  \"title\": \"一句话L2主题标题\",\n"
                "  \"summary\": \"两三句概括该社区的共同主题与差异性\",\n"
                "  \"confidence\": 7.5\n"
                "}"
            )
            return (
                f"你是{language}摘要助手，请对以下L1主题社区生成L2摘要，严格返回JSON：\n"
                f"社区ID: {cid}\n成员L1摘要：\n{body}\n\n"
                f"只返回JSON对象（topic_id/title/summary/confidence），示例：\n{example}\n"
                f"不要输出任何额外文本。"
            )

        for cid, tids in enumerate(comms):
            # Compose centroid as mean of member centroids if available
            member_names = []
            centroids = []
            top_words = []
            for tid in tids:
                n = tid_to_node.get(int(tid))
                if not n:
                    continue
                member_names.append(f"sum:{int(tid)}")
                if n.get("centroid"):
                    centroids.append(np.array(n.get("centroid"), dtype=float))
                tw = n.get("top_words") or []
                top_words.extend(tw[:5])
            centroid = None
            if centroids:
                centroid = np.mean(centroids, axis=0).tolist()

            # Summarize
            prompt = build_comm_prompt(cid, tids)
            title = f"L2主题 {cid}"
            summary_txt = "该社区由多个相关L1摘要构成。"
            conf = 7.0
            try:
                resp = create_chat_completion(
                    LLMModule.SEMANTIC_AGGREGATION,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800
                )
                content = resp.choices[0].message.content
                m = re.search(r"\{[\s\S]*\}", content)
                if m:
                    obj = json.loads(m.group(0))
                    title = obj.get("title", title)
                    summary_txt = obj.get("summary", summary_txt)
                    conf = float(obj.get("confidence", conf))
            except Exception:
                pass

            l2_nodes.append({
                "topic_id": int(cid),
                "title": title,
                "summary": summary_txt,
                "confidence": conf,
                "top_words": list(dict.fromkeys(top_words))[:10],
                "members": member_names,
                "centroid": centroid,
                "level": 2,
            })
            for tid in tids:
                l1_to_l2[str(int(tid))] = int(cid)

        # Persist
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        with open(l2_nodes_path, "w", encoding="utf-8") as f:
            json.dump(l2_nodes, f, ensure_ascii=False, indent=2)
        with open(l1_to_l2_path, "w", encoding="utf-8") as f:
            json.dump(l1_to_l2, f, ensure_ascii=False, indent=2)

        return l2_nodes

    def generate_l1_relations(self,
                              l1_nodes: List[Dict[str, Any]],
                              min_overlap: int = 3,
                              min_jaccard: float = 0.2,
                              min_cosine: float = 0.5,
                              top_k: int | None = None) -> List[Dict[str, Any]]:
        """
        Build heuristic relations between L1 topics using member overlap, Jaccard, and centroid cosine.
        Returns a list of undirected edges with weights and diagnostics.
        """
        if not l1_nodes:
            return []

        # Prepare vectors
        def norm(v: List[float] | None) -> np.ndarray | None:
            if v is None:
                return None
            arr = np.asarray(v, dtype=np.float32)
            n = np.linalg.norm(arr)
            return arr / n if n > 0 else arr

        centroids = {int(n["topic_id"]): norm(n.get("centroid")) for n in l1_nodes}
        members = {int(n["topic_id"]): set(n.get("members", [])) for n in l1_nodes}
        confidences = {int(n["topic_id"]): float(n.get("confidence", 5.0)) for n in l1_nodes}

        topic_ids = sorted(centroids.keys())
        edges: List[Dict[str, Any]] = []

        for i, a in enumerate(topic_ids):
            for b in topic_ids[i+1:]:
                A = members.get(a, set())
                B = members.get(b, set())
                if not A and not B:
                    continue
                inter = A & B
                union = A | B
                overlap = len(inter)
                jaccard = (len(inter) / len(union)) if union else 0.0
                ca = centroids.get(a)
                cb = centroids.get(b)
                cosine = float(np.dot(ca, cb)) if (ca is not None and cb is not None) else 0.0

                pass_rules = (
                    (overlap >= min_overlap) or
                    (jaccard >= min_jaccard) or
                    (cosine >= min_cosine)
                )
                if not pass_rules:
                    continue

                weight = 0.5 * jaccard + 0.5 * cosine
                conf = (confidences.get(a, 5.0) + confidences.get(b, 5.0)) / 2.0

                edges.append({
                    "source": int(a),
                    "target": int(b),
                    "relation_type": "related_summary",
                    "weight": round(weight, 4),
                    "overlap": int(overlap),
                    "jaccard": round(jaccard, 4),
                    "cosine": round(cosine, 4),
                    "confidence": round(conf, 2)
                })

        # Optional top-k pruning by weight
        if top_k is not None and len(edges) > top_k:
            edges = sorted(edges, key=lambda e: e["weight"], reverse=True)[:top_k]

        # Persist
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        with open(os.path.join(ARTIFACT_DIR, "l1_edges.json"), "w", encoding="utf-8") as f:
            json.dump(edges, f, ensure_ascii=False, indent=2)

        # Update nodes with neighbor ids
        adj: Dict[int, List[Tuple[int, float]]] = {int(n["topic_id"]): [] for n in l1_nodes}
        for e in edges:
            adj[e["source"]].append((e["target"], e["weight"]))
            adj[e["target"]].append((e["source"], e["weight"]))
        for n in l1_nodes:
            tid = int(n["topic_id"]) 
            n["neighbors"] = [
                {"topic_id": t, "weight": w} for t, w in sorted(adj.get(tid, []), key=lambda x: x[1], reverse=True)
            ]
        with open(os.path.join(ARTIFACT_DIR, "l1_nodes.json"), "w", encoding="utf-8") as f:
            json.dump(l1_nodes, f, ensure_ascii=False, indent=2)

        return edges

    def judge_samples(self,
                      l1_nodes: List[Dict[str, Any]],
                      l1_edges: List[Dict[str, Any]],
                      node_sample_size: int = 5,
                      edge_sample_size: int = 5,
                      language: str = "zh") -> Dict[str, List[JudgeScore]]:
        """
        Randomly sample nodes/edges and ask Kimi to score (1-10) on consistency, accuracy, informativeness, and overall.
        Persist results to artifacts. Non-blocking; failures fallback to neutral scores.
        """
        if not l1_nodes and not l1_edges:
            return {"nodes": [], "edges": []}

        if not self._check_llm_enabled():
            print("LLM semantic aggregation disabled, returning empty results")
            return {"nodes": [], "edges": []}
        node_samples = random.sample(l1_nodes, k=min(node_sample_size, len(l1_nodes))) if l1_nodes else []
        edge_samples = random.sample(l1_edges, k=min(edge_sample_size, len(l1_edges))) if l1_edges else []

        def build_node_prompt(n: Dict[str, Any]) -> str:
            meta = {
                "topic_id": int(n.get("topic_id")),
                "title": n.get("title"),
                "summary": n.get("summary"),
                "top_words": n.get("top_words", []),
                "members": n.get("members", [])[:10]
            }
            example = (
                "{\n"
                f"  \"id\": {meta['topic_id']},\n"
                "  \"consistency\": 8.0,\n"
                "  \"accuracy\": 7.5,\n"
                "  \"informativeness\": 7.0,\n"
                "  \"overall\": 7.5,\n"
                "  \"comments\": \"简要说明优缺点与改进点\"\n"
                "}"
            )
            return (
                f"你是严格的评审专家，请用{language}对以下主题节点进行打分（1-10，允许小数）。\n"
                f"元信息:\n{json.dumps(meta, ensure_ascii=False, indent=2)}\n"
                f"只返回一个JSON对象，字段为 id/consistency/accuracy/informativeness/overall/comments，示例：\n{example}\n"
                f"不要输出任何其他内容。"
            )

        def build_edge_prompt(e: Dict[str, Any]) -> str:
            meta = {
                "source": e.get("source"),
                "target": e.get("target"),
                "relation_type": e.get("relation_type"),
                "diagnostics": {
                    "overlap": e.get("overlap"),
                    "jaccard": e.get("jaccard"),
                    "cosine": e.get("cosine"),
                    "weight": e.get("weight")
                }
            }
            example = (
                "{\n"
                f"  \"id\": {meta['source']}\n"
                "            ,\n"
                "  \"consistency\": 7.0,\n"
                "  \"accuracy\": 7.0,\n"
                "  \"informativeness\": 6.5,\n"
                "  \"overall\": 7.0,\n"
                "  \"comments\": \"关系是否充分、诊断是否支持该边\"\n"
                "}"
            )
            return (
                f"你是严格的评审专家，请用{language}对以下主题关系进行打分（1-10，允许小数）。\n"
                f"元信息:\n{json.dumps(meta, ensure_ascii=False, indent=2)}\n"
                f"只返回一个JSON对象，字段为 id/consistency/accuracy/informativeness/overall/comments，示例：\n{example}\n"
                f"不要输出任何其他内容。\n"
                f"注意：请用 overall 表示对该关系的总体可信度判断。"
            )

        node_scores: List[JudgeScore] = []
        for n in node_samples:
            prompt = build_node_prompt(n)
            try:
                resp = create_chat_completion(
                    LLMModule.SEMANTIC_AGGREGATION,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = resp.choices[0].message.content
                m = re.search(r"\{[\s\S]*\}", content)
                cleaned = m.group(0) if m else None
                score = TypeAdapter(JudgeScore).validate_json(cleaned) if cleaned else None
                if not score:
                    raise ValueError("no-json")
                node_scores.append(score)
            except Exception:
                # fallback neutral
                node_scores.append(JudgeScore(
                    id=int(n.get("topic_id")),
                    consistency=6.0,
                    accuracy=6.0,
                    informativeness=6.0,
                    overall=6.0,
                    comments="fallback"
                ))

        edge_scores: List[JudgeScore] = []
        for e in edge_samples:
            prompt = build_edge_prompt(e)
            try:
                resp = create_chat_completion(
                    LLMModule.SEMANTIC_AGGREGATION,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = resp.choices[0].message.content
                m = re.search(r"\{[\s\S]*\}", content)
                cleaned = m.group(0) if m else None
                # use combined id as source for identification
                sid = int(e.get("source")) if e.get("source") is not None else 0
                score = TypeAdapter(JudgeScore).validate_json(cleaned) if cleaned else None
                if not score:
                    raise ValueError("no-json")
                edge_scores.append(score)
            except Exception:
                edge_scores.append(JudgeScore(
                    id=int(e.get("source", 0)),
                    consistency=6.0,
                    accuracy=6.0,
                    informativeness=6.0,
                    overall=6.0,
                    comments="fallback"
                ))

        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        with open(os.path.join(ARTIFACT_DIR, "l1_judge_nodes.json"), "w", encoding="utf-8") as f:
            json.dump([s.model_dump() for s in node_scores], f, ensure_ascii=False, indent=2)
        with open(os.path.join(ARTIFACT_DIR, "l1_judge_edges.json"), "w", encoding="utf-8") as f:
            json.dump([s.model_dump() for s in edge_scores], f, ensure_ascii=False, indent=2)

        return {"nodes": node_scores, "edges": edge_scores}

if __name__ == '__main__':
    from ah_rag.extract.hypergraph_extractor import HypergraphExtractor

    print("Running L0 Extractor...")
    extractor = HypergraphExtractor()
    sample_document = """
    In Q4 2023, the tech giant InnovateCorp, led by its visionary CEO Dr. Evelyn Reed,
    announced a groundbreaking new product, the 'QuantumLeap Processor'. 
    This announcement, made during the annual TechSummit in Geneva,
    promises to revolutionize the field of quantum computing.
    The processor's development was a collaborative effort with the
    MIT Department of Physics.
    """
    extractions = extractor.extract(sample_document)
    
    if extractions:
        aggregator = SemanticAggregator()
        aggregator.embed_l0_entities(extractions)
        result = aggregator.cluster_with_bertopic()
        print("Soft clustering & parent assignment complete.")
        print("Artifacts written to:", ARTIFACT_DIR)
        # Summarize topics
        summaries = aggregator.summarize_topics(result["l1_nodes"])
        print(f"Summaries generated: {len(summaries)}")
        # Generate relations
        edges = aggregator.generate_l1_relations(result["l1_nodes"])
        print(f"Relations generated: {len(edges)} (saved to artifacts/l1_edges.json)")
        # Aggregate to L2 via communities
        l2 = aggregator.aggregate_level2_via_communities(result["l1_nodes"])
        print(f"L2 summaries generated: {len(l2)} (saved to artifacts/l2_nodes.json)")
        # L2 judge sampling
        aggregator.judge_level_nodes(l2, node_sample_size=2)
        # Compute escalation metrics and stop flag
        metrics = aggregator.compute_escalation_metrics(result["l1_nodes"], l2)
        print("Escalation metrics:", json.dumps(metrics, ensure_ascii=False))
        if edges:
            print("--- Candidate Edges ---")
            print(json.dumps(edges, ensure_ascii=False, indent=2))
        # Judge sampling
        judge = aggregator.judge_samples(result["l1_nodes"], edges, node_sample_size=2, edge_sample_size=2)
        print(f"Judge scores -> nodes: {len(judge['nodes'])}, edges: {len(judge['edges'])}")
    else:
        print("Could not generate L0 extractions to proceed with aggregation.")
