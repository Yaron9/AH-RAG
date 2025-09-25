import json
import os
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
import datetime

# Vector index (ChromaDB)
import chromadb
from chromadb.config import Settings

ARTIFACT_DIR = "artifacts"
GRAPH_DIR = "graph"
STRUCTURE_FILE = os.path.join(GRAPH_DIR, "structure.json")
META_FILE = os.path.join(GRAPH_DIR, "meta.json")


class HierarchicalGraph:
    """
    Unified NetworkX DiGraph to store L0 entities, L0 hyperedges, and L1+ summaries.
    All metadata are attached to nodes/edges as attributes.
    Large embeddings are referenced by path/index rather than embedded inline.
    """

    def __init__(self) -> None:
        self.G: nx.DiGraph = nx.DiGraph()
        # Lightweight indices for common lookups
        self.name_to_entity_id: Dict[str, str] = {}
        self.topic_to_summary_id: Dict[int, str] = {}
        # Cache for query embedding model
        self._query_embedder_name: Optional[str] = None
        self._query_embedder: Optional[SentenceTransformer] = None
        # Default search params
        self.G.graph.setdefault("search_params", {
            "alpha": 0.6,
            "beta": 0.2,
            "gamma": 0.1,
            "delta": 0.1,
            "judge_overall_min": None,
            "confidence_min": None,
            "member_top_m": 5,
            "type_filter": None,  # e.g., {"entity","summary"}
            "layer_boost": {"entity": 0.0, "summary": 1.0, "hyperedge": 0.0},
        })
        # Dirty tracking & vector index meta
        self.G.graph.setdefault("dirty", False)
        self.G.graph.setdefault("vector_index", {"db_path": "vector_db", "model": "all-MiniLM-L6-v2", "indexed_nodes": 0, "indexed_meta": {}})

    # ----------------------
    # ID helpers (stable)
    # ----------------------
    @staticmethod
    def _hash(text: str, length: int = 10) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]

    @staticmethod
    def make_entity_id(name: str) -> str:
        return f"ent:{HierarchicalGraph._hash(name)}"

    @staticmethod
    def make_hyperedge_id(uid: str) -> str:
        return f"hedge:{uid}"

    @staticmethod
    def make_summary_id(topic_id: int) -> str:
        return f"sum:{int(topic_id)}"

    # ----------------------
    # Node/edge creation
    # ----------------------
    def add_entity(self, name: str, description: Optional[str] = None, entity_type: Optional[str] = None,
                   embedding_ref: Optional[Tuple[str, int]] = None, l1_parents: Optional[Dict[str, float]] = None) -> str:
        node_id = self.name_to_entity_id.get(name)
        if node_id is None:
            node_id = self.make_entity_id(name)
            self.G.add_node(node_id,
                            node_type="entity",
                            name=name,
                            description=description,
                            entity_type=entity_type,
                            embedding_ref=embedding_ref,
                            l1_parents=json.dumps(l1_parents) if l1_parents else None)
            self.name_to_entity_id[name] = node_id
        else:
            # Update existing entity with new information if provided
            node_data = self.G.nodes[node_id]
            old_desc = node_data.get("description")
            if description is not None and (old_desc is None or old_desc == ""):
                print(f"DEBUG: Updating {name} description from '{old_desc}' to '{description}'")
                node_data["description"] = description
            elif description is not None:
                print(f"DEBUG: NOT updating {name} description (current: '{old_desc}', new: '{description}')")
            if entity_type is not None and (node_data.get("entity_type") is None or node_data.get("entity_type") == ""):
                node_data["entity_type"] = entity_type
            if embedding_ref is not None:
                node_data["embedding_ref"] = embedding_ref
            if l1_parents is not None:
                node_data["l1_parents"] = json.dumps(l1_parents)
        self.G.graph["dirty"] = True
        return node_id

    def add_hyperedge(self, uid: str, description: str, relation_type: str,
                      confidence_score: Optional[float] = None, source_text_ref: Optional[str] = None) -> str:
        node_id = self.make_hyperedge_id(uid)
        if not self.G.has_node(node_id):
            self.G.add_node(node_id,
                            node_type="hyperedge",
                            description=description,
                            relation_type=relation_type,
                            confidence_score=confidence_score,
                            source_text_ref=source_text_ref)
        self.G.graph["dirty"] = True
        return node_id

    def add_summary(self, topic_id: int, title: Optional[str] = None, summary_text: Optional[str] = None,
                    confidence: Optional[float] = None, top_words: Optional[List[str]] = None,
                    members: Optional[List[str]] = None, judge_scores: Optional[Dict[str, Any]] = None,
                    centroid: Optional[List[float]] = None) -> str:
        node_id = self.topic_to_summary_id.get(int(topic_id))
        if node_id is None:
            node_id = self.make_summary_id(topic_id)
            self.G.add_node(node_id,
                            node_type="summary",
                            topic_id=int(topic_id),
                            title=title,
                            summary_text=summary_text,
                            confidence=confidence,
                            top_words=json.dumps(top_words) if top_words else None,
                            members=json.dumps(members) if members else None,
                            judge_scores=json.dumps(judge_scores) if judge_scores else None,
                            centroid=json.dumps(centroid) if centroid is not None else None)
            self.topic_to_summary_id[int(topic_id)] = node_id
        else:
            # Update properties if node exists
            data = self.G.nodes[node_id]
            if title is not None:
                data["title"] = title
            if summary_text is not None:
                data["summary_text"] = summary_text
            if confidence is not None:
                data["confidence"] = confidence
            if top_words is not None:
                data["top_words"] = json.dumps(top_words)
            if members is not None:
                data["members"] = json.dumps(members)
            if judge_scores is not None:
                data["judge_scores"] = json.dumps(judge_scores)
            if centroid is not None:
                data["centroid"] = json.dumps(centroid)
        self.G.graph["dirty"] = True
        return node_id

    def add_participation(self, entity_id: str, hyperedge_id: str, role: Optional[str] = None) -> None:
        self.G.add_edge(entity_id, hyperedge_id, edge_type="participates_in", role=role)
        self.G.graph["dirty"] = True

    def add_belongs_to(self, entity_id: str, summary_id: str, prob: Optional[float] = None) -> None:
        self.G.add_edge(entity_id, summary_id, edge_type="belongs_to", prob=prob)
        self.G.graph["dirty"] = True

    def add_related(self, summary_a: str, summary_b: str, weight: Optional[float] = None,
                    jaccard: Optional[float] = None, cosine: Optional[float] = None,
                    overlap: Optional[int] = None, confidence: Optional[float] = None) -> None:
        self.G.add_edge(summary_a, summary_b, edge_type="related_to",
                        weight=weight, jaccard=jaccard, cosine=cosine,
                        overlap=overlap, confidence=confidence)
        self.G.graph["dirty"] = True

    # ----------------------
    # Persistence
    # ----------------------
    def save(self, directory: str = GRAPH_DIR, meta: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(directory, exist_ok=True)
        # Use node-link JSON to preserve arbitrary attributes
        data = nx.node_link_data(self.G)
        with open(os.path.join(directory, "structure.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # Persist meta plus search params if provided
        merged_meta = meta.copy() if meta else {}
        merged_meta["search_params"] = self.G.graph.get("search_params", {})
        # add snapshot hash & dirty flag & vector index meta
        merged_meta["graph_hash"] = self._graph_snapshot_hash()
        merged_meta["dirty"] = self.G.graph.get("dirty", False)
        merged_meta["vector_index"] = self.G.graph.get("vector_index", {})
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(merged_meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str = GRAPH_DIR) -> "HierarchicalGraph":
        with open(os.path.join(directory, "structure.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        hg = cls()
        hg.G = nx.node_link_graph(data, directed=True, edges="links")
        # Rebuild light indices
        for node_id, attr in hg.G.nodes(data=True):
            if attr.get("node_type") == "entity" and attr.get("name"):
                hg.name_to_entity_id[attr["name"]] = node_id
            if attr.get("node_type") == "summary" and attr.get("topic_id") is not None:
                hg.topic_to_summary_id[int(attr["topic_id"])] = node_id
        # Load search params from meta if exists
        meta_path = os.path.join(directory, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                sp = meta.get("search_params")
                if isinstance(sp, dict):
                    hg.G.graph["search_params"] = {**hg.G.graph.get("search_params", {}), **sp}
                if isinstance(meta.get("vector_index"), dict):
                    hg.G.graph["vector_index"] = meta.get("vector_index")
                if isinstance(meta.get("dirty"), bool):
                    hg.G.graph["dirty"] = meta.get("dirty")
            except Exception:
                pass
        return hg

    def _graph_snapshot_hash(self) -> str:
        """Compute a lightweight hash for the current node set and key attributes."""
        items = []
        for nid, d in self.G.nodes(data=True):
            items.append(
                (
                    nid,
                    d.get("node_type"),
                    d.get("name"),
                    d.get("title"),
                    d.get("summary_text"),
                    d.get("description"),
                )
            )
        items.sort(key=lambda x: x[0])
        blob = json.dumps(items, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    # ----------------------
    # Build from artifacts directory
    # ----------------------
    def build_from_artifacts(self, artifacts_dir: str = ARTIFACT_DIR) -> None:
        # topics.json contains entity_to_parents and l1_nodes
        topics_path = os.path.join(artifacts_dir, "topics.json")
        l1_nodes_path = os.path.join(artifacts_dir, "l1_nodes.json")
        l1_edges_path = os.path.join(artifacts_dir, "l1_edges.json")
        l1_summaries_path = os.path.join(artifacts_dir, "l1_summaries.json")
        judge_nodes_path = os.path.join(artifacts_dir, "l1_judge_nodes.json")
        judge_edges_path = os.path.join(artifacts_dir, "l1_judge_edges.json")
        hyperedges_path = os.path.join(artifacts_dir, "extractions.json")
        l2_nodes_path = os.path.join(artifacts_dir, "l2_nodes.json")
        l1_to_l2_path = os.path.join(artifacts_dir, "l1_to_l2.json")

        entity_to_parents: Dict[str, List[Dict[str, Any]]] = {}
        l1_nodes: List[Dict[str, Any]] = []
        l1_edges: List[Dict[str, Any]] = []
        l1_summaries: List[Dict[str, Any]] = []
        judge_nodes: List[Dict[str, Any]] = []
        judge_edges: List[Dict[str, Any]] = []
        hyperedges: List[Dict[str, Any]] = []
        l2_nodes: List[Dict[str, Any]] = []
        l1_to_l2: Dict[str, int] = {}

        if os.path.exists(topics_path):
            with open(topics_path, "r", encoding="utf-8") as f:
                tp = json.load(f)
            entity_to_parents = tp.get("entity_to_parents", {})
            l1_nodes = tp.get("l1_nodes", [])
        if os.path.exists(l1_nodes_path):
            with open(l1_nodes_path, "r", encoding="utf-8") as f:
                l1_nodes = json.load(f)
        if os.path.exists(l1_edges_path):
            with open(l1_edges_path, "r", encoding="utf-8") as f:
                l1_edges = json.load(f)
        if os.path.exists(l1_summaries_path):
            with open(l1_summaries_path, "r", encoding="utf-8") as f:
                l1_summaries = json.load(f)
        if os.path.exists(judge_nodes_path):
            with open(judge_nodes_path, "r", encoding="utf-8") as f:
                judge_nodes = json.load(f)
        if os.path.exists(judge_edges_path):
            with open(judge_edges_path, "r", encoding="utf-8") as f:
                judge_edges = json.load(f)
        if os.path.exists(hyperedges_path):
            with open(hyperedges_path, "r", encoding="utf-8") as f:
                hyperedges = json.load(f)
        if os.path.exists(l2_nodes_path):
            with open(l2_nodes_path, "r", encoding="utf-8") as f:
                l2_nodes = json.load(f)
        if os.path.exists(l1_to_l2_path):
            with open(l1_to_l2_path, "r", encoding="utf-8") as f:
                l1_to_l2 = json.load(f)

        # 1) First pass: collect all entity info from hyperedges
        entity_info = {}
        for h in hyperedges:
            for ent in h.get("entities", []):
                name = ent.get("name")
                if not name:
                    continue
                if name not in entity_info:
                    entity_info[name] = {
                        "descriptions": [],
                        "types": [],
                        "entity_type": ent.get("type"),
                        "description": ent.get("description")
                    }
                # Collect multiple descriptions
                desc = ent.get("description")
                if desc and desc not in entity_info[name]["descriptions"]:
                    entity_info[name]["descriptions"].append(desc)

        # 2) Create entities with rich descriptions
        for name, parents in entity_to_parents.items():
            # Use collected entity info or defaults
            info = entity_info.get(name, {})
            descriptions = info.get("descriptions", [])
            # Combine multiple descriptions for richer context
            combined_desc = "; ".join(descriptions) if descriptions else None
            entity_type = info.get("entity_type")


            ent_id = self.add_entity(name=name,
                                   description=combined_desc,
                                   entity_type=entity_type,
                                   embedding_ref=(os.path.join(ARTIFACT_DIR, "embeddings.npy"), -1),
                                   l1_parents={str(p.get("topic_id")): p.get("prob") for p in parents})

        # 2b) Add any entities from hyperedges not in entity_to_parents
        for name, info in entity_info.items():
            if name not in entity_to_parents:
                print(f"DEBUG: Adding missing entity {name}")
                descriptions = info.get("descriptions", [])
                combined_desc = "; ".join(descriptions) if descriptions else None
                self.add_entity(name=name,
                              description=combined_desc,
                              entity_type=info.get("entity_type"))

        # 3) Summaries (L1); attach summaries and centroid/top_words/members if present
        # Build mapping topic_id -> summary node id
        for node in l1_nodes:
            topic_id = int(node.get("topic_id"))
            title = node.get("title")
            summary_text = node.get("summary") or node.get("summary_text")
            confidence = node.get("confidence")
            top_words = node.get("top_words") or []
            members = node.get("members") or []
            centroid = node.get("centroid")
            sid = self.add_summary(topic_id=topic_id, title=title, summary_text=summary_text,
                                   confidence=confidence, top_words=top_words,
                                   members=members, centroid=centroid)

        # 4) belongs_to edges using entity_to_parents
        for name, parents in entity_to_parents.items():
            ent_id = self.name_to_entity_id.get(name)
            if not ent_id:
                continue
            for p in parents:
                tid = int(p.get("topic_id"))
                prob = p.get("prob")
                sid = self.topic_to_summary_id.get(tid)
                if sid:
                    self.add_belongs_to(ent_id, sid, prob=prob)

        # 4) related_to edges
        for e in l1_edges:
            a = self.topic_to_summary_id.get(int(e.get("source")))
            b = self.topic_to_summary_id.get(int(e.get("target")))
            if a and b:
                self.add_related(a, b, weight=e.get("weight"), jaccard=e.get("jaccard"),
                                 cosine=e.get("cosine"), overlap=e.get("overlap"),
                                 confidence=e.get("confidence"))

        # 4b) hyperedges and participation edges
        for i, h in enumerate(hyperedges):
            # Generate unique ID if not present
            uid = h.get("id")
            if uid is None:
                uid = f"hedge_{i}_{h.get('relation_type', 'unknown')}"
            hid = self.add_hyperedge(uid=str(uid),
                                     description=h.get("hyperedge"),
                                     relation_type=h.get("relation_type"),
                                     confidence_score=h.get("confidence_score"))
            for ent in h.get("entities", []):
                name = ent.get("name")
                if not name:
                    continue
                eid = self.name_to_entity_id.get(name)
                if eid:
                    role = ent.get("role") if isinstance(ent, dict) else None
                    self.add_participation(eid, hid, role=role)

        # 5) L2 summaries and L1->L2 belongs_to
        if l2_nodes:
            for n in l2_nodes:
                tid = int(n.get("topic_id"))
                sid = self.add_summary(topic_id=tid, title=n.get("title"), summary_text=n.get("summary"),
                                       confidence=n.get("confidence"), top_words=n.get("top_words"),
                                       members=n.get("members"), centroid=n.get("centroid"))
                # mark level on node
                self.G.nodes[sid]["level"] = 2
            # map L1 -> L2
            for l1_tid_str, l2_tid in l1_to_l2.items():
                try:
                    l1_tid = int(l1_tid_str)
                except Exception:
                    continue
                l1_sid = self.topic_to_summary_id.get(l1_tid)
                l2_sid = self.topic_to_summary_id.get(int(l2_tid))
                if l1_sid and l2_sid:
                    self.add_belongs_to(l1_sid, l2_sid, prob=1.0)

        # 5) attach judge scores to summary nodes; store edge judge at graph-level meta
        tid_to_node = {int(self.G.nodes[n].get("topic_id")): n for n in self.G.nodes if self.G.nodes[n].get("node_type") == "summary"}
        for s in judge_nodes:
            tid = int(s.get("id"))
            nid = tid_to_node.get(tid)
            if nid:
                self.G.nodes[nid]["judge_scores"] = json.dumps(s)
        # Store edge-level judge scores globally if cannot map precisely
        if judge_edges:
            self.G.graph["judge_edges"] = judge_edges

    # ----------------------
    # Simple query helpers
    # ----------------------
    def get_parents(self, node_id: str) -> List[str]:
        return list(self.G.successors(node_id))

    def get_children(self, node_id: str) -> List[str]:
        return list(self.G.predecessors(node_id))

    def find_entity(self, name: str) -> Optional[str]:
        return self.name_to_entity_id.get(name)

    def find_summary(self, topic_id: int) -> Optional[str]:
        return self.topic_to_summary_id.get(int(topic_id))

    # ----------------------
    # Query helpers (edge-type aware)
    # ----------------------
    def _neighbors_by_edge_type(self, node_id: str, direction: str, edge_type: str) -> List[str]:
        if direction == "out":
            nbrs = []
            for _, v, data in self.G.out_edges(node_id, data=True):
                if data.get("edge_type") == edge_type:
                    nbrs.append(v)
            return nbrs
        else:
            nbrs = []
            for u, _, data in self.G.in_edges(node_id, data=True):
                if data.get("edge_type") == edge_type:
                    nbrs.append(u)
            return nbrs

    # belongs_to
    def get_belongs_to(self, node_id: str) -> List[str]:
        """Summaries this node belongs to (entity -> summary)."""
        return self._neighbors_by_edge_type(node_id, "out", "belongs_to")

    def get_summary_members(self, summary_id: str) -> List[str]:
        """Entities that belong to this summary (entity -> summary)."""
        return self._neighbors_by_edge_type(summary_id, "in", "belongs_to")

    # participates_in
    def get_hyperedge_participants(self, hyperedge_id: str) -> List[str]:
        return self._neighbors_by_edge_type(hyperedge_id, "in", "participates_in")

    def get_entity_hyperedges(self, entity_id: str) -> List[str]:
        return self._neighbors_by_edge_type(entity_id, "out", "participates_in")

    # siblings via shared parent summaries
    def get_siblings(self, node_id: str) -> List[str]:
        parents = set(self.get_belongs_to(node_id))
        sibs: set[str] = set()
        for p in parents:
            for child in self.get_summary_members(p):
                if child != node_id:
                    sibs.add(child)
        return list(sibs)

    # search helpers
    def search_by_name_or_title(self, q: str, limit: int = 20) -> List[Tuple[str, Dict[str, Any]]]:
        ql = q.lower()
        results: List[Tuple[str, Dict[str, Any]]] = []
        for nid, data in self.G.nodes(data=True):
            name = str(data.get("name") or "").lower()
            title = str(data.get("title") or "").lower()
            if ql in name or ql in title:
                results.append((nid, data))
            if len(results) >= limit:
                break
        return results

    def summaries_with_top_word(self, word: str, limit: int = 50) -> List[str]:
        w = word.lower()
        out: List[str] = []
        for nid, data in self.G.nodes(data=True):
            if data.get("node_type") != "summary":
                continue
            tw = data.get("top_words")
            if tw:
                try:
                    arr = json.loads(tw) if isinstance(tw, str) else tw
                except Exception:
                    arr = []
                if any(w in str(x).lower() for x in arr):
                    out.append(nid)
            if len(out) >= limit:
                break
        return out

    # ----------------------
    # Validation utilities
    # ----------------------
    def validate_belongs_to_dag(self) -> bool:
        sub = nx.DiGraph()
        for u, v, data in self.G.edges(data=True):
            if data.get("edge_type") == "belongs_to":
                sub.add_edge(u, v)
        return nx.is_directed_acyclic_graph(sub)

    def validate_required_attributes(self) -> Dict[str, List[str]]:
        problems: Dict[str, List[str]] = {"entity": [], "hyperedge": [], "summary": []}
        for nid, data in self.G.nodes(data=True):
            nt = data.get("node_type")
            if nt == "entity":
                if not data.get("name"):
                    problems["entity"].append(nid)
            elif nt == "hyperedge":
                if not data.get("description") or not data.get("relation_type"):
                    problems["hyperedge"].append(nid)
            elif nt == "summary":
                if data.get("topic_id") is None:
                    problems["summary"].append(nid)
        return problems

    def stats(self) -> Dict[str, Any]:
        counts = {k: 0 for k in ["entity", "hyperedge", "summary"]}
        for _, data in self.G.nodes(data=True):
            t = data.get("node_type")
            if t in counts:
                counts[t] += 1
        edge_counts = {k: 0 for k in ["participates_in", "belongs_to", "related_to"]}
        for _, _, d in self.G.edges(data=True):
            et = d.get("edge_type")
            if et in edge_counts:
                edge_counts[et] += 1
        return {
            "nodes": counts,
            "edges": edge_counts,
            "n_nodes": self.G.number_of_nodes(),
            "n_edges": self.G.number_of_edges(),
        }

    # ----------------------
    # 1.4 Multilayer vector index + hybrid search
    # ----------------------
    def _embedding_text(self, node_id: str) -> Tuple[str, Dict[str, Any]]:
        d = self.G.nodes[node_id]
        node_type = d.get("node_type")
        layer = 0 if node_type == "entity" else (1 if node_type == "summary" else 0)
        judge = None
        if d.get("judge_scores"):
            try:
                js = json.loads(d["judge_scores"]) if isinstance(d["judge_scores"], str) else d["judge_scores"]
                judge = float(js.get("overall", 0.0))
            except Exception:
                judge = None
        confidence = d.get("confidence")

        if node_type == "entity":
            name = d.get("name") or ""
            desc = d.get("description") or ""
            text = f"Entity: {name}. {desc}"
        elif node_type == "summary":
            title = d.get("title") or ""
            summ = d.get("summary_text") or d.get("summary") or ""
            top_words = []
            if d.get("top_words"):
                try:
                    top_words = json.loads(d["top_words"]) if isinstance(d["top_words"], str) else d["top_words"]
                except Exception:
                    top_words = []
            tw = (", ".join(top_words[:10])) if top_words else ""
            text = f"Summary: {title}. {summ}. Keywords: {tw}"
        else:
            # hyperedge (optional)
            desc = d.get("description") or ""
            rtype = d.get("relation_type") or ""
            text = f"Relation: {rtype}. {desc}"

        metadata = {
            "node_id": node_id,
            "node_type": node_type,
            "layer": layer,
            "judge_overall": judge,
            "confidence": confidence,
            "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
        return text, metadata

    def build_vector_index(self, db_path: str = "vector_db", layers: set = {0, 1}, include_hyperedges: bool = False,
                            model_name: str = "all-MiniLM-L6-v2", upsert_only: bool = True, reset: bool = False) -> None:
        os.makedirs(db_path, exist_ok=True)
        client = chromadb.PersistentClient(path=db_path, settings=Settings(allow_reset=True))
        if reset:
            try:
                client.delete_collection(name="nodes")
            except Exception:
                pass
        coll = client.get_or_create_collection(name="nodes", metadata={"hnsw:space": "cosine"})

        # Gather nodes to index
        to_index_ids: List[str] = []
        to_index_docs: List[str] = []
        to_index_meta: List[Dict[str, Any]] = []
        indexed_meta_prev: Dict[str, Any] = {}
        vi = self.G.graph.get("vector_index", {})
        if isinstance(vi.get("indexed_meta"), dict):
            indexed_meta_prev = vi.get("indexed_meta")

        def index_key(nid: str, d: Dict[str, Any]) -> str:
            return hashlib.sha1(
                (str(d.get("node_type")) + "|" + str(d.get("name") or d.get("title") or "") + "|" + str(d.get("summary_text") or "") + "|" + str(d.get("description") or "")).encode("utf-8")
            ).hexdigest()

        for nid, data in self.G.nodes(data=True):
            nt = data.get("node_type")
            if nt == "entity" and 0 in layers:
                key = index_key(nid, data)
                if (not upsert_only) or (indexed_meta_prev.get(nid) != key):
                    text, meta = self._embedding_text(nid)
                    to_index_ids.append(nid)
                    to_index_docs.append(text)
                    to_index_meta.append(meta)
            elif nt == "summary" and 1 in layers:
                key = index_key(nid, data)
                if (not upsert_only) or (indexed_meta_prev.get(nid) != key):
                    text, meta = self._embedding_text(nid)
                    to_index_ids.append(nid)
                    to_index_docs.append(text)
                    to_index_meta.append(meta)
            elif nt == "hyperedge" and include_hyperedges:
                key = index_key(nid, data)
                if (not upsert_only) or (indexed_meta_prev.get(nid) != key):
                    text, meta = self._embedding_text(nid)
                    to_index_ids.append(nid)
                    to_index_docs.append(text)
                    to_index_meta.append(meta)

        if not to_index_ids:
            return

        # Encode
        model = SentenceTransformer(model_name)
        embeddings = model.encode(to_index_docs, show_progress_bar=True, normalize_embeddings=True)

        # Upsert in (ids are unique in one collection)
        # Chroma requires string ids
        coll.upsert(ids=to_index_ids, documents=to_index_docs, embeddings=[e.tolist() for e in embeddings], metadatas=to_index_meta)

        # Update index meta
        new_meta = indexed_meta_prev.copy()
        for nid in to_index_ids:
            d = self.G.nodes[nid]
            new_meta[nid] = index_key(nid, d)
        self.G.graph["vector_index"] = {
            "db_path": db_path,
            "model": model_name,
            "indexed_nodes": len(new_meta),
            "indexed_meta": new_meta,
        }
        self.G.graph["dirty"] = False

    def search(self, query: str, top_k: int = 5, member_top_m: int = 5,
               alpha: Optional[float] = None, beta: Optional[float] = None, gamma: Optional[float] = None, delta: Optional[float] = None,
               judge_overall_min: Optional[float] = None, confidence_min: Optional[float] = None,
               type_filter: Optional[List[str]] = None,
               return_cluster: bool = False,
               db_path: Optional[str] = None, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        # Resolve parameters from stored defaults if not given
        sp = self.G.graph.get("search_params", {})
        alpha = alpha if alpha is not None else sp.get("alpha", 0.6)
        beta = beta if beta is not None else sp.get("beta", 0.2)
        gamma = gamma if gamma is not None else sp.get("gamma", 0.1)
        delta = delta if delta is not None else sp.get("delta", 0.1)
        if judge_overall_min is None:
            judge_overall_min = sp.get("judge_overall_min")
        if confidence_min is None:
            confidence_min = sp.get("confidence_min")
        member_top_m = member_top_m if member_top_m is not None else sp.get("member_top_m", 5)
        if type_filter is None:
            tf = sp.get("type_filter")
            # normalize to list if set
            if isinstance(tf, (set, tuple)):
                type_filter = list(tf)
            else:
                type_filter = tf
        layer_boost_map = sp.get("layer_boost", {"entity": 0.0, "summary": 1.0, "hyperedge": 0.0})

        # Resolve vector index location/model from graph meta when not provided
        if not db_path or not model_name:
            vi = self.G.graph.get("vector_index", {}) if isinstance(self.G.graph.get("vector_index"), dict) else {}
            db_path = db_path or vi.get("db_path", "vector_db")
            model_name = model_name or vi.get("model", "all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=db_path, settings=Settings(allow_reset=True))
        coll = client.get_or_create_collection(name="nodes")
        # Compute query embedding locally to avoid requiring a collection embedding function
        if self._query_embedder is None or self._query_embedder_name != model_name:
            self._query_embedder = SentenceTransformer(model_name)
            self._query_embedder_name = model_name
        q_emb = self._query_embedder.encode([query], normalize_embeddings=True)
        # Query by embedding
        res = coll.query(query_embeddings=[q_emb[0].tolist()], n_results=top_k, include=["distances", "metadatas", "documents"])
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        # Convert cosine distance to similarity
        seeds = []
        for nid, dist, meta in zip(ids, distances, metas):
            if nid is None:
                continue
            semantic = 1.0 - float(dist) if dist is not None else 0.0
            seeds.append({"node_id": nid, "semantic": semantic, "meta": meta})

        # Graph expansion
        expanded: Dict[str, Dict[str, Any]] = {}
        for s in seeds:
            nid = s["node_id"]
            data = self.G.nodes.get(nid, {})
            nt = data.get("node_type")
            expanded[nid] = {**s, "node_type": nt}
            if nt == "entity":
                for parent in self.get_belongs_to(nid)[:2]:
                    if parent not in expanded:
                        expanded[parent] = {"node_id": parent, "semantic": s["semantic"] * 0.9, "meta": {"expanded": True}, "node_type": "summary"}
            elif nt == "summary":
                for child in self.get_summary_members(nid)[:member_top_m]:
                    if child not in expanded:
                        expanded[child] = {"node_id": child, "semantic": s["semantic"] * 0.85, "meta": {"expanded": True}, "node_type": "entity"}

        # Rerank
        results: List[Dict[str, Any]] = []
        for nid, info in expanded.items():
            d = self.G.nodes.get(nid, {})
            judge = None
            if d.get("judge_scores"):
                try:
                    js = json.loads(d["judge_scores"]) if isinstance(d["judge_scores"], str) else d["judge_scores"]
                    judge = float(js.get("overall", 0.0))
                except Exception:
                    judge = None
            conf = d.get("confidence")
            layer = 0 if d.get("node_type") == "entity" else (1 if d.get("node_type") == "summary" else 0)
            # Type filter
            if type_filter and d.get("node_type") not in type_filter:
                continue
            # Threshold filters
            if judge_overall_min is not None and (judge is None or judge < judge_overall_min):
                continue
            if confidence_min is not None and (conf is None or conf < confidence_min):
                continue

            layer_boost = float(layer_boost_map.get(d.get("node_type"), 0.0))
            semantic = float(info.get("semantic", 0.0))
            judge_term = (1.0 / (1.0 + np.exp(-((judge or 0.0) / 10.0)))) if judge is not None else 0.0
            conf_term = float(conf) / 10.0 if conf is not None else 0.0
            score = alpha * semantic + beta * judge_term + gamma * conf_term + delta * layer_boost
            results.append({
                "node_id": nid,
                "node_type": d.get("node_type"),
                "layer": layer,
                "semantic": round(semantic, 4),
                "judge_overall": judge,
                "confidence": conf,
                "score": round(float(score), 4),
                "name": d.get("name"),
                "title": d.get("title"),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        reranked = results[:top_k]
        if return_cluster:
            return {
                "seeds": seeds,
                "expanded": list(expanded.values()),
                "reranked": reranked,
            }
        return reranked


if __name__ == "__main__":
    hg = HierarchicalGraph()
    hg.build_from_artifacts(ARTIFACT_DIR)
    meta = {
        "artifacts": [
            "topics.json", "l1_nodes.json", "l1_edges.json", "l1_summaries.json", "l1_judge_nodes.json", "l1_judge_edges.json", "embeddings.npy"
        ]
    }
    hg.save(GRAPH_DIR, meta=meta)
    print(f"Graph built with {hg.G.number_of_nodes()} nodes and {hg.G.number_of_edges()} edges.")
    print(f"Saved to {STRUCTURE_FILE}")
    # Validation report
    print("Validation - belongs_to DAG:", hg.validate_belongs_to_dag())
    print("Validation - required attributes:", json.dumps(hg.validate_required_attributes(), ensure_ascii=False))
    print("Stats:", json.dumps(hg.stats(), ensure_ascii=False))
    # Build vector index and run a sample search
    print("Building vector index (L0+L1+L2 if available)...")
    hg.build_vector_index(db_path="vector_db", layers={0, 1, 2}, reset=True)
    print("Search results:")
    # Example: apply filters and weights from stored params
    out = hg.search("量子处理器 发布", top_k=5, return_cluster=True)
    print(json.dumps(out, ensure_ascii=False, indent=2))
