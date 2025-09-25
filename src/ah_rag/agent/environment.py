from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import json
import os
import uuid
from datetime import datetime
import time
import networkx as nx

from ..graph.hierarchical_graph import HierarchicalGraph


class GraphEnvironment:
    """
    V1 minimal environment over HierarchicalGraph providing core actions:
    - reset(seed_query|seed_nodes)
    - semantic_anchor(query, top_k, filters/weights)
    - expand_to_lca(node_ids)
    - query_node_details(node_id)

    Returns lightweight observations and an info dict for diagnostics.
    """

    def __init__(self, graph_dir: str = "graph", random_state: int = 42,
                 logging_enabled: bool = True, log_dir: str = "artifacts/phase2", session_id: Optional[str] = None,
                 debug: bool = False, log_level: str = "normal", redact: bool = True) -> None:
        self.graph_dir = graph_dir
        self.random_state = random_state
        self.hg: Optional[HierarchicalGraph] = None
        self.last_query: Optional[str] = None
        self.last_results: Optional[Dict[str, Any]] = None
        self.step_count: int = 0
        # V2 state
        self.selection_set: set[str] = set()
        self.frontier_set: set[str] = set()
        # dynamic filters/weights applied to subsequent searches
        self.current_filters: Dict[str, Any] = {
            "judge_overall_min": None,
            "confidence_min": None,
            "type_filter": None,
        }
        self.current_weights: Dict[str, Any] = {
            "alpha": None, "beta": None, "gamma": None, "delta": None,
            "member_top_m": None,
            "top_k": 5,
        }
        # logging
        self.logging_enabled = logging_enabled
        self.log_dir = log_dir
        self.session_id = session_id or datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.session_path = os.path.join(self.log_dir, self.session_id)
        self.debug = debug
        # cost & summary stats
        self.stats: Dict[str, Any] = {
            "actions": [],
            "cumulative": {"steps": 0, "queries": 0, "expansions": 0, "time_s": 0.0},
        }
        # logger
        self.logger = None
        if self.logging_enabled:
            from ah_rag.utils.logging_init import get_logger
            self.logger = get_logger(self.session_path, self.session_id, level=log_level, redact=redact)
            os.makedirs(self.session_path, exist_ok=True)
            try:
                with open(os.path.join(self.session_path, "session.json"), "w", encoding="utf-8") as f:
                    json.dump({"session_id": self.session_id, "created_at": datetime.utcnow().isoformat() + "Z"}, f)
            except Exception:
                pass

        # lazy load
        self._ensure_graph_loaded()

    def _ensure_graph_loaded(self) -> None:
        if self.hg is None:
            self.hg = HierarchicalGraph.load(self.graph_dir)

    def _log(self, event: Dict[str, Any]) -> None:
        if not self.logging_enabled:
            return
        try:
            event = {**event, "step": self.step_count}
            if self.logger is not None:
                self.logger.info(**event)
        except Exception:
            pass

    # ----------------------
    # Helpers
    # ----------------------
    def _node_brief(self, node_id: str) -> Dict[str, Any]:
        d = self.hg.G.nodes.get(node_id, {})
        node_type = d.get("node_type")
        entity_type = d.get("entity_type")  # Add entity_type field
        title = d.get("title")
        name = d.get("name")
        layer = 0 if node_type == "entity" else (d.get("level") or (1 if node_type == "summary" else 0))
        judge = None
        if d.get("judge_scores"):
            try:
                js = json.loads(d["judge_scores"]) if isinstance(d["judge_scores"], str) else d["judge_scores"]
                judge = float(js.get("overall", 0.0))
            except Exception:
                judge = None
        conf = d.get("confidence")
        return {
            "node_id": node_id,
            "node_type": node_type,
            "entity_type": entity_type,  # Include entity_type in observation
            "layer": layer,
            "title": title,
            "name": name,
            "judge_overall": judge,
            "confidence": conf,
        }

    def _observation(self, seeds: List[Dict[str, Any]], reranked: List[Dict[str, Any]]) -> Dict[str, Any]:
        # compress nodes into briefs
        def brief_from_result(res: Dict[str, Any]) -> Dict[str, Any]:
            nid = res.get("node_id")
            base = self._node_brief(nid)
            base.update({
                "score": res.get("score"),
                "semantic": res.get("semantic"),
            })
            return base

        obs: Dict[str, Any] = {
            "selection": [brief_from_result(x) for x in reranked],
            "seeds": [brief_from_result(x) for x in seeds],
            "state": {
                "selection_ids": sorted(list(self.selection_set)),
                "frontier_ids": sorted(list(self.frontier_set))[:50],
            },
            "counts": {
                "n_nodes": self.hg.G.number_of_nodes(),
                "n_edges": self.hg.G.number_of_edges(),
            },
            "step": self.step_count,
        }
        if self.debug:
            obs["diagnostics"] = {
                "filters": self.current_filters,
                "weights": self.current_weights,
                "last_query": self.last_query,
                "frontier_size": len(self.frontier_set),
                "selection_size": len(self.selection_set),
            }
        return obs

    # ----------------------
    # Core API
    # ----------------------
    def reset(self, seed_query: Optional[str] = None, top_k: int = 5) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Initialize the environment. If seed_query is provided, perform semantic anchor.
        Returns (observation, info)
        """
        self._ensure_graph_loaded()
        self.last_query = None
        self.last_results = None
        self.step_count = 0
        self.selection_set.clear()
        self.frontier_set.clear()

        if seed_query:
            obs, info = self.semantic_anchor(seed_query, top_k=top_k)
            return obs, info
        # empty observation
        obs = {
            "selection": [],
            "seeds": [],
            "counts": {
                "n_nodes": self.hg.G.number_of_nodes(),
                "n_edges": self.hg.G.number_of_edges(),
            },
            "step": self.step_count,
        }
        self._log({"action": "reset", "message": "reset without seed_query"})
        return obs, {"message": "reset without seed_query"}

    def semantic_anchor(
        self,
        query: str,
        top_k: int = 5,
        member_top_m: int = 5,
        judge_overall_min: Optional[float] = None,
        confidence_min: Optional[float] = None,
        type_filter: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Call HierarchicalGraph.search and present a minimal observation.
        """
        self._ensure_graph_loaded()
        self.step_count += 1
        self.last_query = query

        t0 = time.perf_counter()
        params = {
            "top_k": top_k,
            "member_top_m": member_top_m,
            "judge_overall_min": self.current_filters.get("judge_overall_min") if judge_overall_min is None else judge_overall_min,
            "confidence_min": self.current_filters.get("confidence_min") if confidence_min is None else confidence_min,
            "type_filter": self.current_filters.get("type_filter") if type_filter is None else type_filter,
            "alpha": self.current_weights.get("alpha"),
            "beta": self.current_weights.get("beta"),
            "gamma": self.current_weights.get("gamma"),
            "delta": self.current_weights.get("delta"),
        }
        cluster = self.hg.search(
            query=query,
            top_k=params["top_k"],
            member_top_m=params["member_top_m"],
            judge_overall_min=params["judge_overall_min"],
            confidence_min=params["confidence_min"],
            type_filter=params["type_filter"],
            alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"], delta=params["delta"],
            return_cluster=True,
        )
        dur = time.perf_counter() - t0
        seeds = cluster.get("seeds", [])
        reranked = cluster.get("reranked", [])
        # update frontier with returned nodes
        self.frontier_set = set([x.get("node_id") for x in reranked if x.get("node_id")])
        obs = self._observation(seeds, reranked)
        info = {
            "action": "semantic_anchor",
            "query": query,
            "top_k": params["top_k"],
            "returned": len(reranked),
            "time_s": round(dur, 4),
        }
        self.last_results = cluster
        self._log({**info, "filters": self.current_filters, "weights": self.current_weights})
        # stats
        self.stats["actions"].append(info)
        self.stats["cumulative"]["steps"] += 1
        self.stats["cumulative"]["queries"] += 1
        self.stats["cumulative"]["time_s"] += dur
        return obs, info

    def expand_to_lca(self, node_ids: List[str], max_results: int = 5) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compute LCAs over belongs_to edges (DAG). Return brief nodes for LCAs.
        """
        self._ensure_graph_loaded()
        self.step_count += 1

        # Build belongs_to subgraph
        dag = nx.DiGraph()
        for u, v, data in self.hg.G.edges(data=True):
            if data.get("edge_type") == "belongs_to":
                dag.add_edge(u, v)

        # Fast guard: if not DAG, degrade to nearest common parents approx
        if not nx.is_directed_acyclic_graph(dag):
            # nearest common parents: all parents of each node, then intersect lowest by depth
            ancestors_sets = []
            for nid in node_ids:
                anc = nx.ancestors(dag, nid) | {nid}
                ancestors_sets.append(anc)
            inter = set.intersection(*ancestors_sets) if ancestors_sets else set()
            # choose nodes in intersection that have no successors inside intersection (lowest)
            lcas = []
            for n in inter:
                succ = set(dag.successors(n)) & inter
                if len(succ) == 0:
                    lcas.append(n)
        else:
            # DAG case
            ancestors_sets = []
            for nid in node_ids:
                anc = nx.ancestors(dag, nid) | {nid}
                ancestors_sets.append(anc)
            inter = set.intersection(*ancestors_sets) if ancestors_sets else set()
            lcas = []
            for n in inter:
                succ = set(dag.successors(n)) & inter
                if len(succ) == 0:
                    lcas.append(n)

        t0 = time.perf_counter()
        lcas_sorted = sorted(lcas, key=lambda x: (self.hg.G.nodes[x].get("level") or 1, x))[:max_results]
        seeds = [{"node_id": nid, "semantic": 0.0} for nid in lcas_sorted]
        reranked = [{"node_id": nid, "score": 0.0, "semantic": 0.0} for nid in lcas_sorted]
        obs = self._observation(seeds, reranked)
        info = {
            "action": "expand_to_lca",
            "inputs": node_ids,
            "lca_count": len(lcas_sorted),
            "dag": nx.is_directed_acyclic_graph(dag),
            "time_s": round(time.perf_counter() - t0, 4),
        }
        self._log(info)
        self.stats["actions"].append(info)
        self.stats["cumulative"]["steps"] += 1
        self.stats["cumulative"]["expansions"] += 1
        return obs, info

    def query_node_details(self, node_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Return detailed metadata for a node with trimmed text fields.
        """
        self._ensure_graph_loaded()
        self.step_count += 1

        if node_id not in self.hg.G:
            return {"selection": [], "seeds": [], "counts": {"n_nodes": self.hg.G.number_of_nodes(), "n_edges": self.hg.G.number_of_edges()}, "step": self.step_count}, {"error": "node_not_found", "node_id": node_id}

        d = self.hg.G.nodes[node_id]
        brief = self._node_brief(node_id)
        details: Dict[str, Any] = {
            **brief,
            "title": d.get("title"),
            "name": d.get("name"),
            "summary_text": (d.get("summary_text") or d.get("summary") or "")[:500],
            "description": (d.get("description") or "")[:500],
            "top_words": d.get("top_words"),
            "members": d.get("members"),
        }
        obs = {
            "selection": [details],
            "seeds": [],
            "counts": {
                "n_nodes": self.hg.G.number_of_nodes(),
                "n_edges": self.hg.G.number_of_edges(),
            },
            "step": self.step_count,
        }
        info = {"action": "query_node_details", "node_id": node_id}
        self._log(info)
        self.stats["actions"].append(info)
        self.stats["cumulative"]["steps"] += 1
        return obs, info

    # ----------------------
    # V2: State management & dynamic params
    # ----------------------
    def commit_selection(self, node_ids: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._ensure_graph_loaded()
        self.step_count += 1
        added = []
        for nid in node_ids:
            if nid in self.hg.G and nid not in self.selection_set:
                self.selection_set.add(nid)
                added.append(nid)
                if nid in self.frontier_set:
                    self.frontier_set.discard(nid)
        obs = self._observation([], [{"node_id": n, "score": 0.0, "semantic": 0.0} for n in added])
        info = {"action": "commit_selection", "added": added, "total_selection": len(self.selection_set)}
        self._log(info)
        self.stats["actions"].append(info)
        self.stats["cumulative"]["steps"] += 1
        return obs, info

    def set_filters(self, judge_overall_min: Optional[float] = None, confidence_min: Optional[float] = None,
                    type_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        if judge_overall_min is not None:
            self.current_filters["judge_overall_min"] = judge_overall_min
        if confidence_min is not None:
            self.current_filters["confidence_min"] = confidence_min
        if type_filter is not None:
            self.current_filters["type_filter"] = list(type_filter)
        info = {"action": "set_filters", **self.current_filters}
        self._log(info)
        self.stats["actions"].append(info)
        return info

    def set_search_weights(self, alpha: Optional[float] = None, beta: Optional[float] = None,
                            gamma: Optional[float] = None, delta: Optional[float] = None,
                            member_top_m: Optional[int] = None, top_k: Optional[int] = None) -> Dict[str, Any]:
        if alpha is not None:
            self.current_weights["alpha"] = alpha
        if beta is not None:
            self.current_weights["beta"] = beta
        if gamma is not None:
            self.current_weights["gamma"] = gamma
        if delta is not None:
            self.current_weights["delta"] = delta
        if member_top_m is not None:
            self.current_weights["member_top_m"] = member_top_m
        if top_k is not None:
            self.current_weights["top_k"] = top_k
        info = {"action": "set_search_weights", **self.current_weights}
        self._log(info)
        self.stats["actions"].append(info)
        return info

    # ----------------------
    # V2: Graph expansions
    # ----------------------
    def expand_children(self, node_ids: List[str], limit: int = 10) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._ensure_graph_loaded()
        self.step_count += 1
        expanded: List[str] = []
        for nid in node_ids:
            if nid not in self.hg.G:
                continue
            # children via belongs_to in-edge (members)
            for child in list(self.hg.G.predecessors(nid)):
                if self.hg.G.get_edge_data(child, nid).get("edge_type") == "belongs_to":
                    expanded.append(child)
            if len(expanded) >= limit:
                break
        expanded = list(dict.fromkeys(expanded))[:limit]
        seeds = [{"node_id": n, "semantic": 0.0} for n in expanded]
        obs = self._observation(seeds, [{"node_id": n, "score": 0.0, "semantic": 0.0} for n in expanded])
        info = {"action": "expand_children", "inputs": node_ids, "returned": len(expanded)}
        self.frontier_set.update(expanded)
        self._log(info)
        self.stats["actions"].append(info)
        self.stats["cumulative"]["steps"] += 1
        self.stats["cumulative"]["expansions"] += 1
        return obs, info

    def expand_parents(self, node_ids: List[str], limit: int = 10) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._ensure_graph_loaded()
        self.step_count += 1
        expanded: List[str] = []
        for nid in node_ids:
            if nid not in self.hg.G:
                continue
            for parent in list(self.hg.G.successors(nid)):
                if self.hg.G.get_edge_data(nid, parent).get("edge_type") == "belongs_to":
                    expanded.append(parent)
            if len(expanded) >= limit:
                break
        expanded = list(dict.fromkeys(expanded))[:limit]
        seeds = [{"node_id": n, "semantic": 0.0} for n in expanded]
        obs = self._observation(seeds, [{"node_id": n, "score": 0.0, "semantic": 0.0} for n in expanded])
        info = {"action": "expand_parents", "inputs": node_ids, "returned": len(expanded)}
        self.frontier_set.update(expanded)
        self._log(info)
        self.stats["actions"].append(info)
        self.stats["cumulative"]["steps"] += 1
        self.stats["cumulative"]["expansions"] += 1
        return obs, info

    def expand_related(self, node_ids: List[str], limit: int = 10) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._ensure_graph_loaded()
        self.step_count += 1
        expanded: List[str] = []
        for nid in node_ids:
            if nid not in self.hg.G:
                continue
            # consider both in/out edges with edge_type=related_to (for summaries)
            for _, v, data in self.hg.G.out_edges(nid, data=True):
                if data.get("edge_type") == "related_to":
                    expanded.append(v)
            for u, _, data in self.hg.G.in_edges(nid, data=True):
                if data.get("edge_type") == "related_to":
                    expanded.append(u)

            # ENHANCEMENT: Also explore hyperedge connections for entities
            node_data = self.hg.G.nodes.get(nid, {})
            if node_data.get("node_type") == "entity":
                # Find hyperedges this entity participates in
                for _, v, data in self.hg.G.out_edges(nid, data=True):
                    if data.get("edge_type") == "participates_in":
                        # Add the hyperedge
                        expanded.append(v)
                        # Also add other entities that participate in the same hyperedge
                        for u, _, edge_data in self.hg.G.in_edges(v, data=True):
                            if (edge_data.get("edge_type") == "participates_in" and
                                u != nid):  # Don't add the original entity
                                expanded.append(u)

            if len(expanded) >= limit:
                break
        expanded = list(dict.fromkeys(expanded))[:limit]
        seeds = [{"node_id": n, "semantic": 0.0} for n in expanded]
        obs = self._observation(seeds, [{"node_id": n, "score": 0.0, "semantic": 0.0} for n in expanded])
        info = {"action": "expand_related", "inputs": node_ids, "returned": len(expanded)}
        self.frontier_set.update(expanded)
        self._log(info)
        self.stats["actions"].append(info)
        self.stats["cumulative"]["steps"] += 1
        self.stats["cumulative"]["expansions"] += 1
        return obs, info

    # ----------------------
    # V3: Debug toggle & end episode
    # ----------------------
    def set_debug(self, enabled: bool = True) -> Dict[str, Any]:
        self.debug = enabled
        info = {"action": "set_debug", "debug": self.debug}
        self._log(info)
        self.stats["actions"].append(info)
        return info

    def end_episode(self) -> Dict[str, Any]:
        summary = {
            "session_id": self.session_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "selection_size": len(self.selection_set),
            "frontier_size": len(self.frontier_set),
            "stats": self.stats,
            "filters": self.current_filters,
            "weights": self.current_weights,
            "last_query": self.last_query,
        }
        try:
            with open(os.path.join(self.session_path, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        self._log({"action": "end_episode"})
        return summary


