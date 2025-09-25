import argparse
import json
import os
from typing import List

import sys
from pathlib import Path

# Ensure src is on sys.path so that ah_rag package can be imported when running the script directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ah_rag.extract.hypergraph_extractor import HypergraphExtractor
from ah_rag.aggregate.semantic_aggregator import SemanticAggregator
from ah_rag.graph.hierarchical_graph import HierarchicalGraph

import tiktoken


def tokenize_len(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def smart_chunks(text: str, model_ctx: int = 8000, max_output: int = 1200, buffer: int = 400,
                 est_model: str = "gpt-4o-mini") -> List[str]:
    limit = model_ctx - max_output - buffer
    if tokenize_len(text, est_model) <= limit:
        return [text]
    # First split by blank lines
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0
    for p in parts:
        ptok = tokenize_len(p, est_model)
        if cur_tokens + ptok + 1 <= limit:
            cur.append(p)
            cur_tokens += ptok + 1
        else:
            if cur:
                chunks.append("\n\n".join(cur))
            # if single part too large, hard split by lines
            if ptok > limit:
                lines = p.splitlines()
                buf: List[str] = []
                buf_tok = 0
                for ln in lines:
                    lt = tokenize_len(ln, est_model)
                    if buf_tok + lt + 1 <= limit:
                        buf.append(ln)
                        buf_tok += lt + 1
                    else:
                        if buf:
                            chunks.append("\n".join(buf))
                        buf = [ln]
                        buf_tok = lt + 1
                if buf:
                    chunks.append("\n".join(buf))
                cur = []
                cur_tokens = 0
            else:
                cur = [p]
                cur_tokens = ptok + 1
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def run_pipeline(input_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        document = f.read()

    print("[1/5] Extracting L0 hyperedges...")
    extractor = HypergraphExtractor()

    chunks = smart_chunks(document)
    all_extractions = []
    for i, ch in enumerate(chunks):
        ex = extractor.extract(ch)
        if ex:
            all_extractions.extend(ex)
        else:
            print(f"  [warn] chunk {i} produced no extractions; skipped")

    if not all_extractions:
        print("[fatal] no valid extractions produced; aborting.")
        return None

    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/extractions.json', 'w', encoding='utf-8') as f:
        json.dump([e.model_dump() for e in all_extractions], f, ensure_ascii=False, indent=2)

    print("[2/5] Aggregating to L1 (embeddings, topics, summaries, relations, judge)...")
    aggregator = SemanticAggregator()
    aggregator.embed_l0_entities(all_extractions)
    clust = aggregator.cluster_with_bertopic()
    aggregator.summarize_topics(clust['l1_nodes'])
    aggregator.generate_l1_relations(clust['l1_nodes'], min_overlap=1, min_jaccard=0.05, min_cosine=0.3)
    # L2 community-based aggregation
    print("[2.5/5] Aggregating to L2 via communities...")
    aggregator.aggregate_level2_via_communities(clust['l1_nodes'])

    print("[3/5] Building unified graph...")
    hg = HierarchicalGraph()
    hg.build_from_artifacts('artifacts')
    hg.save('graph', meta={"source": os.path.abspath(input_path)})

    print("[4/5] Building vector index...")
    hg.build_vector_index(db_path='vector_db', layers={0, 1, 2}, reset=True)

    print("[5/5] Ready. You can now run interactive queries.")
    return hg


def interactive_search(hg: HierarchicalGraph):
    print("Enter your queries (empty line to exit):")
    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        results = hg.search(q, top_k=5)
        print(json.dumps(results, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Ingest a document and run hybrid search.")
    parser.add_argument("path", help="Path to a UTF-8 text file")
    args = parser.parse_args()
    hg = run_pipeline(args.path)
    if hg is None:
        print("Pipeline failed. See warnings above.")
        return
    interactive_search(hg)


if __name__ == "__main__":
    main()
