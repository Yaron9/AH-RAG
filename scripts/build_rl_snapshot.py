#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load_items(dataset: str, split: str, limit: int, seed: int) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Please install 'datasets' (pip install datasets)") from e

    if dataset.lower() == 'hotpotqa':
        ds = load_dataset('hotpot_qa', 'distractor')[split]
        items: List[Dict[str, Any]] = []
        for i in ds:
            ctx = i.get('context')
            if not ctx:
                continue
            items.append({
                'id': str(i.get('_id') or i.get('id') or ''),
                'question': i.get('question') or '',
                'context': ctx,
            })
    elif dataset.lower() == 'triviaqa':
        ds = load_dataset('trivia_qa', 'rc')[split]
        items = []
        for i in ds:
            items.append({
                'id': str(i.get('question_id') or i.get('id') or ''),
                'question': i.get('question') or '',
                'context': None,  # TriviaQA doesn't ship per-question wiki context in the same way
            })
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    random.seed(seed)
    random.shuffle(items)
    return items[:limit]


def build_corpus_from_hotpot(items: List[Dict[str, Any]]) -> str:
    buf: List[str] = []
    for it in items:
        ctx = it.get('context') or {}
        titles = ctx.get('title') or []
        sentences = ctx.get('sentences') or []
        for t, sens in zip(titles, sentences):
            buf.append(f"\n\n=== {t} ===\n")
            for s in sens:
                if s:
                    buf.append(s.strip() + ' ')
    return ''.join(buf).strip()


def run_demo_ingest(corpus_path: str) -> None:
    exe = sys.executable or 'python3'
    res = subprocess.run([exe, 'scripts/demo_cli.py', corpus_path], cwd=ROOT)
    if res.returncode != 0:
        raise RuntimeError('demo_cli.py failed')


def relocate_outputs(graph_out: str, vdb_out: str, reset: bool = False) -> None:
    g_src = os.path.join(ROOT, 'graph')
    v_src = os.path.join(ROOT, 'vector_db')
    g_dst = os.path.join(ROOT, graph_out)
    v_dst = os.path.join(ROOT, vdb_out)
    if reset:
        shutil.rmtree(g_dst, ignore_errors=True)
        shutil.rmtree(v_dst, ignore_errors=True)
    if os.path.exists(g_dst) or os.path.exists(v_dst):
        raise RuntimeError(f'Target exists: {g_dst} or {v_dst}. Use --reset to overwrite.')
    shutil.move(g_src, g_dst)
    shutil.move(v_src, v_dst)

    # Patch vector_index path inside meta.json
    meta_path = os.path.join(g_dst, 'meta.json')
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        vi = meta.get('vector_index') or {}
        vi['db_path'] = vdb_out
        meta['vector_index'] = vi
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description='Build a larger RL training snapshot (graph/vector_db) in Colab or locally')
    ap.add_argument('--dataset', type=str, default='hotpotqa', help='hotpotqa|triviaqa')
    ap.add_argument('--split', type=str, default='validation')
    ap.add_argument('--samples', type=int, default=300, help='number of questions to aggregate context from')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--graph-out', type=str, default='graph_rl', help='output dir for graph snapshot')
    ap.add_argument('--vdb-out', type=str, default='vector_db_rl', help='output dir for vector index')
    ap.add_argument('--reset', action='store_true', help='overwrite existing outputs')
    args = ap.parse_args()

    print(f'[1/4] Loading dataset {args.dataset}:{args.split} (samples={args.samples})')
    items = load_items(args.dataset, args.split, args.samples, args.seed)
    if args.dataset.lower() != 'hotpotqa':
        raise RuntimeError('This builder currently supports HotpotQA with per-question context aggregation.')

    print('[2/4] Building aggregated corpus from HotpotQA context...')
    corpus = build_corpus_from_hotpot(items)
    if not corpus:
        raise RuntimeError('No context collected from dataset')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tf:
        tf.write(corpus)
        corpus_path = tf.name
    print(f'  Corpus saved to {corpus_path} (size={len(corpus)} chars)')

    try:
        print('[3/4] Running demo_cli.py to extract/aggregate/build graph...')
        run_demo_ingest(corpus_path)
        print('[4/4] Relocating outputs...')
        relocate_outputs(args.graph_out, args.vdb_out, reset=bool(args.reset))
        print(f'Done. Graph: {args.graph_out} | Vector DB: {args.vdb_out}')
        print('Tip: Configure training to use this snapshot (e.g., rl.train.graph_dir).')
    finally:
        try:
            os.unlink(corpus_path)
        except Exception:
            pass


if __name__ == '__main__':
    main()

