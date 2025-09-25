#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def run_bench(dataset: str, limit: int, out_path: str, judge_sample: float = 0.5) -> None:
    exe = sys.executable or 'python3'
    cmd = [
        exe, 'scripts/run_benchmark.py',
        '--dataset', dataset,
        '--system', 'ah_rag',
        '--limit', str(limit),
        '--judge-sample', str(judge_sample),
        '--out', out_path,
    ]
    res = subprocess.run(cmd, cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    if res.returncode != 0:
        raise SystemExit('run_benchmark failed')


def main() -> None:
    ap = argparse.ArgumentParser(description='Evaluation gate according to Phase 4 thresholds')
    ap.add_argument('--dataset', type=str, default='hotpotqa')
    ap.add_argument('--limit', type=int, default=5)
    ap.add_argument('--out', type=str, default='reports/rl_gate.json')
    ap.add_argument('--f1-min', type=float, default=0.55, help='Minimum aggregate F1 to pass')
    ap.add_argument('--faith-min', type=float, default=0.6, help='Minimum aggregate faithfulness to pass')
    ap.add_argument('--latency-p95-max', type=float, default=None, help='Optional latency bound (if available)')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    run_bench(args.dataset, args.limit, args.out, judge_sample=0.5)

    with open(args.out, 'r', encoding='utf-8') as f:
        rep = json.load(f)
    agg = rep.get('aggregate', [{}])[0]
    f1 = float(agg.get('f1', 0.0))
    faith = float(agg.get('faithfulness', 0.0))
    passed = (f1 >= args.f1_min) and (faith >= args.faith_min)
    # Latency gate could be added when stats available in report

    print(json.dumps({
        'f1': f1,
        'faithfulness': faith,
        'passed': passed
    }, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

