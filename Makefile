PY := python3

.PHONY: rl-gate rl-collect rl-bc rl-ppo rl-eval-bc rl-eval-ppo

rl-gate:
	$(PY) scripts/eval_gate.py --dataset hotpotqa --limit 5 --out reports/rl_gate.json

rl-collect:
	$(PY) scripts/collect_trajectories.py --dataset hotpotqa --limit 20 --out artifacts/rl/trajectories.jsonl

rl-bc:
	$(PY) scripts/train_bc.py --traj artifacts/rl/trajectories.jsonl --out artifacts/rl/bc_policy.pt --epochs 5

rl-ppo:
	$(PY) scripts/train_ppo.py --episodes 10 --max-steps 6 --out artifacts/rl/ppo_policy.pt

rl-eval-bc:
	$(PY) scripts/eval_rl_policy.py --dataset hotpotqa --limit 5 --bc-model artifacts/rl/bc_policy.pt --out artifacts/rl/eval_bc.json

rl-eval-ppo:
	$(PY) scripts/eval_rl_policy.py --dataset hotpotqa --limit 5 --ppo-model artifacts/rl/ppo_policy.pt --out artifacts/rl/eval_ppo.json

