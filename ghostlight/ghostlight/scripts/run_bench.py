"""Run WornBench and save results to bench_results.json.

    python scripts/run_bench.py --stones 40
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ghostlight import bench

ap = argparse.ArgumentParser()
ap.add_argument("--stones", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
res = bench.run(wears=(0.2, 0.4, 0.6, 0.8), n_stones=a.stones, seed=a.seed)
print(json.dumps(res, indent=2))
with open("bench_results.json", "w") as f:
    json.dump(res, f, indent=2)
