#!/usr/bin/env python3
"""
Quick script to compare LP vs Multiplicative Weights algorithms
"""

import subprocess
import sys
import json
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print("Usage: python compare_algorithms.py <scenario>")
        print("Example: python compare_algorithms.py 1")
        sys.exit(1)

    scenario = int(sys.argv[1])
    if scenario not in [1, 2, 3]:
        print("Error: Scenario must be 1, 2, or 3")
        sys.exit(1)

    print(f"Running benchmark for scenario {scenario}...")
    print("This will run 100 games with Multiplicative Weights and 100 with LP solver")
    print("Estimated time: ~2-3 minutes\n")

    # Run the benchmark
    cmd = [
        sys.executable, "run_benchmark.py",
        "--scenario", str(scenario),
        "--runs", "100"
    ]

    result = subprocess.run(cmd, cwd="/Users/barrowsolomon/listen_labs_puzzle")

    if result.returncode == 0:
        print("\nBenchmark completed successfully!")
        print("Check the generated JSON file for detailed results.")
    else:
        print("\nBenchmark failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
