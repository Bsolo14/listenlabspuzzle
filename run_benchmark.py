#!/usr/bin/env python3
"""
Benchmark script to run multiple games against the simulated API
and compare different algorithm configurations.
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def run_single_game(scenario: int, use_lp: bool = True, debug: bool = False, quiet: bool = False, neither_wiggle: bool = False) -> Tuple[bool, int]:
    """
    Run a single game and return (success, rejected_count)
    """
    cmd = [
        sys.executable, "-m", "bba_cli.cli",
        "--scenario", str(scenario),
        "--simulated"
    ]

    if not use_lp:
        cmd.append("--use-mw")
    if debug:
        cmd.append("--debug")
    if quiet:
        cmd.append("--quiet")  # Add quiet flag to reduce output
    if neither_wiggle:
        cmd.append("--neither-wiggle")  # Add neither-wiggle flag

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,  # 3 minute timeout (reduced from 5)
            cwd="/Users/barrowsolomon/listen_labs_puzzle"
        )

        # Parse the JSON output from the last line
        lines = result.stdout.strip().split('\n')
        json_line = None
        for line in reversed(lines):
            if line.strip().startswith('{'):
                json_line = line
                break

        if json_line:
            data = json.loads(json_line)
            status = data.get('status')
            rejected_count = data.get('rejectedCount', 0)
            success = status == 'completed'
            return success, rejected_count

    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
        print(f"Error running game: {e}")
        return False, 0

    return False, 0


def run_benchmark(scenario: int, num_runs: int = 100, configs: List[Dict] = None, quiet: bool = False) -> Dict:
    """
    Run benchmark with multiple configurations
    """
    if configs is None:
        configs = [
            {"name": "LP Solver", "use_lp": True, "neither_wiggle": False},
            {"name": "LP Solver + Neither Wiggle", "use_lp": True, "neither_wiggle": True}
        ]

    results = {}

    for config in configs:
        config_name = config["name"]
        use_lp = config.get("use_lp", False)
        debug = config.get("debug", False)
        config_neither_wiggle = config.get("neither_wiggle", False)

        print(f"\n{'='*60}")
        print(f"Running {num_runs} games with {config_name}")
        print(f"{'='*60}")

        successes = 0
        rejection_counts = []
        failed_games = 0

        start_time = time.time()

        for i in range(num_runs):
            if (i + 1) % 50 == 0:
                print(f"Progress: {i + 1}/{num_runs} games completed")

            success, rejected_count = run_single_game(scenario, use_lp, debug, quiet, config_neither_wiggle)

            if success:
                successes += 1
                rejection_counts.append(rejected_count)
            else:
                failed_games += 1

        elapsed_time = time.time() - start_time

        # Calculate statistics
        if rejection_counts:
            avg_rejections = sum(rejection_counts) / len(rejection_counts)
            min_rejections = min(rejection_counts)
            max_rejections = max(rejection_counts)
            median_rejections = sorted(rejection_counts)[len(rejection_counts) // 2]
        else:
            avg_rejections = min_rejections = max_rejections = median_rejections = 0

        success_rate = (successes / num_runs) * 100

        config_results = {
            "config_name": config_name,
            "num_runs": num_runs,
            "successes": successes,
            "failed_games": failed_games,
            "success_rate": success_rate,
            "avg_rejections": avg_rejections,
            "min_rejections": min_rejections,
            "max_rejections": max_rejections,
            "median_rejections": median_rejections,
            "total_time": elapsed_time,
            "avg_time_per_game": elapsed_time / num_runs,
            "rejection_counts": rejection_counts
        }

        results[config_name] = config_results

        print(f"\nResults for {config_name}:")
        print(f"  Success Rate: {success_rate:.1f}% ({successes}/{num_runs})")
        print(f"  Failed Games: {failed_games}")
        if rejection_counts:
            print(f"  Avg Rejections: {avg_rejections:.1f}")
            print(f"  Min Rejections: {min_rejections}")
            print(f"  Max Rejections: {max_rejections}")
            print(f"  Median Rejections: {median_rejections}")
        print(f"  Total Time: {elapsed_time:.1f}s")
        print(f"  Avg Time per Game: {elapsed_time/num_runs:.3f}s")

    return results


def save_results(results: Dict, scenario: int, output_file: str = None):
    """Save results to a JSON file"""
    if output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"benchmark_scenario_{scenario}_{timestamp}.json"

    # Remove the large rejection_counts array for cleaner output
    clean_results = {}
    for config_name, config_data in results.items():
        clean_results[config_name] = {k: v for k, v in config_data.items() if k != "rejection_counts"}

    with open(output_file, 'w') as f:
        json.dump(clean_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def print_comparison(results: Dict):
    """Print a comparison table of the results"""
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    print("<15")
    print("-" * 75)

    for config_name, data in results.items():
        success_rate = data["success_rate"]
        avg_rejections = data["avg_rejections"]
        median_rejections = data["median_rejections"]
        avg_time = data["avg_time_per_game"]

        print("<15")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Berghain Bouncer algorithms")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3], required=True,
                       help="Scenario number to test")
    parser.add_argument("--runs", type=int, default=500,
                       help="Number of runs per configuration (default: 500)")
    parser.add_argument("--output", type=str,
                       help="Output file for results (default: auto-generated)")
    parser.add_argument("--configs", type=str,
                       help="JSON string of configs to test (default: LP vs LP+NeitherWiggle)")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output from individual games (default: enabled)")

    args = parser.parse_args()

    # Parse custom configs if provided
    if args.configs:
        try:
            configs = json.loads(args.configs)
        except json.JSONDecodeError:
            print("Error: Invalid JSON in --configs")
            return
    else:
        configs = None

    print(f"Starting benchmark for scenario {args.scenario}")
    print(f"Running {args.runs} games per configuration...")

    results = run_benchmark(args.scenario, args.runs, configs, args.quiet)

    print_comparison(results)
    save_results(results, args.scenario, args.output)


if __name__ == "__main__":
    main()
