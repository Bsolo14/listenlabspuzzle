#!/usr/bin/env python3
"""
Script to run multiple instances of the Berghain Bouncer Algorithm in parallel.
"""
import os
import subprocess
import sys
import time
import argparse
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import json


def run_single_instance(instance_id: int, scenario: int, player_id: str, base_url: str, N: int = 1000) -> Dict[str, Any]:
    """
    Run a single instance of the algorithm and return the results.
    """
    try:
        # Use the same player ID for all instances (server may allow multiple games per player)
        # Add a small delay to avoid overwhelming the server
        import time
        time.sleep(instance_id * 0.1)  # Stagger requests by 100ms
        unique_player_id = player_id

        cmd = [
            "/home/ec2-user/venv/bin/python3", "-m", "bba_cli.cli",
            "--base-url", base_url,
            "--scenario", str(scenario),
            "--player-id", unique_player_id,
            "--N", str(N)
        ]

        print(f"[Instance {instance_id}] Starting: {' '.join(cmd)}")

        # Set up environment for subprocess to inherit virtual environment
        env = os.environ.copy()
        venv_bin = "/home/ec2-user/venv/bin"
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = "/home/ec2-user/venv"
        env["PYTHONHOME"] = ""

        # Run the command and capture output
        result = subprocess.run(
            cmd,
            cwd="/home/ec2-user/listen_labs_puzzle",
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per instance
            env=env
        )

        output_lines = result.stdout.strip().split('\n')
        error_lines = result.stderr.strip().split('\n') if result.stderr else []

        # Try to parse the final JSON result
        final_result = None
        for line in reversed(output_lines):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    final_result = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

        return {
            "instance_id": instance_id,
            "player_id": unique_player_id,
            "original_player_id": player_id,  # Keep track of original for reference
            "return_code": result.returncode,
            "stdout": output_lines,
            "stderr": error_lines,
            "result": final_result,
            "success": result.returncode == 0 and final_result is not None
        }

    except subprocess.TimeoutExpired:
        return {
            "instance_id": instance_id,
            "player_id": f"timeout_{instance_id}",
            "original_player_id": player_id,
            "error": "Timeout after 5 minutes",
            "success": False
        }
    except Exception as e:
        return {
            "instance_id": instance_id,
            "player_id": f"error_{instance_id}",
            "original_player_id": player_id,
            "error": str(e),
            "success": False
        }


def run_parallel_instances(num_instances: int, scenario: int, player_id: str, base_url: str, N: int = 1000) -> List[Dict[str, Any]]:
    """
    Run multiple instances in parallel using ThreadPoolExecutor.
    """
    results = []

    print(f"🚀 Starting {num_instances} parallel instances...")
    print(f"📊 Scenario: {scenario}, Base URL: {base_url}, N: {N}")
    print(f"👤 Using player ID: {player_id} (same for all instances)")
    print("=" * 60)

    start_time = time.time()

    # Use ThreadPoolExecutor for concurrent execution
    with ThreadPoolExecutor(max_workers=min(num_instances, 10)) as executor:
        # Submit all tasks
        future_to_instance = {
            executor.submit(run_single_instance, i, scenario, player_id, base_url, N): i
            for i in range(num_instances)
        }

        # Process completed tasks as they finish
        for future in as_completed(future_to_instance):
            instance_id = future_to_instance[future]
            try:
                result = future.result()
                results.append(result)

                # Print immediate feedback
                if result.get("success"):
                    final_result = result.get("result", {})
                    status = final_result.get("status", "unknown")
                    rejected = final_result.get("rejectedCount", "unknown")
                    print(f"[Instance {instance_id}] ✅ SUCCESS - Status: {status}, Rejected: {rejected}")
                else:
                    # Show detailed error information
                    return_code = result.get("return_code", "unknown")
                    stderr = result.get("stderr", [])
                    stdout = result.get("stdout", [])

                    print(f"[Instance {instance_id}] ❌ FAILED - Return code: {return_code}")
                    if stderr and len(stderr) > 0 and stderr[0].strip():
                        print(f"    STDERR: {stderr[-1] if stderr else 'No stderr'}")
                    if stdout and len(stdout) > 0 and stdout[-1].strip():
                        print(f"    STDOUT: {stdout[-1] if stdout else 'No stdout'}")

            except Exception as exc:
                print(f"[Instance {instance_id}] ❌ EXCEPTION - {exc}")
                results.append({
                    "instance_id": instance_id,
                    "player_id": f"exception_{instance_id}",
                    "original_player_id": player_id,
                    "error": str(exc),
                    "success": False
                })

    # Sort results by instance ID
    results.sort(key=lambda x: x["instance_id"])

    # Calculate statistics
    total_time = time.time() - start_time
    successful_runs = sum(1 for r in results if r.get("success"))
    success_rate = successful_runs / num_instances * 100

    print("\n" + "=" * 60)
    print("📈 FINAL RESULTS SUMMARY")
    print(f"Total instances: {num_instances}")
    print(f"Successful runs: {successful_runs}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Success rate: {success_rate:.2f}%")
    # Show detailed results for each instance
    print("\n📋 DETAILED RESULTS:")
    for result in results:
        instance_id = result["instance_id"]
        if result.get("success"):
            final_result = result.get("result", {})
            status = final_result.get("status", "unknown")
            rejected = final_result.get("rejectedCount", "unknown")
            print(f"  Instance {instance_id}: ✅ {status} (rejected: {rejected})")
        else:
            return_code = result.get("return_code", "unknown")
            stderr = result.get("stderr", [])
            error_msg = f"Return code: {return_code}"
            if stderr and len(stderr) > 0 and stderr[0].strip():
                error_msg += f" | Last stderr: {stderr[-1][:50]}..."
            print(f"  Instance {instance_id}: ❌ {error_msg}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run multiple Berghain Bouncer Algorithm instances in parallel")
    parser.add_argument("--instances", type=int, default=10, help="Number of parallel instances to run")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--player-id", type=str, required=True)
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--N", type=int, default=1000, help="Population size")

    args = parser.parse_args()

    # Ensure we're in the virtual environment
    print("🔧 Activating virtual environment...")
    print("📂 Working directory:", "/home/ec2-user/listen_labs_puzzle")

    # Run the parallel instances
    results = run_parallel_instances(
        num_instances=args.instances,
        scenario=args.scenario,
        player_id=args.player_id,
        base_url=args.base_url,
        N=args.N
    )

    # Save results to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"parallel_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    return 0 if all(r.get("success", False) for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
