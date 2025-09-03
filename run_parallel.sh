#!/bin/bash
# Script to run multiple Berghain Bouncer Algorithm instances in parallel

set -e  # Exit on any error

# Configuration
SCENARIO=${SCENARIO:-1}
PLAYER_ID=${PLAYER_ID:-"parallel_test"}
BASE_URL=${BASE_URL:-"https://berghain.challenges.listenlabs.ai"}
N=${N:-1000}
NUM_INSTANCES=${NUM_INSTANCES:-10}

echo "🚀 Starting $NUM_INSTANCES parallel instances..."
echo "📊 Scenario: $SCENARIO, Base URL: $BASE_URL, N: $N"
echo "================================================================"

# Function to run a single instance
run_instance() {
    local instance_id=$1
    local player_id="${PLAYER_ID}_instance_${instance_id}"

    echo "[Instance $instance_id] Starting with player ID: $player_id"

    # Run the command
    if python3 -m bba_cli.cli \
        --base-url "$BASE_URL" \
        --scenario "$SCENARIO" \
        --player-id "$player_id" \
        --N "$N" 2>&1; then

        echo "[Instance $instance_id] ✅ SUCCESS"
        return 0
    else
        echo "[Instance $instance_id] ❌ FAILED"
        return 1
    fi
}

# Export function for parallel execution
export -f run_instance
export SCENARIO PLAYER_ID BASE_URL N

# Create temporary directory for results
RESULTS_DIR="parallel_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR"

# Run instances in parallel using xargs
seq 0 $((NUM_INSTANCES-1)) | xargs -n 1 -P $NUM_INSTANCES -I {} bash -c 'run_instance "$@"' _ {}

echo ""
echo "================================================================"
echo "🎉 All instances completed!"
echo "📊 Check the results in directory: $RESULTS_DIR"

# Optional: Generate a summary
echo "📈 Generating summary..."
echo "Parallel execution completed at $(date)" > "$RESULTS_DIR/summary.txt"
echo "Instances: $NUM_INSTANCES" >> "$RESULTS_DIR/summary.txt"
echo "Scenario: $SCENARIO" >> "$RESULTS_DIR/summary.txt"
echo "Base URL: $BASE_URL" >> "$RESULTS_DIR/summary.txt"
echo "N: $N" >> "$RESULTS_DIR/summary.txt"

echo "✅ Done! Results saved to $RESULTS_DIR"
