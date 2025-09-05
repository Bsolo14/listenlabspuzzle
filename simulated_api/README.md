# Simulated Berghain API

This is a local simulation of the Berghain Challenge API for development and testing purposes.

## Features

- **Same API endpoints** as the real Berghain API
- **Correlated attribute generation** using Gaussian copula
- **Game state management** with proper constraint checking
- **Real-time simulation** of people arriving with binary attributes

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the simulated API server:
```bash
python run_server.py
```

The server will run on `http://localhost:5000` by default.

## Usage

Use the CLI with the `--simulated` flag to use the local API:

```bash
python -m bba_cli.cli --scenario 1 --simulated
```

## API Endpoints

### GET /new-game
Creates a new game with the specified scenario.

**Parameters:**
- `scenario` (int): Scenario ID (1, 2, or 3)
- `playerId` (str): Player identifier

**Response:** Same format as real API

### GET /decide-and-next
Gets the next person and optionally makes an accept/reject decision.

**Parameters:**
- `gameId` (str): Game ID from new-game response
- `personIndex` (int): Current person index
- `accept` (bool, optional): Accept/reject decision (required for personIndex > 0)

**Response:** Same format as real API

## Implementation Details

- **Person Generation**: Uses Gaussian copula to generate correlated binary attributes based on the scenario's relative frequencies and correlation matrix
- **Game State**: Tracks admitted/rejected counts and constraint satisfaction
- **Scenarios**: Uses data from `scenarios.json` in the parent directory
- **Completion**: Game ends when venue is full (1000 people) or rejection limit reached (20,000)

## Benchmarking

Use the benchmark scripts to compare algorithm performance:

### Quick Comparison (LP vs MW)
```bash
python compare_algorithms.py 1
```
*Runs 100 games per algorithm (~2-3 minutes)*

### Advanced Benchmarking
```bash
python run_benchmark.py --scenario 1 --runs 100
```
*Default: 100 runs per configuration*

### Custom Configurations
```bash
python run_benchmark.py --scenario 1 --configs '[{"name": "LP", "use_lp": true}, {"name": "MW", "use_lp": false}]'
```

## Algorithm Accuracy

**Important**: The simulated API and benchmark scripts maintain 100% algorithm accuracy:

- Uses identical sample sizes (150,000) for joint distribution computation
- Uses identical iteration counts (450) for multiplicative weights algorithm
- Uses exact same probability distributions and correlations as real API
- No optimizations that could affect algorithm behavior

**Current Limitations**:
- Scenario 2 data is missing from `scenarios.json` (only scenarios 1 and 3 available)
- Person generation uses Gaussian copula which may differ slightly from real API implementation