Berghain Bouncer Algorithm (BBA) CLI

Usage

- Create and activate venv (optional):
  python3 -m venv .venv
  source .venv/bin/activate

- Install deps:
  pip install -r requirements.txt

- Run:
  python -m bba_cli --scenario 1 --player-id YOUR_UUID --simulated  # embedded simulator (no server)

- Run with debug info:
  python -m bba_cli --scenario 1 --player-id YOUR_UUID --simulated  # embedded simulator (no server) --debug

Description

This CLI implements the strategy described in `stratey.md` and plays the game via the API documented in `api.md`.

Features

- Joint distribution approximation from marginals + correlations via Gaussian copula
- MW-LP optimization for base acceptance probabilities (prioritizes high-leverage types)
- Online policy with deficit-aware nudging and hard endgame guard
- Comprehensive feasibility analysis with expected rejection estimates
- Detailed diagnostics showing constraint analysis and top accepted types
- Debug mode to inspect joint distribution and optimization behavior
- Graceful handling of potentially infeasible scenarios



Note: Use --simulated to run against the embedded simulator (no local server needed).
