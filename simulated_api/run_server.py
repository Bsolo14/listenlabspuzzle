#!/usr/bin/env python3
"""
Script to run the simulated API server for local development.
"""

import sys
import os

# Add parent directory to Python path so we can import bba_cli
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import SimulatedAPIServer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Simulated API Server")
    parser.add_argument("--quiet", action="store_true", help="Reduce server output and logging")
    args = parser.parse_args()

    server = SimulatedAPIServer(quiet=args.quiet)
    server.run()


if __name__ == '__main__':
    main()
