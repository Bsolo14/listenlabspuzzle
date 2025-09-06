#!/usr/bin/env python3
"""
Convenience script to run the CLI with simulated API
"""

import subprocess
import sys
import os

def main():
    # Change to the project directory
    os.chdir(os.path.dirname(__file__))

    # Run the CLI with simulated flag
    cmd = [sys.executable, "-m", "bba_cli.cli", "--simulated"] + sys.argv[1:]
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
