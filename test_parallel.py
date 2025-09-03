#!/usr/bin/env python3
"""
Test script to verify parallel implementations work correctly.
"""
import sys
import os

# Add the current directory to the path so we can import bba_cli
sys.path.insert(0, os.path.dirname(__file__))

from bba_cli.strategy import test_parallel_functionality

if __name__ == "__main__":
    print("Running parallel functionality tests...")
    success = test_parallel_functionality()

    if success:
        print("\n🎉 All tests passed! Parallel implementations are ready to use.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed. Please check the implementation.")
        sys.exit(1)
