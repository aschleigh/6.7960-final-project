#!/usr/bin/env python
"""
Main entry point for Week 2 experiments.

Usage:
    python run_week2.py
    python run_week2.py --quick  # Fast test with fewer samples
    python run_week2.py --concepts formal casual positive negative
"""

from experiments.week2_composition import main

if __name__ == "__main__":
    main()