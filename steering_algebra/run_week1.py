#!/usr/bin/env python
"""
Main entry point for Week 1 experiments.

Usage:
    python run_week1.py --model meta-llama/Meta-Llama-3-8B-Instruct
    OR
    python run_week1.py --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
    python run_week1.py --concepts formal positive technical --skip_extraction
"""

from experiments.validate_single import main

if __name__ == "__main__":
    main()