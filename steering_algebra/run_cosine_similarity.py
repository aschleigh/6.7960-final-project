#!/usr/bin/env python
"""
Run cosine similarity analysis.

Usage:
    python run_cosine_similarity.py
    python run_cosine_similarity.py --quick --max_pairs 10
    python run_cosine_similarity.py --concepts formal casual positive negative
"""

from experiments.cosine_similarity_analysis import main

if __name__ == "__main__":
    main()