"""
Shared utilities for the steering vectors project.
"""

import torch
import random
import numpy as np
from pathlib import Path
from typing import Any, Dict
import json


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(data: Dict, path: Path):
    """Save dictionary to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Dict:
    """Load dictionary from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def get_device() -> str:
    """Get available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_number(n: float, decimals: int = 3) -> str:
    """Format a number for display."""
    if abs(n) < 0.001:
        return f"{n:.2e}"
    return f"{n:.{decimals}f}"


def print_table(headers: list, rows: list, col_width: int = 12):
    """Print a formatted table."""
    header_str = " | ".join(f"{h:>{col_width}}" for h in headers)
    print(header_str)
    print("-" * len(header_str))
    for row in rows:
        row_str = " | ".join(f"{str(v):>{col_width}}" for v in row)
        print(row_str)

def format_prompt_for_model(
    prompt: str,
    tokenizer,
    model_name: str
) -> str:
    """
    Format a prompt appropriately for the model's expected format.
    """
    # Check if model uses chat template
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            pass
    
    # Fallback: return raw prompt
    return prompt


class ResultsLogger:
    """Logger for experiment results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def log(self, key: str, value: Any):
        """Log a single value."""
        self.results[key] = value
    
    def log_dict(self, d: Dict):
        """Log a dictionary of values."""
        self.results.update(d)
    
    def save(self, filename: str = "results.json"):
        """Save results to file."""
        save_json(self.results, self.output_dir / filename)
    
    def print_summary(self):
        """Print a summary of results."""
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        for key, value in self.results.items():
            if isinstance(value, float):
                print(f"{key}: {format_number(value)}")
            else:
                print(f"{key}: {value}")