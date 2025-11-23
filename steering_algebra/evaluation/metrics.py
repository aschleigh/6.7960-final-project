"""
Additional evaluation metrics: perplexity, diversity, fluency.
"""

import torch
from torch import Tensor
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import numpy as np
import math


class PerplexityCalculator:
    """
    Calculate perplexity of generated text using a reference model.
    Lower perplexity = more fluent/natural text.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2-large",
        device: str = "cuda"
    ):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # GPT-2 doesn't have a pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    # def compute(self, text: str) -> float:
    #     """
    #     Compute perplexity for a single text.
        
    #     Returns:
    #         Perplexity score (lower = more fluent)
    #     """
    #     inputs = self.tokenizer(
    #         text,
    #         return_tensors="pt",
    #         truncation=True,
    #         max_length=1024
    #     ).to(self.device)
        
    #     with torch.no_grad():
    #         outputs = self.model(**inputs, labels=inputs["input_ids"])
    #         loss = outputs.loss
        
    #     return math.exp(loss.item())
    
    # def compute_batch(self, texts: List[str]) -> List[float]:
    #     """Compute perplexity for multiple texts."""
    #     return [self.compute(text) for text in texts]
    def compute(self, text: str) -> float:
        """
        Compute perplexity for a single text.
        
        Returns:
            Perplexity score (lower = more fluent)
        """
        # Handle empty or very short text
        if not text or len(text.strip()) < 5:
            return float('inf')  # Return infinity for invalid text
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # Check if tokenization produced valid input
            if inputs["input_ids"].shape[1] == 0:
                return float('inf')
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
            
            return math.exp(loss.item())
        
        except Exception as e:
            # If perplexity computation fails, return a large value
            print(f"Warning: Perplexity computation failed for text: '{text[:50]}...' Error: {e}")
            return float('inf')


    def compute_batch(self, texts: List[str]) -> List[float]:
        """Compute perplexity for multiple texts."""
        perplexities = []
        for text in texts:
            ppl = self.compute(text)
            # Cap extremely high perplexities for stability
            perplexities.append(min(ppl, 10000.0))
        return perplexities


def compute_distinct_n(texts: List[str], n: int = 2) -> float:
    """
    Compute Distinct-n metric: fraction of unique n-grams.
    Higher = more diverse text.
    
    Args:
        texts: List of generated texts
        n: N-gram size (typically 1 or 2)
        
    Returns:
        Distinct-n score in [0, 1]
    """
    all_ngrams = []
    
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    
    if len(all_ngrams) == 0:
        return 0.0
    
    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    
    return unique_ngrams / total_ngrams


def compute_self_bleu(texts: List[str], n: int = 4) -> float:
    """
    Compute Self-BLEU: average BLEU of each text against all others.
    Lower = more diverse (each text is different from others).
    
    Note: Requires nltk.translate.bleu_score
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        raise ImportError("Install nltk: pip install nltk")
    
    if len(texts) < 2:
        return 0.0
    
    smoothing = SmoothingFunction().method1
    scores = []
    
    for i, text in enumerate(texts):
        hypothesis = text.lower().split()
        references = [t.lower().split() for j, t in enumerate(texts) if j != i]
        
        if len(hypothesis) < n:
            continue
            
        score = sentence_bleu(
            references,
            hypothesis,
            weights=[1/n] * n,
            smoothing_function=smoothing
        )
        scores.append(score)
    
    return np.mean(scores) if scores else 0.0


def compute_repetition_ratio(text: str, n: int = 3) -> float:
    """
    Compute ratio of repeated n-grams in a single text.
    Lower = less repetitive.
    """
    tokens = text.lower().split()
    if len(tokens) < n:
        return 0.0
    
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    ngram_counts = Counter(ngrams)
    
    repeated = sum(count - 1 for count in ngram_counts.values() if count > 1)
    total = len(ngrams)
    
    return repeated / total if total > 0 else 0.0


def compute_length_stats(texts: List[str]) -> Dict:
    """Compute statistics about text lengths."""
    lengths = [len(text.split()) for text in texts]
    
    return {
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "min_length": np.min(lengths),
        "max_length": np.max(lengths),
    }


class QualityMetrics:
    """
    Comprehensive quality evaluation combining multiple metrics.
    """
    
    def __init__(self, device: str = "cuda"):
        self.perplexity_calc = PerplexityCalculator(device=device)
    
    def evaluate(self, texts: List[str]) -> Dict:
        """
        Compute all quality metrics for a set of generated texts.
        
        Returns:
            Dict with all metrics
        """
        # Filter out empty or very short texts
        valid_texts = [t for t in texts if t and len(t.strip()) > 5]
        
        if len(valid_texts) == 0:
            # All texts are invalid - return placeholder metrics
            return {
                "perplexity_mean": float('inf'),
                "perplexity_std": 0.0,
                "distinct_1": 0.0,
                "distinct_2": 0.0,
                "self_bleu": 0.0,
                "repetition_mean": 0.0,
                "mean_length": 0.0,
                "std_length": 0.0,
                "min_length": 0,
                "max_length": 0,
                "n_valid_texts": 0,
                "n_total_texts": len(texts)
            }
    
        # Perplexity
        perplexities = self.perplexity_calc.compute_batch(valid_texts)
        
        # Diversity
        distinct_1 = compute_distinct_n(valid_texts, n=1)
        distinct_2 = compute_distinct_n(valid_texts, n=2)
        
        # Self-BLEU (diversity across samples)
        try:
            self_bleu = compute_self_bleu(valid_texts) if len(valid_texts) > 1 else 0.0
        except Exception:
            self_bleu = 0.0
        
        # Repetition (within each text)
        repetition_ratios = [compute_repetition_ratio(t) for t in valid_texts]
        
        # Length
        length_stats = compute_length_stats(valid_texts)
        
        return {
            "perplexity_mean": np.mean(perplexities),
            "perplexity_std": np.std(perplexities),
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "self_bleu": self_bleu,
            "repetition_mean": np.mean(repetition_ratios),
            "n_valid_texts": len(valid_texts),
            "n_total_texts": len(texts),
            **length_stats
        }
        
    def compare(
        self,
        baseline_texts: List[str],
        steered_texts: List[str]
    ) -> Dict:
        """
        Compare quality metrics between baseline and steered generations.
        """
        baseline_metrics = self.evaluate(baseline_texts)
        steered_metrics = self.evaluate(steered_texts)
        
        comparison = {}
        for key in baseline_metrics:
            comparison[f"baseline_{key}"] = baseline_metrics[key]
            comparison[f"steered_{key}"] = steered_metrics[key]
            comparison[f"delta_{key}"] = steered_metrics[key] - baseline_metrics[key]
        
        return comparison