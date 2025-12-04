"""
Configuration and constants for steering vector experiments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

# @dataclass
# class ModelConfig:
#     """Model configuration."""
#     # name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
#     name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
#     device: str = "cuda"
#     torch_dtype: str = "bfloat16"
#     # Layers to extract steering vectors from (middle layers typically work best)
#     steering_layers: List[int] = field(default_factory=lambda: [12, 14, 16, 18, 20])
#     # Default layer for steering
#     default_layer: int = 16
@dataclass
class ModelConfig:
    """Model configuration."""
    # TinyLlama for development/testing
    
    name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    # TinyLlama has 22 layers, so adjust layer indices
    steering_layers: List[int] = field(default_factory=lambda: [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    # Default to middle-ish layer
    default_layer: int = 12
    optimal_layers: Dict[str, List[int]] = field(default_factory=dict)
    coefficient_candidates = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    optimal_coefficient = {}
    default_coefficient = 1.0

    


@dataclass
class ExtractionConfig:
    """Configuration for steering vector extraction."""
    # Number of contrastive pairs per concept
    n_pairs: int = 100
    # Batch size for processing pairs
    batch_size: int = 8
    # Token position to extract activations from (-1 = last token)
    token_position: int = -1
    # Whether to normalize steering vectors
    normalize: bool = True


@dataclass 
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    # Number of generations per prompt for evaluation
    n_generations: int = 5


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    # Threshold for classifier "success"
    classifier_threshold: float = 0.5
    # Number of samples for human evaluation
    n_human_eval_samples: int = 30
    


@dataclass
class Config:
    """Master configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Paths
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path("cache")
    
    # Concepts to study
    concepts: List[str] = field(default_factory=lambda: [
        "formal", "casual",
        "positive", "negative",
        "verbose", "concise",
        "confident", "uncertain",
        "technical", "simple",
    ])

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)


# Global config instance
cfg = Config()