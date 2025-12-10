# """
# Configuration and constants for steering vector experiments.
# """

# from dataclasses import dataclass, field
# from typing import List, Dict, Optional
# from pathlib import Path

# @dataclass
# class ModelConfig:
#     """Model configuration."""
#     name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#     device: str = "cuda"
#     torch_dtype: str = "bfloat16"
    
#     # 1.1B models often work best at middle-late layers for style
#     steering_layers: List[int] = field(default_factory=lambda: [10, 11, 12, 13, 14, 15, 16])
    
#     # Fixed target layer for experiments
#     default_layer: int = 16 
    
#     # Per-concept tuning (Tone = Low, Content = High)
#     concept_coefficients: Dict[str, float] = field(default_factory=lambda: {
#         "positive": 0.5,
#         "negative": 0.5,
#         "formal": 0.5,
#         "slang": 0.5,
#         "fantasy": 2.0,
#         "science": 2.0,
#         "technical": 1.5,
#         "simple": 1.5,
#         "confident": 1.0,
#         "uncertain": 1.0,
#         # SYNONYMS (Standard strength)
#         "smart": 1.0,
#         "intelligent": 1.0,
#         "unhappy": 0.8,
#         "sad": 0.8,
#         "angry": 1.0,
#         "fearful": 1.0,
#         "furious": 1.0, 
#         "scared": 1.0,
#     })
    
#     default_coefficient: float = 1.0

# @dataclass
# class ExtractionConfig:
#     n_pairs: int = 100
#     batch_size: int = 8
#     token_position: int = -1
#     normalize: bool = True

# @dataclass 
# class GenerationConfig:
#     max_new_tokens: int = 128
#     temperature: float = 0.7
#     top_p: float = 0.9
#     do_sample: bool = True
#     n_generations: int = 5

# @dataclass
# class EvaluationConfig:
#     classifier_threshold: float = 0.6
#     n_human_eval_samples: int = 30

# @dataclass
# class Config:
#     """Master configuration."""
#     model: ModelConfig = field(default_factory=ModelConfig)
#     extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
#     generation: GenerationConfig = field(default_factory=GenerationConfig)
#     evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
#     # Paths
#     output_dir: Path = Path("outputs")
#     cache_dir: Path = Path("cache")
    
#     # Concepts to study
#     # Includes Synonyms (Smart/Intelligent) and Fixes (Slang instead of Casual)
#     concepts: List[str] = field(default_factory=lambda: [
#         "formal", "slang",
#         "fantasy", "science",
#         "positive", "negative",
#         "confident", "uncertain",
#         "technical", "simple",
#         "smart", "intelligent", # Synonym Pair 1
#         "unhappy", "sad",        # Synonym Pair 2
#         "angry", "furious",
#         "scared", "fearful", 
#     ])

#     def __post_init__(self):
#         self.output_dir.mkdir(exist_ok=True)
#         self.cache_dir.mkdir(exist_ok=True)

# cfg = Config()


"""
Configuration and constants for steering vector experiments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class ModelConfig:
    """Model configuration."""
    # UPGRADE: Qwen2.5-3B is SOTA for this size and is NOT gated (no access token needed)
    name: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    
    # Qwen2.5-3B has 36 layers.
    # We target the middle-late range where style/persona lives.
    steering_layers: List[int] = field(default_factory=lambda: [14, 16, 18, 20, 22, 24, 26])
    
    # Target Layer 20 (Middle of the stack)
    default_layer: int = 20
    
    # Per-concept tuning
    concept_coefficients: Dict[str, float] = field(default_factory=lambda: {
        # Tone/Style: These were too weak (0.0 interference).
        # Crank them up to 2.0 - 2.5 to force a clash.
        "positive": 2.0, "negative": 2.0,
        "formal": 2.0, "slang": 2.0,
        "confident": 2.0, "uncertain": 2.0,
        "technical": 2.0, "simple": 2.0,
        "smart": 2.0, "intelligent": 2.0,
        
        # Emotions: Increase to 1.8
        "angry": 1.8, "furious": 1.8,
        "scared": 1.8, "fearful": 1.8,
        "unhappy": 1.5, "sad": 1.5,
        
        # Content: These are sparse, so they need massive force
        "fantasy": 3.0, "science": 3.0,
    })
    
    default_coefficient: float = 1.0

@dataclass
class ExtractionConfig:
    # UPGRADE: More data = cleaner vectors
    n_pairs: int = 300 
    batch_size: int = 4
    token_position: int = -1
    normalize: bool = True

@dataclass 
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    n_generations: int = 5

@dataclass
class EvaluationConfig:
    classifier_threshold: float = 0.6
    n_human_eval_samples: int = 30

@dataclass
class Config:
    """Master configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path("cache")
    
    # Full Concept List (18 Concepts)
    concepts: List[str] = field(default_factory=lambda: [
        "formal", "slang",
        "fantasy", "science",
        "positive", "negative",
        "confident", "uncertain",
        "technical", "simple",
        "smart", "intelligent",
        "unhappy", "sad",
        "angry", "furious",
        "scared", "fearful"
    ])

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

cfg = Config()