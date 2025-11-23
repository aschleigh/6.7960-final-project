"""
Generate contrastive prompt pairs for steering vector extraction.

Each concept needs pairs of (positive_prompt, negative_prompt) where:
- positive_prompt: Induces the target behavior
- negative_prompt: Neutral or opposite behavior
"""

from typing import List, Tuple, Dict
import random

ContrastivePair = Tuple[str, str]  # (positive, negative)

# below we have formal vs. casual, positive vs. negative, verbose vs. concise, confident vs. uncertain, technical vs. simple


def get_contrastive_pairs(concept: str, n_pairs: int = 100) -> List[ContrastivePair]:
    generators = {
        "formal": _formal_pairs,
        "casual": _casual_pairs,
        "positive": _positive_pairs,
        "negative": _negative_pairs,
        "verbose": _verbose_pairs,
        "concise": _concise_pairs,
        "confident": _confident_pairs,
        "uncertain": _uncertain_pairs,
        "technical": _technical_pairs,
        "simple": _simple_pairs,
    }
    
    # if concept not in generators:
    #     raise ValueError(f"Unknown concept: {concept}. Available: {list(generators.keys())}")
    
    return generators[concept](n_pairs) 


def _formal_pairs(n: int) -> List[ContrastivePair]:
    templates = [
        (
            "Please write a formal response: ",
            "Write a casual response: ",
        ),
        (
            "Compose a professional email: ",
            "Write a quick message: ",
        ),
        (
            "Draft a formal letter regarding: ",
            "Jot down a note about: ",
        ),
    ]
    
    topics = [
        "the quarterly financial results",
        "the upcoming team meeting",
        "a request for time off",
        "feedback on the project proposal",
        "the new company policy",
        "a job application",
        "a complaint about service",
        "an invitation to a conference",
        "a progress update",
        "a recommendation letter",
        "the budget allocation",
        "a partnership opportunity",
        "technical documentation",
        "a product announcement",
        "meeting rescheduling",
        "project deadline extension",
        "performance review",
        "client presentation",
        "vendor negotiation",
        "team restructuring",
    ]
    
    pairs = []
    for _ in range(n):
        template = random.choice(templates) # choose a random prompt
        topic = random.choice(topics) # choose a random subject
        # prompt creation 
        positive = f"{template[0]}{topic}" 
        negative = f"{template[1]}{topic}"
        pairs.append((positive, negative))
    
    return pairs


def _casual_pairs(n: int) -> List[ContrastivePair]:
    formal_pairs = _formal_pairs(n)
    # opposite of formal pairs
    return [(neg, pos) for pos, neg in formal_pairs]


def _positive_pairs(n: int) -> List[ContrastivePair]:
    templates = [
        (
            "Write a positive review of: ",
            "Write a neutral description of: ",
        ),
        (
            "Describe the benefits of: ",
            "Describe: ",
        ),
        (
            "Explain why someone would love: ",
            "Explain what is: ",
        ),
        (
            "Write enthusiastically about: ",
            "Write about: ",
        ),
    ]
    
    topics = [
        "a new smartphone",
        "a restaurant experience",
        "a vacation destination",
        "a book you read",
        "a movie",
        "a software tool",
        "a fitness program",
        "a cooking recipe",
        "an online course",
        "a productivity app",
        "a coffee shop",
        "a hiking trail",
        "a music album",
        "a video game",
        "a podcast",
        "a streaming service",
        "a smart home device",
        "a meal delivery service",
        "an electric vehicle",
        "a meditation app",
    ]
    
    pairs = []
    for _ in range(n):
        template = random.choice(templates)
        topic = random.choice(topics)
        positive = f"{template[0]}{topic}"
        negative = f"{template[1]}{topic}"
        pairs.append((positive, negative))
    
    return pairs


def _negative_pairs(n: int) -> List[ContrastivePair]:
    positive_pairs = _positive_pairs(n)
    return [(neg, pos) for pos, neg in positive_pairs]


def _verbose_pairs(n: int) -> List[ContrastivePair]:
    templates = [
        (
            "Write a detailed, comprehensive explanation of: ",
            "Briefly explain: ",
        ),
        (
            "Provide an in-depth analysis with examples of: ",
            "Summarize: ",
        ),
        (
            "Elaborate extensively on: ",
            "Describe: ",
        ),
        (
            "Write a thorough, exhaustive description of: ",
            "Write about: ",
        ),
    ]
    
    topics = [
        "how neural networks learn",
        "the water cycle",
        "photosynthesis",
        "how computers work",
        "the solar system",
        "climate change",
        "machine learning",
        "blockchain technology",
        "quantum computing",
        "the immune system",
        "economic inflation",
        "the French Revolution",
        "DNA replication",
        "the internet",
        "artificial intelligence",
        "renewable energy",
        "the stock market",
        "black holes",
        "evolution",
        "cryptocurrency",
    ]
    
    pairs = []
    for _ in range(n):
        template = random.choice(templates)
        topic = random.choice(topics)
        positive = f"{template[0]}{topic}"
        negative = f"{template[1]}{topic}"
        pairs.append((positive, negative))
    
    return pairs


def _concise_pairs(n: int) -> List[ContrastivePair]:
    verbose_pairs = _verbose_pairs(n)
    return [(neg, pos) for pos, neg in verbose_pairs]


def _confident_pairs(n: int) -> List[ContrastivePair]:
    templates = [
        (
            "State with absolute certainty: ",
            "Speculate about: ",
        ),
        (
            "Definitively explain: ",
            "Consider the possibilities of: ",
        ),
        (
            "Assert confidently: ",
            "Wonder about: ",
        ),
        (
            "Declare with conviction: ",
            "Ponder: ",
        ),
    ]
    
    topics = [
        "the best programming language",
        "whether AI will surpass humans",
        "the future of remote work",
        "the healthiest diet",
        "the cause of the problem",
        "the solution to climate change",
        "the best investment strategy",
        "the meaning of the data",
        "the correct interpretation",
        "the optimal approach",
        "the right decision",
        "the best course of action",
        "the true explanation",
        "the fundamental cause",
        "the key insight",
        "the critical factor",
        "the winning strategy",
        "the correct answer",
        "the best practice",
        "the optimal solution",
    ]
    
    pairs = []
    for _ in range(n):
        template = random.choice(templates)
        topic = random.choice(topics)
        positive = f"{template[0]}{topic}"
        negative = f"{template[1]}{topic}"
        pairs.append((positive, negative))
    
    return pairs


def _uncertain_pairs(n: int) -> List[ContrastivePair]:
    confident_pairs = _confident_pairs(n)
    return [(neg, pos) for pos, neg in confident_pairs]


def _technical_pairs(n: int) -> List[ContrastivePair]:
    templates = [
        (
            "Explain to an expert with technical terminology: ",
            "Explain to a child: ",
        ),
        (
            "Write a technical documentation for: ",
            "Write a simple guide for: ",
        ),
        (
            "Describe using precise scientific language: ",
            "Describe in everyday words: ",
        ),
        (
            "Provide a rigorous technical analysis of: ",
            "Give a simple overview of: ",
        ),
    ]
    
    topics = [
        "how transformers work",
        "gradient descent",
        "TCP/IP networking",
        "database indexing",
        "memory allocation",
        "compiler optimization",
        "cryptographic hashing",
        "neural network backpropagation",
        "distributed systems consensus",
        "garbage collection",
        "cache coherence",
        "floating point arithmetic",
        "kernel scheduling",
        "virtual memory",
        "load balancing",
        "API design",
        "version control",
        "containerization",
        "microservices architecture",
        "message queuing",
    ]
    
    pairs = []
    for _ in range(n):
        template = random.choice(templates)
        topic = random.choice(topics)
        positive = f"{template[0]}{topic}"
        negative = f"{template[1]}{topic}"
        pairs.append((positive, negative))
    
    return pairs


def _simple_pairs(n: int) -> List[ContrastivePair]:
    technical_pairs = _technical_pairs(n)
    return [(neg, pos) for pos, neg in technical_pairs]

def get_all_pairs(concepts: List[str], n_pairs: int = 100) -> Dict[str, List[ContrastivePair]]:
    # 500 pairs total. Wondering if reversing is a good idea -- introducing some kind of bias or correlation?
    # actually nvm they are regenerated 
    return {concept: get_contrastive_pairs(concept, n_pairs) for concept in concepts}


# def validate_pairs(pairs: List[ContrastivePair]) -> bool:
#     for pos, neg in pairs:
#         if not isinstance(pos, str) or not isinstance(neg, str):
#             return False
#         if len(pos) == 0 or len(neg) == 0:
#             return False
#     return True