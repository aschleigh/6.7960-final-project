"""
Generate contrastive prompt pairs.
"""
from typing import List, Tuple, Dict
import random

ContrastivePair = Tuple[str, str]

def get_contrastive_pairs(concept: str, n_pairs: int = 100) -> List[ContrastivePair]:
    generators = {
        "formal": _formal_pairs,
        "casual": _casual_pairs,
        "slang": _slang_pairs,
        "positive": _positive_pairs,
        "negative": _negative_pairs,
        "confident": _confident_pairs,
        "uncertain": _uncertain_pairs,
        "fantasy": _fantasy_pairs,
        "science": _science_pairs,
        "technical": _technical_pairs,
        "simple": _simple_pairs,
        # NEW SYNONYMS
        "smart": _smart_pairs,
        "intelligent": _intelligent_pairs,
        "unhappy": _unhappy_pairs,
        "sad": _sad_pairs,
        "angry": _angry_pairs,
        "furious": _furious_pairs,
        "scared": _scared_pairs,
        "fearful": _fearful_pairs
    }
    # Fallback for inverse concepts if needed
    if concept not in generators:
        return []
    return generators[concept](n_pairs)

# --- NEW SYNONYM GENERATORS ---

def _smart_pairs(n: int) -> List[ContrastivePair]:
    """Smart vs Average/Dumb."""
    templates = [
        ("Give a really smart answer: ", "Give a dumb answer: "),
        ("Explain this brilliantly: ", "Explain this stupidly: "),
        ("Write a clever response: ", "Write a foolish response: "),
    ]
    topics = ["math", "philosophy", "the universe", "logic", "puzzles"]
    pairs = []
    for _ in range(n):
        t = random.choice(templates)
        top = random.choice(topics)
        pairs.append((f"{t[0]}{top}", f"{t[1]}{top}"))
    return pairs

def _intelligent_pairs(n: int) -> List[ContrastivePair]:
    """Intelligent vs Simple/Basic (Nuance difference from Smart)."""
    templates = [
        ("Provide an intelligent analysis of: ", "Provide a basic description of: "),
        ("Demonstrate high IQ in this text: ", "Demonstrate low IQ in this text: "),
        ("Write with intellectual depth about: ", "Write shallowly about: "),
    ]
    topics = ["consciousness", "AI", "politics", "economics", "history"]
    pairs = []
    for _ in range(n):
        t = random.choice(templates)
        top = random.choice(topics)
        pairs.append((f"{t[0]}{top}", f"{t[1]}{top}"))
    return pairs

def _angry_pairs(n: int) -> List[ContrastivePair]:
    templates = [
        ("Write an angry rant about: ", "Write a calm description of: "),
        ("Express anger regarding: ", "Express neutrality regarding: "),
        ("Complain bitterly about: ", "Speak indifferently about: "),
        ("React with irritation to: ", "React without emotion to: "),
    ]
    topics = ["the traffic", "the rude waiter", "the broken phone", "the delay", "the mistake"]
    pairs = []
    for _ in range(n):
        t = random.choice(templates)
        top = random.choice(topics)
        pairs.append((f"{t[0]}{top}", f"{t[1]}{top}"))
    return pairs

def _furious_pairs(n: int) -> List[ContrastivePair]:
    templates = [
        ("Scream in rage about: ", "Whisper calmly about: "),
        ("Write a furious, enraged message about: ", "Write a peaceful message about: "),
        ("Describe with burning fury: ", "Describe with serenity: "),
    ]
    topics = ["the injustice", "the betrayal", "the lost money", "the insult"]
    pairs = []
    for _ in range(n):
        t = random.choice(templates)
        top = random.choice(topics)
        pairs.append((f"{t[0]}{top}", f"{t[1]}{top}"))
    return pairs

def _scared_pairs(n: int) -> List[ContrastivePair]:
    templates = [
        ("Write a scared, terrified response to: ", "Write a brave, confident response to: "),
        ("Express fear of: ", "Express indifference to: "),
        ("Panic about: ", "Stay calm about: "),
    ]
    topics = ["the dark", "the spider", "the noise downstairs", "the shadow"]
    pairs = []
    for _ in range(n):
        t = random.choice(templates)
        top = random.choice(topics)
        pairs.append((f"{t[0]}{top}", f"{t[1]}{top}"))
    return pairs

def _fearful_pairs(n: int) -> List[ContrastivePair]:
    templates = [
        ("Describe with trembling fear: ", "Describe with bold courage: "),
        ("Show anxiety and dread about: ", "Show comfort and safety about: "),
        ("Write a fearful warning about: ", "Write a reassuring note about: "),
    ]
    topics = ["the future", "the unknown", "the monster", "the storm"]
    pairs = []
    for _ in range(n):
        t = random.choice(templates)
        top = random.choice(topics)
        pairs.append((f"{t[0]}{top}", f"{t[1]}{top}"))
    return pairs

def _unhappy_pairs(n: int) -> List[ContrastivePair]:
    """Unhappy vs Happy."""
    templates = [
        ("Write an unhappy story about: ", "Write a happy story about: "),
        ("Describe a disappointing experience at: ", "Describe a wonderful experience at: "),
        ("Complain about: ", "Praise: "),
    ]
    topics = ["the park", "the restaurant", "the birthday party", "the vacation"]
    pairs = []
    for _ in range(n):
        t = random.choice(templates)
        top = random.choice(topics)
        pairs.append((f"{t[0]}{top}", f"{t[1]}{top}"))
    return pairs

def _sad_pairs(n: int) -> List[ContrastivePair]:
    """Sad vs Joyful (Nuance: Sadness is deeper/quieter than Unhappiness)."""
    templates = [
        ("Write a tragic, sad poem about: ", "Write a joyful, upbeat poem about: "),
        ("Express deep sorrow regarding: ", "Express pure joy regarding: "),
        ("Describe a heartbreaking scene involving: ", "Describe a heartwarming scene involving: "),
    ]
    topics = ["lost love", "a rainy day", "saying goodbye", "loneliness"]
    pairs = []
    for _ in range(n):
        t = random.choice(templates)
        top = random.choice(topics)
        pairs.append((f"{t[0]}{top}", f"{t[1]}{top}"))
    return pairs

# --- STANDARD GENERATORS ---

def _formal_pairs(n: int) -> List[ContrastivePair]:
    """Extreme Formal vs Extreme Casual."""
    templates = [
        ("Write a response using extremely formal, archaic, and professional language: ", "Write a quick, messy, informal text message: "),
        ("Compose a rigid, bureaucratic official statement about: ", "Jot down a quick, slang-filled note about: "),
        ("Draft a legalistic and highly professional letter regarding: ", "Write a casual, lower-case message regarding: "),
    ]
    topics = ["the project", "the meeting", "the request", "the apology", "the announcement"]
    pairs = []
    for _ in range(n):
        t = random.choice(templates)
        top = random.choice(topics)
        pairs.append((f"{t[0]}{top}", f"{t[1]}{top}"))
    return pairs

def _slang_pairs(n: int) -> List[ContrastivePair]:
    """Extreme Slang vs Standard English."""
    templates = [
        ("Translate this into heavy internet slang and Gen-Z speak: ", "Translate this into standard, grammatically correct English: "),
        ("Write a text message full of abbreviations, emojis, and slang: ", "Write a polished, professional sentence: "),
        ("Describe using street slang and informal vernacular: ", "Describe using academic, formal English: "),
    ]
    topics = ["hello", "goodbye", "that is funny", "I am tired", "friend", "money", "food"]
    pairs = []
    for _ in range(n):
        t = random.choice(templates)
        top = random.choice(topics)
        pairs.append((f"{t[0]}{top}", f"{t[1]}{top}"))
    return pairs

def _casual_pairs(n: int) -> List[ContrastivePair]:
    return [(neg, pos) for pos, neg in _formal_pairs(n)]

def _positive_pairs(n: int) -> List[ContrastivePair]:
    return [("Write a positive review of the movie", "Write a negative review of the movie")] * n

def _negative_pairs(n: int) -> List[ContrastivePair]:
    return [(neg, pos) for pos, neg in _positive_pairs(n)]

def _confident_pairs(n: int) -> List[ContrastivePair]:
    return [("State with absolute certainty: X", "State with hesitation: X")] * n

def _uncertain_pairs(n: int) -> List[ContrastivePair]:
    return [(neg, pos) for pos, neg in _confident_pairs(n)]

def _technical_pairs(n: int) -> List[ContrastivePair]:
    return [("Explain O(n) complexity", "Explain sorting")] * n

def _simple_pairs(n: int) -> List[ContrastivePair]:
    return [(neg, pos) for pos, neg in _technical_pairs(n)]

def _fantasy_pairs(n: int) -> List[ContrastivePair]:
    # Fixed: Topic vs Topic
    templates = [
        ("Write a story about a wizard", "Write a story about an accountant"),
        ("Describe a dragon", "Describe a lizard"),
        ("Explain magic spells", "Explain tax laws"),
        ("Tell a tale of a magical kingdom", "Tell a tale of a modern city"),
    ]
    pairs = []
    for _ in range(n):
        t = random.choice(templates)
        pairs.append(t)
    return pairs

def _science_pairs(n: int) -> List[ContrastivePair]:
    # Fixed: Topic vs Topic
    templates = [
        ("Explain the physics of gravity", "Explain the history of art"),
        ("Describe a chemical reaction", "Describe a painting"),
        ("Write a lab report", "Write a poem"),
    ]
    pairs = []
    for _ in range(n):
        t = random.choice(templates)
        pairs.append(t)
    return pairs

def get_all_pairs(concepts: List[str], n_pairs: int = 100) -> Dict[str, List[ContrastivePair]]:
    return {concept: get_contrastive_pairs(concept, n_pairs) for concept in concepts}