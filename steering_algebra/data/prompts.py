"""
Test prompts for evaluating steering effects.
These are neutral prompts that don't bias toward any particular style.
"""

from typing import List

def get_test_prompts() -> List[str]:
    """
    Get neutral test prompts for generation.
    These should not inherently bias toward any concept.
    """
    return [
        "Write about the weather today.",
        "Describe a typical morning routine.",
        "Explain how to make a cup of coffee.",
        "Tell me about your favorite hobby.",
        "Describe a memorable meal.",
        "Write about a walk in the park.",
        "Explain what you do for work.",
        "Describe your ideal weekend.",
        "Tell me about a book you enjoyed.",
        "Write about learning something new.",
        "Describe a city you'd like to visit.",
        "Explain how to organize a closet.",
        "Tell me about a pet or animal.",
        "Write about a recent discovery.",
        "Describe the process of cooking dinner.",
        "Explain how to plan a trip.",
        "Tell me about a skill you have.",
        "Write about a place that feels like home.",
        "Describe how to stay healthy.",
        "Explain your thoughts on technology.",
        "Tell me about a challenge you faced.",
        "Write about what motivates you.",
        "Describe a tradition you enjoy.",
        "Explain how to be productive.",
        "Tell me about something you're proud of.",
        "Write about the changing seasons.",
        "Describe a creative project.",
        "Explain how to learn a language.",
        "Tell me about your neighborhood.",
        "Write about a goal you have.",
    ]


def get_domain_specific_prompts() -> dict:
    """
    Get prompts organized by domain for more targeted evaluation.
    """
    return {
        "work": [
            "Write an email to a colleague about a project update.",
            "Describe your approach to problem-solving at work.",
            "Explain how you handle deadlines.",
            "Write about collaboration in a team setting.",
            "Describe a successful project you completed.",
        ],
        "technical": [
            "Explain how a computer stores data.",
            "Describe what happens when you load a webpage.",
            "Explain the concept of an algorithm.",
            "Write about how apps communicate with servers.",
            "Describe what machine learning does.",
        ],
        "personal": [
            "Write about what makes you happy.",
            "Describe your approach to making decisions.",
            "Explain how you handle stress.",
            "Write about a relationship that matters to you.",
            "Describe what you value most in life.",
        ],
        "opinion": [
            "Share your thoughts on social media.",
            "Write about the role of education.",
            "Describe your view on work-life balance.",
            "Explain what makes a good leader.",
            "Write about the importance of creativity.",
        ],
    }