"""
Attribute classifiers for evaluating steering success.
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
from dataclasses import dataclass
import numpy as np


@dataclass
class ClassifierResult:
    """Result from attribute classification."""
    concept: str
    score: float  # P(concept | text) in [0, 1]
    raw_logits: Optional[Tensor] = None


class AttributeClassifier:
    """
    Classifier for a single attribute/concept.
    Supports pre-trained models and zero-shot classification.
    """
    
    # Map concepts to pre-trained models
    PRETRAINED_MODELS = {
        "formal": "s-nlp/roberta-base-formality-ranker",
        "positive": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "negative": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "toxic": "unitary/toxic-bert",
    }
    
    def __init__(
        self,
        concept: str,
        method: str = "auto",  # "auto", "pretrained", "zero_shot"
        device: str = "cuda",
        zero_shot_model: str = "facebook/bart-large-mnli"
    ):
        self.concept = concept
        self.device = device
        
        # Determine method
        if method == "auto":
            method = "pretrained" if concept in self.PRETRAINED_MODELS else "zero_shot"
        
        self.method = method
        
        if method == "pretrained":
            self._load_pretrained(concept)
        else:
            self._load_zero_shot(zero_shot_model)
    
    def _load_pretrained(self, concept: str):
        """Load a pre-trained classifier."""
        model_name = self.PRETRAINED_MODELS[concept]
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Handle sentiment model specially (has multiple classes)
        self._is_sentiment = "sentiment" in model_name
    
    def _load_zero_shot(self, model_name: str):
        """Load zero-shot classification pipeline."""
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if self.device == "cuda" else -1
        )
    
    def score(self, text: str) -> float:
        """
        Compute P(concept | text).
        
        Returns:
            Score in [0, 1] where higher = more of the concept
        """
        # Validate input
        if not text or len(text.strip()) < 3:
            # Return neutral score for empty/invalid text
            return 0.5
        
        if self.method == "pretrained":
            return self._score_pretrained(text)
        else:
            return self._score_zero_shot(text)
    
    def _score_pretrained(self, text: str) -> float:
        """Score using pre-trained classifier."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        probs = torch.softmax(logits, dim=-1)[0]
        
        # Handle different model outputs
        if self._is_sentiment:
            # Sentiment models typically: [negative, neutral, positive]
            if self.concept == "positive":
                return probs[2].item()  # positive class
            elif self.concept == "negative":
                return probs[0].item()  # negative class
        else:
            # Binary classifiers: [negative, positive]
            return probs[1].item()
    
    def _score_zero_shot(self, text: str) -> float:
        """Score using zero-shot classification."""
        
        # Additional safety check
        if not text or len(text.strip()) < 3:
            return 0.5
        
        # Create hypothesis for the concept
        hypothesis_templates = {
            "formal": "This text is written in a formal style.",
            "casual": "This text is written in a casual style.",
            "verbose": "This text is detailed and comprehensive.",
            "concise": "This text is brief and to the point.",
            "confident": "This text expresses certainty and conviction.",
            "uncertain": "This text expresses uncertainty and doubt.",
            "technical": "This text uses technical language and jargon.",
            "simple": "This text uses simple, everyday language.",
            "positive": "This text has a positive tone.",
            "negative": "This text has a negative tone.",
        }
        
        candidate_labels = [
            hypothesis_templates.get(self.concept, f"This text is {self.concept}.")
        ]
        
        # Add a neutral counter-label
        try:
            result = self.classifier(
                text,
                candidate_labels=candidate_labels + ["This is a neutral text."],
                multi_label=False
            )
            
            # Return score for our concept
            idx = result["labels"].index(candidate_labels[0])
            return result["scores"][idx]
        
        except Exception as e:
            print(f"Warning: Zero-shot classification failed for text: '{text[:50]}...' Error: {e}")
            return 0.5  # Return neutral score on error
    
    def score_batch(self, texts: List[str]) -> List[float]:
        """Score multiple texts."""
        return [self.score(text) for text in texts]


class MultiAttributeEvaluator:
    """
    Evaluate text for multiple attributes simultaneously.
    """
    
    def __init__(
        self,
        concepts: List[str],
        device: str = "cuda"
    ):
        self.concepts = concepts
        self.classifiers = {
            concept: AttributeClassifier(concept, device=device)
            for concept in concepts
        }
    
    def evaluate(
        self,
        text: str,
        target_concepts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate text for target concepts.
        
        Args:
            text: Text to evaluate
            target_concepts: Concepts to evaluate (None = all)
            
        Returns:
            Dict mapping concept -> score
        """
        if target_concepts is None:
            target_concepts = self.concepts
        
        scores = {}
        for concept in target_concepts:
            if concept in self.classifiers:
                scores[concept] = self.classifiers[concept].score(text)
        
        return scores
    
    def evaluate_batch(
        self,
        texts: List[str],
        target_concepts: Optional[List[str]] = None
    ) -> List[Dict[str, float]]:
        """Evaluate multiple texts."""
        return [self.evaluate(text, target_concepts) for text in texts]
    
    def compute_joint_score(
        self,
        text: str,
        target_concepts: List[str]
    ) -> float:
        """
        Compute joint score: product of individual scores.
        High only if ALL target concepts are present.
        """
        scores = self.evaluate(text, target_concepts)
        return np.prod(list(scores.values()))
    
    def compute_steering_success(
        self,
        baseline_text: str,
        steered_text: str,
        target_concept: str,
        threshold: float = 0.1
    ) -> Dict:
        """
        Evaluate whether steering successfully increased the target concept.
        
        Returns:
            Dict with success metrics
        """
        baseline_score = self.classifiers[target_concept].score(baseline_text)
        steered_score = self.classifiers[target_concept].score(steered_text)
        
        improvement = steered_score - baseline_score
        success = improvement > threshold
        
        return {
            "baseline_score": baseline_score,
            "steered_score": steered_score,
            "improvement": improvement,
            "success": success,
            "relative_improvement": improvement / (baseline_score + 1e-6)
        }


class LLMJudge:
    """
    Use an LLM to score attributes when no classifier is available.
    Requires API access (OpenAI, Anthropic, etc.)
    """
    
    def __init__(self, api_type: str = "openai"):
        self.api_type = api_type
        # Note: Implement API initialization based on your setup
    
    def score(self, text: str, concept: str) -> float:
        """
        Use LLM to score text for a concept.
        """
        prompt = f"""Rate the following text on a scale from 0.0 to 1.0 for how {concept} it is.

0.0 = Not at all {concept}
0.5 = Somewhat {concept}
1.0 = Extremely {concept}

Text: "{text}"

Respond with only a decimal number between 0.0 and 1.0, nothing else."""
        
        # Implement API call based on your setup
        # response = call_api(prompt)
        # return float(response)
        raise NotImplementedError("Implement API call for your LLM provider")
    
    def score_batch(self, texts: List[str], concept: str) -> List[float]:
        """Score multiple texts."""
        return [self.score(text, concept) for text in texts]