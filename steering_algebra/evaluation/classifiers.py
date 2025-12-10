"""
Attribute classifiers for evaluating steering success.
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from dataclasses import dataclass
import numpy as np

@dataclass
class ClassifierResult:
    concept: str
    score: float

class AttributeClassifier:
    PRETRAINED_MODELS = {
        "formal": "s-nlp/roberta-base-formality-ranker",
        "slang": "s-nlp/roberta-base-formality-ranker",
        "positive": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "negative": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    }
    
    def __init__(self, concept: str, method: str = "auto", device: str = "cuda", zero_shot_model: str = "facebook/bart-large-mnli"):
        self.concept = concept
        self.device = device
        self.method = "pretrained" if concept in self.PRETRAINED_MODELS else "zero_shot"
        
        if self.method == "pretrained":
            self._load_pretrained(concept)
        else:
            self._load_zero_shot(zero_shot_model)
    
    def _load_pretrained(self, concept: str):
        model_name = self.PRETRAINED_MODELS[concept]
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self._is_sentiment = "sentiment" in model_name or "sst-2" in model_name
    
    def _load_zero_shot(self, model_name: str):
        self.classifier = pipeline("zero-shot-classification", model=model_name, device=0 if self.device == "cuda" else -1)
    
    def score(self, text: str) -> float:
        if not text or len(text.strip()) < 3: return 0.5 
        if self.method == "pretrained": return self._score_pretrained(text)
        return self._score_zero_shot(text)

    def _score_pretrained(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        
        if self._is_sentiment:
            return probs[1].item() if self.concept == "positive" else probs[0].item()
        else:
            return probs[1].item() if self.concept == "formal" else probs[0].item()
    
    def _score_zero_shot(self, text: str) -> float:
        hypothesis_templates = {
            "formal": "This text is written in a formal, professional style.",
            "slang": "This text contains slang, internet abbreviations, or informal street language.",
            "confident": "This text expresses certainty, conviction, and confidence.",
            "uncertain": "This text expresses uncertainty, doubt, or confusion.",
            "technical": "This text uses technical terminology and jargon.",
            "simple": "This text uses simple, everyday language that a child could understand.",
            "fantasy": "This text is a fantasy story featuring magic, dragons, or mythical creatures.",
            "science": "This text is a scientific explanation discussing physics, biology, or research.",
            # NEW SYNONYMS
            "smart": "This text is clever, witty, and intelligent.",
            "intelligent": "This text is intellectual, analytical, and smart.",
            "unhappy": "This text expresses unhappiness, dissatisfaction, or complaint.",
            "sad": "This text expresses sadness, grief, sorrow, or tragedy.",
            "angry": "This text expresses anger, irritation, or annoyance.",
            "furious": "This text expresses fury, rage, or extreme anger.",
            "scared": "This text expresses fear, terror, or panic.",
            "fearful": "This text expresses anxiety, dread, or trembling fear.",
        }
        
        hypothesis = hypothesis_templates.get(self.concept, f"This text is {self.concept}.")
        
        try:
            result = self.classifier(
                text,
                candidate_labels=[hypothesis, "This is a neutral text."],
                multi_label=False
            )
            idx = result["labels"].index(hypothesis)
            return result["scores"][idx]
        except Exception:
            return 0.5

class MultiAttributeEvaluator:
    def __init__(self, concepts: List[str], device: str = "cuda"):
        self.concepts = concepts
        self.classifiers = {c: AttributeClassifier(c, device=device) for c in concepts}
    
    def evaluate(self, text: str, target_concepts: Optional[List[str]] = None) -> Dict[str, float]:
        if target_concepts is None: target_concepts = self.concepts
        return {c: self.classifiers[c].score(text) for c in target_concepts if c in self.classifiers}