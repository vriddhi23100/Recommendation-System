import torch
from transformers import BertTokenizer, BertModel, pipeline
import numpy as np
import re

class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def get_sentiment_score(self, text):
        # BERT for sentiment (very basic)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        sentiment_score = torch.mean(cls_embedding).item()
        sentiment_score = np.tanh(sentiment_score)
        return sentiment_score
    
    def get_movie_sentiment(self, movie_reviews):
        if not movie_reviews:
            return 0.0
        sentiment_scores = [self.get_sentiment_score(review) for review in movie_reviews]
        return np.mean(sentiment_scores)

class AdvancedNLPExtractor:
    def __init__(self):
        # Zero-shot classifier for themes/moods
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        # Example candidate labels (customize as needed)
        self.theme_labels = [
            "romance", "action", "comedy", "drama", "thriller", "horror", "sci-fi", "fantasy", "coming of age", "revenge", "friendship", "family"
        ]
        self.mood_labels = [
            "uplifting", "dark", "tense", "inspiring", "sad", "joyful", "melancholic", "exciting", "mysterious"
        ]
        # Regex for comparative opinions
        self.comparative_patterns = [
            r"better than ([A-Za-z0-9: ]+)",
            r"worse than ([A-Za-z0-9: ]+)",
            r"as good as ([A-Za-z0-9: ]+)",
            r"similar to ([A-Za-z0-9: ]+)"
        ]

    def extract_themes_and_moods(self, text):
        theme_result = self.classifier(text, self.theme_labels, multi_label=True)
        mood_result = self.classifier(text, self.mood_labels, multi_label=True)
        # Return top 3 with scores
        themes = list(zip(theme_result['labels'][:3], theme_result['scores'][:3]))
        moods = list(zip(mood_result['labels'][:3], mood_result['scores'][:3]))
        return {"themes": themes, "moods": moods}

    def extract_comparative_opinions(self, text):
        opinions = []
        for pattern in self.comparative_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                opinions.append({
                    "pattern": pattern,
                    "match": match.group(0),
                    "target": match.group(1),
                    "confidence": 1.0  # Regex match is binary; can be refined with ML
                })
        return opinions

    def analyze_review(self, text):
        """
        Extracts themes, moods, and comparative opinions from a review.
        Returns a dict with confidence scores.
        """
        aspects = self.extract_themes_and_moods(text)
        opinions = self.extract_comparative_opinions(text)
        return {"themes": aspects["themes"], "moods": aspects["moods"], "comparative_opinions": opinions} 