import torch
from transformers import BertTokenizer, BertModel
import numpy as np

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