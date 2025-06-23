import torch
import torch.nn as nn
import numpy as np

class HybridRecommender:
    def __init__(self, n_users, n_items, n_factors=100, hidden_dim=256):
        self.n_factors = n_factors
        self.hidden_dim = hidden_dim
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_users = n_users
        self.n_items = n_items
        self._build_model(n_users, n_items)
    
    def _build_model(self, n_users, n_items):
        class HybridModel(nn.Module):
            def __init__(self, n_users, n_items, n_factors, hidden_dim):
                super(HybridModel, self).__init__()
                # Embeddings
                self.user_embedding = nn.Embedding(n_users, n_factors)
                self.item_embedding = nn.Embedding(n_items, n_factors)
                # Simple MLP
                self.fc_layers = nn.Sequential(
                    nn.Linear(n_factors * 2 + 1, hidden_dim),  # +1 for sentiment
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim // 2, 1)
                )
            def forward(self, user_input, item_input, sentiment):
                user_embedded = self.user_embedding(user_input)
                item_embedded = self.item_embedding(item_input)
                # Stack all features
                concat = torch.cat([user_embedded, item_embedded, sentiment.unsqueeze(1)], dim=1)
                return self.fc_layers(concat)
        self.model = HybridModel(n_users, n_items, self.n_factors, self.hidden_dim)
        self.model.to(self.device)
    
    def train(self, train_data, content_features, n_epochs=10, batch_size=32, learning_rate=0.001):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        users = torch.LongTensor(train_data['user_idx'].values).to(self.device)
        items = torch.LongTensor(train_data['movie_idx'].values).to(self.device)
        ratings = torch.FloatTensor(train_data['rating'].values).to(self.device)
        sentiments = torch.FloatTensor([0.0] * len(train_data)).to(self.device)  # placeholder
        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0
            for i in range(0, len(train_data), batch_size):
                batch_users = users[i:i+batch_size]
                batch_items = items[i:i+batch_size]
                batch_ratings = ratings[i:i+batch_size]
                batch_sentiments = sentiments[i:i+batch_size]
                preds = self.model(batch_users, batch_items, batch_sentiments)
                loss = criterion(preds.squeeze(), batch_ratings)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / (len(train_data) / batch_size)
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, user_idx, item_idx, sentiment_score=0.0):
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx]).to(self.device)
            item_tensor = torch.LongTensor([item_idx]).to(self.device)
            sentiment_tensor = torch.FloatTensor([sentiment_score]).to(self.device)
            prediction = self.model(user_tensor, item_tensor, sentiment_tensor)
            return prediction.item()
    
    def get_recommendations(self, user_idx, n_recommendations=10, sentiment_scores=None):
        if sentiment_scores is None:
            sentiment_scores = {idx: 0.0 for idx in range(self.model.item_embedding.num_embeddings)}
        predictions = []
        for item_idx in range(self.model.item_embedding.num_embeddings):
            pred = self.predict(user_idx, item_idx, sentiment_scores[item_idx])
            predictions.append((item_idx, pred))
        # Top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [item_idx for item_idx, _ in predictions[:n_recommendations]] 