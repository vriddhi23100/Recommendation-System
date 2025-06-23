import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    def __init__(self, n_factors=100):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
    
    def train(self, train_data):
        n_users = train_data['user_idx'].max() + 1
        n_items = train_data['movie_idx'].max() + 1
        # Init factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        # Build rating matrix
        rating_matrix = np.zeros((n_users, n_items))
        for _, row in train_data.iterrows():
            rating_matrix[row['user_idx'], row['movie_idx']] = row['rating']
        # Simple matrix factorization
        learning_rate = 0.005
        n_iterations = 300
        for _ in range(n_iterations):
            for user_idx in range(n_users):
                for item_idx in range(n_items):
                    if rating_matrix[user_idx, item_idx] > 0:
                        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
                        error = rating_matrix[user_idx, item_idx] - prediction
                        # Update
                        self.user_factors[user_idx] += learning_rate * error * self.item_factors[item_idx]
                        self.item_factors[item_idx] += learning_rate * error * self.user_factors[user_idx]
    
    def predict(self, user_idx, item_idx):
        return np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
    
    def get_recommendations(self, user_idx, n_recommendations=10):
        user_ratings = np.dot(self.user_factors[user_idx], self.item_factors.T)
        return np.argsort(user_ratings)[-n_recommendations:][::-1] 