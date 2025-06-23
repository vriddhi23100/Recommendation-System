import os
import sys
from src.data.load_data import (
    load_movielens_data,
    create_rating_matrix
)
from src.data.preprocess import preprocess_data
from src.models.collaborative import CollaborativeFiltering
from src.models.hybrid import HybridRecommender
from src.evaluation.metrics import evaluate_model, evaluate_recommendations
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    print("Loading MovieLens data...")
    ratings_df, _ = load_movielens_data('data/raw/ml-100k')

    print("Loading movies with TMDB descriptions...")
    movies_df = pd.read_csv('data/raw/ml-100k/movies_with_tmdb_desc.csv')

    print("Creating rating matrix...")
    rating_matrix = create_rating_matrix(ratings_df)
    print(rating_matrix.head())

    print("Preprocessing data...")
    train_data, test_data, _, _ = preprocess_data(ratings_df, movies_df)

    print("TF-IDF features from TMDB descriptions...")
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(movies_df['tmdb_description'].fillna(''))

    print("Init models...")
    n_users = train_data['user_idx'].max() + 1
    n_items = movies_df.shape[0]
    cf_model = CollaborativeFiltering(n_factors=100)
    hybrid_model = HybridRecommender(n_users=n_users, n_items=n_items, n_factors=100, hidden_dim=256)

    print("Training collaborative filtering...")
    cf_model.train(train_data)

    print("Training hybrid model...")
    hybrid_model.train(train_data, tfidf_matrix)

    print("Evaluating models...")
    cf_metrics = evaluate_model(cf_model, test_data)
    hybrid_metrics = evaluate_model(hybrid_model, test_data)

    print("Evaluating recommendations...")
    cf_rec_metrics = evaluate_recommendations(cf_model, test_data)
    hybrid_rec_metrics = evaluate_recommendations(hybrid_model, test_data)

    print("\nResults:")
    print("Collaborative Filtering Metrics:")
    print(f"RMSE: {cf_metrics['RMSE']:.4f}")
    print(f"MAE: {cf_metrics['MAE']:.4f}")
    print(f"Precision: {cf_rec_metrics['Precision']:.4f}")
    print(f"Recall: {cf_rec_metrics['Recall']:.4f}")

    print("\nHybrid Model Metrics:")
    print(f"RMSE: {hybrid_metrics['RMSE']:.4f}")
    print(f"MAE: {hybrid_metrics['MAE']:.4f}")
    print(f"Precision: {hybrid_rec_metrics['Precision']:.4f}")
    print(f"Recall: {hybrid_rec_metrics['Recall']:.4f}")

    sample_user = 1
    print(f"\nTop 10 recommendations for user {sample_user}:")
    cf_recommendations = cf_model.get_recommendations(sample_user, n_recommendations=20)
    hybrid_recommendations = hybrid_model.get_recommendations(sample_user, n_recommendations=20)

    print("\nCollaborative Filtering Recommendations:")
    seen = set()
    count = 1
    for movie_idx in cf_recommendations:
        if movie_idx in seen:
            continue
        seen.add(movie_idx)
        movie_title = movies_df.iloc[movie_idx]['title']
        print(f"{count}. {movie_title}")
        count += 1
        if count > 10:
            break

    print("\nHybrid Model Recommendations:")
    seen = set()
    count = 1
    for movie_idx in hybrid_recommendations:
        if movie_idx in seen:
            continue
        seen.add(movie_idx)
        movie_title = movies_df.iloc[movie_idx]['title']
        print(f"{count}. {movie_title}")
        count += 1
        if count > 10:
            break

if __name__ == "__main__":
    main() 