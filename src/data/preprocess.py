import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(ratings_df, movies_df, test_size=0.2, random_state=42):
    """Preprocess and split ratings."""
    valid_movie_ids = set(movies_df['movie_id'])
    ratings_df = ratings_df[ratings_df['movie_id'].isin(valid_movie_ids)].copy()
    user_ids = sorted(ratings_df['user_id'].unique())
    movie_ids = list(movies_df['movie_id'])  # keep order
    user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
    movie_to_idx = {movie: idx for idx, movie in enumerate(movie_ids)}
    ratings_df['user_idx'] = ratings_df['user_id'].map(user_to_idx)
    ratings_df['movie_idx'] = ratings_df['movie_id'].map(movie_to_idx)
    train_data, test_data = train_test_split(
        ratings_df,
        test_size=test_size,
        random_state=random_state
    )
    return train_data, test_data, user_to_idx, movie_to_idx 