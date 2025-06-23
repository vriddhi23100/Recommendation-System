import pandas as pd
import os
import zipfile

def load_movielens_data(data_dir):
    # Load MovieLens ratings and movies
    ratings_path = os.path.join(data_dir, 'u.data')
    ratings_df = pd.read_csv(ratings_path, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    movies_path = os.path.join(data_dir, 'u.item')
    movies_df = pd.read_csv(movies_path, sep='|', encoding='latin-1', names=['movie_id', 'title', 'release_date', 'video_release', 'url'] + [f'genre_{i}' for i in range(19)])
    return ratings_df, movies_df

def load_imdb_data(data_dir):
    # Load IMDB reviews
    csv_path = os.path.join(data_dir, 'IMDB_Dataset.csv')
    reviews_df = pd.read_csv(csv_path)
    reviews_df['review'] = reviews_df['review'].str.lower()
    reviews_df['sentiment'] = reviews_df['sentiment'].map({'positive': 1, 'negative': 0})
    return reviews_df

def create_rating_matrix(ratings_df):
    # User-item matrix
    return ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

def create_sentiment_matrix(reviews_df):
    # Movie-sentiment matrix
    sentiment_matrix = reviews_df.groupby('movie_id')['sentiment'].agg(['mean', 'count']).reset_index()
    sentiment_matrix.columns = ['movie_id', 'avg_sentiment', 'review_count']
    return sentiment_matrix

def merge_datasets(movielens_movies, imdb_reviews):
    # Merge on cleaned title
    movielens_movies['clean_title'] = movielens_movies['title'].str.lower()
    imdb_reviews['clean_title'] = imdb_reviews['title'].str.lower()
    merged_df = pd.merge(movielens_movies, imdb_reviews, on='clean_title', how='inner')
    return merged_df 