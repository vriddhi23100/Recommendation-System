import pandas as pd
import requests
from tqdm import tqdm
import time
import os

# Set your TMDB API key as an environment variable: TMDB_API_KEY
TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
if not TMDB_API_KEY:
    raise ValueError('TMDB_API_KEY environment variable not set. Please set it in your environment or .env file.')

def fetch_tmdb_description(title, year=None):
    """Fetch movie description from TMDB by title (and optionally year)."""
    url = f'https://api.themoviedb.org/3/search/movie'
    params = {
        'api_key': TMDB_API_KEY,
        'query': title,
        'include_adult': False,
    }
    if year:
        params['year'] = year
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get('results')
        if results:
            return results[0].get('overview', '')
    return ''

def main():
    # Load movies
    movies = pd.read_csv('data/raw/ml-100k/u.item', sep='|', encoding='latin-1', names=['movie_id', 'title', 'release_date', 'video_release', 'url'] + [f'genre_{i}' for i in range(19)])
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').fillna('').astype(str)
    movies['clean_title'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
    descriptions = []
    for _, row in tqdm(movies.iterrows(), total=len(movies)):
        desc = fetch_tmdb_description(row['clean_title'], row['year'] if row['year'] else None)
        descriptions.append(desc)
        time.sleep(0.25)  # don't spam TMDB
    movies['tmdb_description'] = descriptions
    movies.to_csv('data/raw/ml-100k/movies_with_tmdb_desc.csv', index=False)
    print('Saved movies_with_tmdb_desc.csv with TMDB descriptions!')

if __name__ == '__main__':
    main()