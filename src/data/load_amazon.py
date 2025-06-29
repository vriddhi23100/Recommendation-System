import os
import json
import pandas as pd
from collections import defaultdict

RAW_PATH = os.path.join(os.path.dirname(__file__), '../../data/raw/amazon/Movies_and_TV_5.json')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed/')

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

ratings = []
item_texts = defaultdict(list)

with open(RAW_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        user = data.get('reviewerID')
        item = data.get('asin')
        rating = data.get('overall')
        text = data.get('reviewText', '')
        if user and item and rating is not None:
            ratings.append({'userId': user, 'itemId': item, 'rating': rating})
            if text:
                item_texts[item].append(text)

# Save ratings
ratings_df = pd.DataFrame(ratings)
ratings_df.to_csv(os.path.join(PROCESSED_DIR, 'ratings.csv'), index=False)

# Aggregate texts per item
item_texts_agg = [{'itemId': item, 'text': ' '.join(texts)} for item, texts in item_texts.items()]
item_texts_df = pd.DataFrame(item_texts_agg)
item_texts_df.to_csv(os.path.join(PROCESSED_DIR, 'item_texts.csv'), index=False)

print(f"Saved {len(ratings_df)} ratings and {len(item_texts_df)} item texts.") 