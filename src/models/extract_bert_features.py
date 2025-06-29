import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed/')
ITEM_TEXTS_PATH = os.path.join(PROCESSED_DIR, 'item_texts.csv')
EMBEDDINGS_PATH = os.path.join(PROCESSED_DIR, 'item_bert_embeddings.npy')
IDS_PATH = os.path.join(PROCESSED_DIR, 'item_bert_ids.npy')

# Load item texts
df = pd.read_csv(ITEM_TEXTS_PATH)
item_ids = df['itemId'].tolist()
texts = df['text'].fillna("").tolist()

# Load pretrained sentence-transformer (frozen)
model = SentenceTransformer('all-MiniLM-L6-v2')
model.eval()

# Compute embeddings
embeddings = []
for text in tqdm(texts, desc='Encoding items'):
    with torch.no_grad():
        emb = model.encode(text, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    embeddings.append(emb)
embeddings = np.stack(embeddings)

# Save embeddings and item IDs
np.save(EMBEDDINGS_PATH, embeddings)
np.save(IDS_PATH, np.array(item_ids))

print(f"Saved {embeddings.shape[0]} item embeddings of dim {embeddings.shape[1]}.") 