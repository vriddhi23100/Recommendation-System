import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed/')
RATINGS_PATH = os.path.join(PROCESSED_DIR, 'ratings.csv')
BERT_EMB_PATH = os.path.join(PROCESSED_DIR, 'item_bert_embeddings.npy')
ITEM_MAP_PATH = os.path.join(PROCESSED_DIR, 'item_id_map.json')
CBF_BASELINE_CKPT = os.path.join(PROCESSED_DIR, 'cbf_baseline.ckpt')

BATCH_SIZE = 1024
EPOCHS = 5

# 1. Load ratings and map item IDs to indices
df = pd.read_csv(RATINGS_PATH)
with open(ITEM_MAP_PATH) as f:
    item2idx = json.load(f)
bert_embs = np.load(BERT_EMB_PATH)
df = df[df['itemId'].isin(item2idx)]
df['item_idx'] = df['itemId'].map(item2idx)

# 2. Dataset
class RatingsDataset(Dataset):
    def __init__(self, df):
        self.items = df['item_idx'].values
        self.ratings = df['rating'].values.astype(np.float32)
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx], self.ratings[idx]

dataset = RatingsDataset(df)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 3. CBF Model
class CBFOnlyModel(pl.LightningModule):
    def __init__(self, bert_embs):
        super().__init__()
        self.bert_embs = nn.Embedding.from_pretrained(torch.tensor(bert_embs, dtype=torch.float32), freeze=True)
        emb_dim = bert_embs.shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.loss_fn = nn.MSELoss()
    def forward(self, item_idx):
        b = self.bert_embs(item_idx)
        out = self.mlp(b).squeeze(-1)
        return out
    def training_step(self, batch, batch_idx):
        item_idx, rating = batch
        pred = self(item_idx)
        loss = self.loss_fn(pred, rating)
        self.log('train_loss', loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

model = CBFOnlyModel(bert_embs)

# 4. Train
trainer = pl.Trainer(max_epochs=EPOCHS, logger=False, enable_checkpointing=True, callbacks=[ModelCheckpoint(dirpath=PROCESSED_DIR, filename='cbf_baseline', save_top_k=1, monitor='train_loss', mode='min')], enable_model_summary=False)
trainer.fit(model, dataloader)

print(f"Saved CBF baseline model.") 