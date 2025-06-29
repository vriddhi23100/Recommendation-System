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
USER_EMB_PATH = os.path.join(PROCESSED_DIR, 'cf_baseline_user_emb.npy')
ITEM_EMB_PATH = os.path.join(PROCESSED_DIR, 'cf_baseline_item_emb.npy')
USER_MAP_PATH = os.path.join(PROCESSED_DIR, 'user_id_map.json')
ITEM_MAP_PATH = os.path.join(PROCESSED_DIR, 'item_id_map.json')
CF_BASELINE_CKPT = os.path.join(PROCESSED_DIR, 'cf_baseline.ckpt')

EMB_DIM = 64
BATCH_SIZE = 1024
EPOCHS = 5

# 1. Load ratings and map IDs to indices
df = pd.read_csv(RATINGS_PATH)
with open(USER_MAP_PATH) as f:
    user2idx = json.load(f)
with open(ITEM_MAP_PATH) as f:
    item2idx = json.load(f)
df = df[df['userId'].isin(user2idx) & df['itemId'].isin(item2idx)]
df['user_idx'] = df['userId'].map(user2idx)
df['item_idx'] = df['itemId'].map(item2idx)

# 2. Dataset
class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = df['user_idx'].values
        self.items = df['item_idx'].values
        self.ratings = df['rating'].values.astype(np.float32)
    def __len__(self):
        return len(self.users)
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

dataset = RatingsDataset(df)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 3. CF Model
class CFOnlyModel(pl.LightningModule):
    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.loss_fn = nn.MSELoss()
    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        return (u * i).sum(dim=-1)
    def training_step(self, batch, batch_idx):
        user_idx, item_idx, rating = batch
        pred = self(user_idx, item_idx)
        loss = self.loss_fn(pred, rating)
        self.log('train_loss', loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

n_users = len(user2idx)
n_items = len(item2idx)
model = CFOnlyModel(n_users, n_items, EMB_DIM)

# 4. Train
trainer = pl.Trainer(max_epochs=EPOCHS, logger=False, enable_checkpointing=True, callbacks=[ModelCheckpoint(dirpath=PROCESSED_DIR, filename='cf_baseline', save_top_k=1, monitor='train_loss', mode='min')], enable_model_summary=False)
trainer.fit(model, dataloader)

# 5. Save embeddings
user_emb = model.user_emb.weight.detach().cpu().numpy()
item_emb = model.item_emb.weight.detach().cpu().numpy()
np.save(USER_EMB_PATH, user_emb)
np.save(ITEM_EMB_PATH, item_emb)
print(f"Saved CF baseline user/item embeddings: {user_emb.shape}, {item_emb.shape}") 