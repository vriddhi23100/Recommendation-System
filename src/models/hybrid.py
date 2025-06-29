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
USER_EMB_PATH = os.path.join(PROCESSED_DIR, 'user_cf_embeddings.npy')
ITEM_EMB_PATH = os.path.join(PROCESSED_DIR, 'item_cf_embeddings.npy')
BERT_EMB_PATH = os.path.join(PROCESSED_DIR, 'item_bert_embeddings.npy')
USER_MAP_PATH = os.path.join(PROCESSED_DIR, 'user_id_map.json')
ITEM_MAP_PATH = os.path.join(PROCESSED_DIR, 'item_id_map.json')
BERT_IDS_PATH = os.path.join(PROCESSED_DIR, 'item_bert_ids.npy')

BATCH_SIZE = 1024
EPOCHS = 5

# 1. Load mappings and embeddings
with open(USER_MAP_PATH) as f:
    user2idx = json.load(f)
with open(ITEM_MAP_PATH) as f:
    item2idx = json.load(f)
bert_item_ids = np.load(BERT_IDS_PATH, allow_pickle=True)
bert_embs = np.load(BERT_EMB_PATH)
user_embs = np.load(USER_EMB_PATH)
item_embs = np.load(ITEM_EMB_PATH)

# Map itemId to BERT embedding index
bert_id2idx = {item_id: i for i, item_id in enumerate(bert_item_ids)}

# 2. Load ratings and map IDs to indices
df = pd.read_csv(RATINGS_PATH)
df = df[df['userId'].isin(user2idx) & df['itemId'].isin(item2idx) & df['itemId'].isin(bert_id2idx)]
df['user_idx'] = df['userId'].map(user2idx)
df['item_idx'] = df['itemId'].map(item2idx)
df['bert_idx'] = df['itemId'].map(bert_id2idx)

# 3. Dataset
class HybridDataset(Dataset):
    def __init__(self, df):
        self.user_idx = df['user_idx'].values
        self.item_idx = df['item_idx'].values
        self.bert_idx = df['bert_idx'].values
        self.ratings = df['rating'].values.astype(np.float32)
    def __len__(self):
        return len(self.user_idx)
    def __getitem__(self, idx):
        return self.user_idx[idx], self.item_idx[idx], self.bert_idx[idx], self.ratings[idx]

dataset = HybridDataset(df)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 4. Hybrid Model
class HybridModel(pl.LightningModule):
    def __init__(self, user_embs, item_embs, bert_embs):
        super().__init__()
        self.user_embs = nn.Embedding.from_pretrained(torch.tensor(user_embs, dtype=torch.float32), freeze=True)
        self.item_embs = nn.Embedding.from_pretrained(torch.tensor(item_embs, dtype=torch.float32), freeze=True)
        self.bert_embs = nn.Embedding.from_pretrained(torch.tensor(bert_embs, dtype=torch.float32), freeze=True)
        cf_dim = user_embs.shape[1] + item_embs.shape[1]
        cbf_dim = bert_embs.shape[1]
        gate_in_dim = cf_dim + cbf_dim
        # Explicit scalar gate
        self.gate_layer = nn.Sequential(
            nn.Linear(gate_in_dim, 1),
            nn.Sigmoid()
        )
        # Dense layers after fusion
        self.mlp = nn.Sequential(
            nn.Linear(max(cf_dim, cbf_dim), 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.loss_fn = nn.MSELoss()
    def forward(self, user_idx, item_idx, bert_idx):
        u = self.user_embs(user_idx)
        i = self.item_embs(item_idx)
        b = self.bert_embs(bert_idx)
        cf_vec = torch.cat([u, i], dim=-1)  # shape: (batch, cf_dim)
        cbf_vec = b  # shape: (batch, cbf_dim)
        # Gate input: concat original cf_vec and cbf_vec
        gate_input = torch.cat([cf_vec, cbf_vec], dim=-1)  # shape: (batch, cf_dim + cbf_dim)
        gate = self.gate_layer(gate_input)  # shape: (batch, 1)
        # Pad cf_vec or cbf_vec for blending if needed
        if cf_vec.shape[1] > cbf_vec.shape[1]:
            pad = cf_vec.shape[1] - cbf_vec.shape[1]
            cbf_vec = torch.nn.functional.pad(cbf_vec, (0, pad))
        elif cbf_vec.shape[1] > cf_vec.shape[1]:
            pad = cbf_vec.shape[1] - cf_vec.shape[1]
            cf_vec = torch.nn.functional.pad(cf_vec, (0, pad))
        final_vec = gate * cf_vec + (1 - gate) * cbf_vec
        out = self.mlp(final_vec).squeeze(-1)
        return out
    def training_step(self, batch, batch_idx):
        user_idx, item_idx, bert_idx, rating = batch
        pred = self(user_idx, item_idx, bert_idx)
        loss = self.loss_fn(pred, rating)
        self.log('train_loss', loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

model = HybridModel(user_embs, item_embs, bert_embs)

# 5. Train with checkpointing
checkpoint_callback = ModelCheckpoint(
    dirpath=PROCESSED_DIR, filename='hybrid_model_best', save_top_k=1, monitor='train_loss', mode='min')
trainer = pl.Trainer(max_epochs=EPOCHS, logger=False, callbacks=[checkpoint_callback], enable_model_summary=False)
trainer.fit(model, dataloader)

print("Training complete. Best model saved.") 