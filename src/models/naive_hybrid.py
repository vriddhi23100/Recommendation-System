import os
import json
import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed/')
USER_EMB_PATH = os.path.join(PROCESSED_DIR, 'cf_baseline_user_emb.npy')
ITEM_EMB_PATH = os.path.join(PROCESSED_DIR, 'cf_baseline_item_emb.npy')
BERT_EMB_PATH = os.path.join(PROCESSED_DIR, 'item_bert_embeddings.npy')
USER_MAP_PATH = os.path.join(PROCESSED_DIR, 'user_id_map.json')
ITEM_MAP_PATH = os.path.join(PROCESSED_DIR, 'item_id_map.json')
CF_BASELINE_CKPT = os.path.join(PROCESSED_DIR, 'cf_baseline.ckpt')
CBF_BASELINE_CKPT = os.path.join(PROCESSED_DIR, 'cbf_baseline.ckpt')

# 1. Load mappings and embeddings
with open(USER_MAP_PATH) as f:
    user2idx = json.load(f)
with open(ITEM_MAP_PATH) as f:
    item2idx = json.load(f)
user_embs = np.load(USER_EMB_PATH)
item_embs = np.load(ITEM_EMB_PATH)
bert_embs = np.load(BERT_EMB_PATH)

# 2. CF Model
class CFOnlyModel(pl.LightningModule):
    def __init__(self, user_embs, item_embs):
        super().__init__()
        self.user_emb = nn.Embedding.from_pretrained(torch.tensor(user_embs, dtype=torch.float32), freeze=True)
        self.item_emb = nn.Embedding.from_pretrained(torch.tensor(item_embs, dtype=torch.float32), freeze=True)
    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        return (u * i).sum(dim=-1)

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
    def forward(self, item_idx):
        b = self.bert_embs(item_idx)
        out = self.mlp(b).squeeze(-1)
        return out

# 4. Load models
cf_model = CFOnlyModel(user_embs, item_embs)
cf_model = CFOnlyModel.load_from_checkpoint(CF_BASELINE_CKPT, user_embs=user_embs, item_embs=item_embs)
cf_model.eval()
cbf_model = CBFOnlyModel(bert_embs)
cbf_model = CBFOnlyModel.load_from_checkpoint(CBF_BASELINE_CKPT, bert_embs=bert_embs)
cbf_model.eval()

# 5. Naive Hybrid predict function
def predict(user_idx, item_idx):
    user_idx_tensor = torch.tensor([user_idx])
    item_idx_tensor = torch.tensor([item_idx])
    with torch.no_grad():
        cf_pred = cf_model(user_idx_tensor, item_idx_tensor).cpu().numpy()[0]
        cbf_pred = cbf_model(item_idx_tensor).cpu().numpy()[0]
    return 0.5 * cf_pred + 0.5 * cbf_pred

print("Naive hybrid model ready. Use predict(user_idx, item_idx) for evaluation.") 