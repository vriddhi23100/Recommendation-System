import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pytorch_lightning as pl
from torch import nn
import math

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed/')
RATINGS_PATH = os.path.join(PROCESSED_DIR, 'ratings.csv')
USER_EMB_PATH = os.path.join(PROCESSED_DIR, 'user_cf_embeddings.npy')
ITEM_EMB_PATH = os.path.join(PROCESSED_DIR, 'item_cf_embeddings.npy')
BERT_EMB_PATH = os.path.join(PROCESSED_DIR, 'item_bert_embeddings.npy')
USER_MAP_PATH = os.path.join(PROCESSED_DIR, 'user_id_map.json')
ITEM_MAP_PATH = os.path.join(PROCESSED_DIR, 'item_id_map.json')
BERT_IDS_PATH = os.path.join(PROCESSED_DIR, 'item_bert_ids.npy')
BEST_MODEL_PATH = os.path.join(PROCESSED_DIR, 'hybrid_model_best.ckpt')

BATCH_SIZE = 1024
K = 10

# 1. Load mappings and embeddings
with open(USER_MAP_PATH) as f:
    user2idx = json.load(f)
with open(ITEM_MAP_PATH) as f:
    item2idx = json.load(f)
bert_item_ids = np.load(BERT_IDS_PATH, allow_pickle=True)
bert_embs = np.load(BERT_EMB_PATH)
user_embs = np.load(USER_EMB_PATH)
item_embs = np.load(ITEM_EMB_PATH)
bert_id2idx = {item_id: i for i, item_id in enumerate(bert_item_ids)}

# 2. Load ratings and map IDs to indices
df = pd.read_csv(RATINGS_PATH)
df = df[df['userId'].isin(user2idx) & df['itemId'].isin(item2idx) & df['itemId'].isin(bert_id2idx)]
df['user_idx'] = df['userId'].map(user2idx)
df['item_idx'] = df['itemId'].map(item2idx)
df['bert_idx'] = df['itemId'].map(bert_id2idx)

# 3. Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 4. Dataset
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

test_dataset = HybridDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 5. Hybrid Model (must match training)
class HybridModel(pl.LightningModule):
    def __init__(self, user_embs, item_embs, bert_embs):
        super().__init__()
        self.user_embs = nn.Embedding.from_pretrained(torch.tensor(user_embs, dtype=torch.float32), freeze=True)
        self.item_embs = nn.Embedding.from_pretrained(torch.tensor(item_embs, dtype=torch.float32), freeze=True)
        self.bert_embs = nn.Embedding.from_pretrained(torch.tensor(bert_embs, dtype=torch.float32), freeze=True)
        emb_dim = user_embs.shape[1] + item_embs.shape[1] + bert_embs.shape[1]
        self.gate = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, user_idx, item_idx, bert_idx):
        u = self.user_embs(user_idx)
        i = self.item_embs(item_idx)
        b = self.bert_embs(bert_idx)
        x = torch.cat([u, i, b], dim=-1)
        gated = x * self.gate(x)
        out = self.mlp(gated).squeeze(-1)
        return out

# 6. Load best model
model = HybridModel.load_from_checkpoint(BEST_MODEL_PATH, user_embs=user_embs, item_embs=item_embs, bert_embs=bert_embs)
model.eval()
model.freeze()

# 7. Predict on test set
all_preds = []
all_targets = []
with torch.no_grad():
    for user_idx, item_idx, bert_idx, rating in test_loader:
        preds = model(user_idx, item_idx, bert_idx)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(rating.cpu().numpy())
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

# 8. Regression metrics
rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
mae = mean_absolute_error(all_targets, all_preds)
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# 9. Ranking metrics (Precision@K, NDCG@K)
def precision_at_k(y_true, y_pred, k):
    # y_true, y_pred: list of (user, item, rating, pred)
    user_to_true = {}
    user_to_pred = {}
    for u, i, r, p in zip(test_df['user_idx'], test_df['item_idx'], y_true, y_pred):
        user_to_true.setdefault(u, []).append((i, r))
        user_to_pred.setdefault(u, []).append((i, p))
    precisions = []
    for u in user_to_true:
        true_items = set(i for i, r in sorted(user_to_true[u], key=lambda x: -x[1])[:k])
        pred_items = [i for i, p in sorted(user_to_pred[u], key=lambda x: -x[1])[:k]]
        hit = len(set(pred_items) & true_items)
        precisions.append(hit / k)
    return np.mean(precisions)

def ndcg_at_k(y_true, y_pred, k):
    user_to_true = {}
    user_to_pred = {}
    for u, i, r, p in zip(test_df['user_idx'], test_df['item_idx'], y_true, y_pred):
        user_to_true.setdefault(u, []).append((i, r))
        user_to_pred.setdefault(u, []).append((i, p))
    ndcgs = []
    for u in user_to_true:
        true_sorted = sorted(user_to_true[u], key=lambda x: -x[1])[:k]
        pred_sorted = sorted(user_to_pred[u], key=lambda x: -x[1])[:k]
        ideal_dcg = sum((2 ** r - 1) / np.log2(idx + 2) for idx, (i, r) in enumerate(true_sorted))
        dcg = 0.0
        for idx, (i, _) in enumerate(pred_sorted):
            for j, (ti, tr) in enumerate(true_sorted):
                if i == ti:
                    dcg += (2 ** tr - 1) / np.log2(idx + 2)
        ndcgs.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
    return np.mean(ndcgs)

prec_at_k = precision_at_k(all_targets, all_preds, K)
ndcg_k = ndcg_at_k(all_targets, all_preds, K)
print(f"Precision@{K}: {prec_at_k:.4f}")
print(f"NDCG@{K}: {ndcg_k:.4f}")

# --- Inspect gate values on the test set ---
gate_sum = 0.0
gate_min = math.inf
gate_max = -math.inf
gate_count = 0

user_hist = train_df['user_idx'].value_counts().to_dict()

# Helper to get the gate layer regardless of attribute name
get_gate_layer = getattr(model, 'gate_layer', None)
if get_gate_layer is None:
    get_gate_layer = getattr(model, 'gate', None)

with torch.no_grad():
    for user_idx, item_idx, bert_idx, rating in test_loader:
        u = model.user_embs(user_idx)
        i = model.item_embs(item_idx)
        b = model.bert_embs(bert_idx)
        cf_vec = torch.cat([u, i], dim=-1)
        cbf_vec = b
        gate_input = torch.cat([cf_vec, cbf_vec], dim=-1)
        gate = get_gate_layer(gate_input).cpu().numpy()
        gate_sum += gate.sum()
        gate_min = min(gate_min, gate.min())
        gate_max = max(gate_max, gate.max())
        gate_count += gate.size

mean_gate = gate_sum / gate_count if gate_count > 0 else float('nan')
print(f"\nGate (CF weight) statistics on test set:")
print(f"Mean: {mean_gate:.3f}, Min: {gate_min:.3f}, Max: {gate_max:.3f}")

# Show a few examples for new vs. experienced users
print("\nExample gate values for new vs. experienced users:")
for n in [1, 5, 20, 100]:
    users = [u for u, cnt in user_hist.items() if cnt == n]
    if users:
        user = users[0]
        # Pick a random item for this user in test set
        items = test_df[test_df['user_idx'] == user]['item_idx'].values
        if len(items) > 0:
            item = items[0]
            bert = test_df[(test_df['user_idx'] == user) & (test_df['item_idx'] == item)]['bert_idx'].values[0]
            u_tensor = torch.tensor([user])
            i_tensor = torch.tensor([item])
            b_tensor = torch.tensor([bert])
            with torch.no_grad():
                u_emb = model.user_embs(u_tensor)
                i_emb = model.item_embs(i_tensor)
                b_emb = model.bert_embs(b_tensor)
                cf_vec = torch.cat([u_emb, i_emb], dim=-1)
                cbf_vec = b_emb
                gate_input = torch.cat([cf_vec, cbf_vec], dim=-1)
                gate_val = get_gate_layer(gate_input).cpu().numpy()
                gate = float(gate_val.flatten()[0])
            print(f"User with {n} train ratings: gate={gate:.3f} (1=CF, 0=CBF)") 