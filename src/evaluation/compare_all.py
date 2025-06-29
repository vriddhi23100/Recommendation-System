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

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '../../data/processed/')
RATINGS_PATH = os.path.join(PROCESSED_DIR, 'ratings.csv')
USER_EMB_PATH = os.path.join(PROCESSED_DIR, 'cf_baseline_user_emb.npy')
ITEM_EMB_PATH = os.path.join(PROCESSED_DIR, 'cf_baseline_item_emb.npy')
BERT_EMB_PATH = os.path.join(PROCESSED_DIR, 'item_bert_embeddings.npy')
USER_MAP_PATH = os.path.join(PROCESSED_DIR, 'user_id_map.json')
ITEM_MAP_PATH = os.path.join(PROCESSED_DIR, 'item_id_map.json')
BERT_IDS_PATH = os.path.join(PROCESSED_DIR, 'item_bert_ids.npy')
CF_BASELINE_CKPT = os.path.join(PROCESSED_DIR, 'cf_baseline.ckpt')
CBF_BASELINE_CKPT = os.path.join(PROCESSED_DIR, 'cbf_baseline.ckpt')
HYBRID_CKPT = os.path.join(PROCESSED_DIR, 'hybrid_model_best.ckpt')

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

# 3. Train/test split (consistent for all models)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 4. Dataset
class TestDataset(Dataset):
    def __init__(self, df):
        self.user_idx = df['user_idx'].values
        self.item_idx = df['item_idx'].values
        self.bert_idx = df['bert_idx'].values
        self.ratings = df['rating'].values.astype(np.float32)
    def __len__(self):
        return len(self.user_idx)
    def __getitem__(self, idx):
        return self.user_idx[idx], self.item_idx[idx], self.bert_idx[idx], self.ratings[idx]

test_dataset = TestDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 5. Model definitions
class CFOnlyModel(pl.LightningModule):
    def __init__(self, user_embs, item_embs):
        super().__init__()
        self.user_emb = nn.Embedding.from_pretrained(torch.tensor(user_embs, dtype=torch.float32), freeze=True)
        self.item_emb = nn.Embedding.from_pretrained(torch.tensor(item_embs, dtype=torch.float32), freeze=True)
    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        return (u * i).sum(dim=-1)

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

class HybridModel(pl.LightningModule):
    def __init__(self, user_embs, item_embs, bert_embs):
        super().__init__()
        self.user_embs = nn.Embedding.from_pretrained(torch.tensor(user_embs, dtype=torch.float32), freeze=True)
        self.item_embs = nn.Embedding.from_pretrained(torch.tensor(item_embs, dtype=torch.float32), freeze=True)
        self.bert_embs = nn.Embedding.from_pretrained(torch.tensor(bert_embs, dtype=torch.float32), freeze=True)
        cf_dim = user_embs.shape[1] + item_embs.shape[1]
        cbf_dim = bert_embs.shape[1]
        gate_in_dim = cf_dim + cbf_dim
        self.gate_layer = nn.Sequential(
            nn.Linear(gate_in_dim, 1),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Linear(max(cf_dim, cbf_dim), 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, user_idx, item_idx, bert_idx):
        u = self.user_embs(user_idx)
        i = self.item_embs(item_idx)
        b = self.bert_embs(bert_idx)
        cf_vec = torch.cat([u, i], dim=-1)
        cbf_vec = b
        gate_input = torch.cat([cf_vec, cbf_vec], dim=-1)
        gate = self.gate_layer(gate_input)
        if cf_vec.shape[1] > cbf_vec.shape[1]:
            pad = cf_vec.shape[1] - cbf_vec.shape[1]
            cbf_vec = torch.nn.functional.pad(cbf_vec, (0, pad))
        elif cbf_vec.shape[1] > cf_vec.shape[1]:
            pad = cbf_vec.shape[1] - cf_vec.shape[1]
            cf_vec = torch.nn.functional.pad(cf_vec, (0, pad))
        final_vec = gate * cf_vec + (1 - gate) * cbf_vec
        out = self.mlp(final_vec).squeeze(-1)
        return out

# 6. Load models
cf_model = CFOnlyModel(user_embs, item_embs)
cf_model = CFOnlyModel.load_from_checkpoint(CF_BASELINE_CKPT, user_embs=user_embs, item_embs=item_embs)
cf_model.eval()
cbf_model = CBFOnlyModel(bert_embs)
cbf_model = CBFOnlyModel.load_from_checkpoint(CBF_BASELINE_CKPT, bert_embs=bert_embs)
cbf_model.eval()
hybrid_model = HybridModel(user_embs, item_embs, bert_embs)
hybrid_model = HybridModel.load_from_checkpoint(HYBRID_CKPT, user_embs=user_embs, item_embs=item_embs, bert_embs=bert_embs)
hybrid_model.eval()

# 7. Evaluation function
def evaluate_model(model, test_loader, mode='hybrid'):
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for user_idx, item_idx, bert_idx, rating in test_loader:
            if mode == 'cf':
                preds = model(user_idx, item_idx)
            elif mode == 'cbf':
                preds = model(item_idx)
            elif mode == 'naive':
                preds_cf = cf_model(user_idx, item_idx)
                preds_cbf = cbf_model(item_idx)
                preds = 0.5 * preds_cf + 0.5 * preds_cbf
            else:  # hybrid
                preds = model(user_idx, item_idx, bert_idx)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(rating.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    return all_preds, all_targets, rmse, mae

# 8. Ranking metrics
def precision_at_k(y_true, y_pred, k):
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

# 9. Run all models
results = []
for name, model, mode in [
    ("CF Only", cf_model, 'cf'),
    ("CBF Only", cbf_model, 'cbf'),
    ("Naive Hybrid", None, 'naive'),
    ("Neural Gating Hybrid", hybrid_model, 'hybrid')
]:
    print(f"Evaluating {name}...")
    preds, targets, rmse, mae = evaluate_model(model, test_loader, mode=mode)
    prec = precision_at_k(targets, preds, K)
    ndcg = ndcg_at_k(targets, preds, K)
    results.append((name, rmse, mae, prec, ndcg))

# 10. Print results table
print("\nModel Comparison Table:")
print(f"{'Model':<22} {'RMSE':<8} {'MAE':<8} {'Prec@10':<10} {'NDCG@10':<10}")
for name, rmse, mae, prec, ndcg in results:
    print(f"{name:<22} {rmse:<8.4f} {mae:<8.4f} {prec:<10.4f} {ndcg:<10.4f}") 