# Hybrid Recommendation System

## Objective
This project implements a **hybrid recommendation system** for movies/TV shows, designed to dynamically combine collaborative filtering (CF) and content-based filtering (CBF) using a gating network. The system leverages **BERT-based embeddings** for content features and compares its performance against traditional baselines.

---

## Key Features
- **Hybrid Model with Dynamic Weighting:**  
  Uses an explicit gating network to learn the optimal blend between collaborative (CF) and content-based (CBF) signals for each prediction.
- **BERT for Content Features:**  
  Movie/item texts are embedded using a pretrained, frozen Sentence-BERT model, providing rich semantic representations for CBF.
- **Collaborative Filtering:**  
  User and item embeddings are learned from rating data using PyTorch Lightning.
- **Comprehensive Evaluation:**  
  The system is benchmarked against pure CF and pure CBF models, reporting metrics like RMSE, MAE, Precision@10, and NDCG@10.

---

## Workflow
1. **Data Preprocessing:**  
   - Amazon 5-core Movies & TV dataset is processed to generate:
     - `item_texts.csv` (aggregated review text per item)
     - `ratings.csv` (userId, itemId, rating)
2. **Feature Extraction:**  
   - **CBF:**  
     - Sentence-BERT encodes item texts into dense vectors (`item_bert_embeddings.npy`).
   - **CF:**  
     - User and item embeddings are learned from ratings (`user_cf_embeddings.npy`, `item_cf_embeddings.npy`).
3. **Hybrid Model:**  
   - Both embedding types are fed into a neural network with a gating mechanism that dynamically weights CF and CBF for each prediction.
4. **Prediction & Evaluation:**  
   - The model outputs ratings/rankings, evaluated using standard metrics and gate analysis to interpret the blend between CF and CBF.

---

## Model Comparison
- **CF Only:** Uses only collaborative signals.
- **CBF Only:** Uses only BERT-based content features.
- **Naive Hybrid:** Combines CF and CBF features using a simple (static or concatenation-based) approach, without dynamic weighting.
- **Neural Gating Hybrid:** Learns to adaptively combine both, outperforming static or single-source models.

---

## Why BERT?
BERT-based embeddings capture deep semantic meaning from item texts and reviews, enabling the content-based branch to understand context, sentiment, and nuance far beyond traditional TF-IDF or metadata approaches.

---

## Summary
- **Hybrid, BERT-powered, dynamically weighted recommender**
- **Compares with classic CF, CBF, and naive hybrid**
- **Evaluates with robust metrics and interprets gating behavior**

---

## Collaborate
For questions, suggestions, or collaboration, please contact or open an issue on the repo.

## License
This project is licensed under the MIT License.
