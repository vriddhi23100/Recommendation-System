import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, test_data):
    """Quick eval: RMSE, MAE."""
    predictions = []
    actuals = []
    if hasattr(model, 'user_factors'):
        n_users = model.user_factors.shape[0]
        n_items = model.item_factors.shape[0]
    else:
        n_users = model.n_users
        n_items = model.n_items
    for _, row in test_data.iterrows():
        user_idx = row['user_idx']
        item_idx = row['movie_idx']
        actual_rating = row['rating']
        if user_idx >= n_users or item_idx >= n_items:
            continue
        if hasattr(model, 'predict'):
            pred_rating = model.predict(user_idx, item_idx)
        else:
            pred_rating = model.predict(user_idx, item_idx, 0.0)
        predictions.append(pred_rating)
        actuals.append(actual_rating)
    rmse = np.sqrt(mean_squared_error(actuals, predictions)) if predictions else float('nan')
    mae = mean_absolute_error(actuals, predictions) if predictions else float('nan')
    return {'RMSE': rmse, 'MAE': mae}

def evaluate_recommendations(model, test_data, n_recommendations=10):
    """Precision/Recall for top-N."""
    total_precision = 0
    total_recall = 0
    if hasattr(model, 'user_factors'):
        n_users = model.user_factors.shape[0]
        n_items = model.item_factors.shape[0]
    else:
        n_users = model.n_users
        n_items = model.n_items
    user_indices = test_data['user_idx'].unique()
    valid_user_indices = [u for u in user_indices if u < n_users]
    n_valid_users = len(valid_user_indices)
    for user_idx in valid_user_indices:
        user_ratings = test_data[test_data['user_idx'] == user_idx]
        actual_items = set(user_ratings[(user_ratings['rating'] >= 4) & (user_ratings['movie_idx'] < n_items)]['movie_idx'])
        if not actual_items:
            continue
        if hasattr(model, 'get_recommendations'):
            recommended_items = set([i for i in model.get_recommendations(user_idx, n_recommendations) if i < n_items])
        else:
            recommended_items = set([i for i in model.get_recommendations(user_idx, n_recommendations, {}) if i < n_items])
        if recommended_items:
            precision = len(actual_items.intersection(recommended_items)) / len(recommended_items)
            recall = len(actual_items.intersection(recommended_items)) / len(actual_items)
            total_precision += precision
            total_recall += recall
    return {
        'Precision': total_precision / n_valid_users if n_valid_users else float('nan'),
        'Recall': total_recall / n_valid_users if n_valid_users else float('nan')
    } 