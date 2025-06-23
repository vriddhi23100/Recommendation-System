# Movie Recommendation System

This project implements a hybrid movie recommendation system that leverages both collaborative filtering and content-based techniques. By combining user-item interaction data with movie metadata and natural language processing (NLP) features, the system aims to provide more accurate and personalized movie recommendations.

The core of the system integrates:
- **Collaborative Filtering:** Learns user and item preferences from historical ratings.
- **Content-Based Features:** Utilizes movie descriptions and (optionally) sentiment analysis from user reviews to enrich recommendations.
- **Deep Learning:** Employs neural networks to blend collaborative and content signals for improved prediction accuracy.


## Results

- The script will print RMSE, MAE, Precision, and Recall for both collaborative and hybrid models.
- Top-10 movie recommendations for a sample user will be displayed.

## Customization

- **Add new models:** Implement in `src/models/`.
- **Add new evaluation metrics:** Implement in `src/evaluation/`.
- **Use real sentiment:** Integrate with `src/models/nlp.py` and pass sentiment scores to the hybrid model.
  
