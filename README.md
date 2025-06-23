# Movie Recommendation System

A simple hybrid movie recommender using collaborative filtering, NLP, and deep learning.

## How to Run

1. Set up a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

## Folders
- `data/` — Datasets live here
- `src/` — All the code
- `main.py` — Entry point

## Needs
- Python 3.8+
- pandas, numpy, scikit-learn, torch, transformers, nltk, matplotlib, seaborn 

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environment
venv/

# Jupyter Notebook checkpoints
.ipynb_checkpoints

# Data files (customize as needed)
data/processed/
*.csv
*.gz
*.tar
*.zip

# OS files
.DS_Store
Thumbs.db 