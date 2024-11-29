# config.py
from typing import Set, Dict, Optional
from pathlib import Path

ALLOWED_PACKAGES = {
    # Core data science and ML
    'pandas',
    'numpy',
    'scikit-learn',
    'scipy',
    'statsmodels',
    'category_encoders',
    
    # Visualization
    'matplotlib',
    'matplotlib.pyplot',
    'seaborn',
    'plotly',
    'plotly.express',
    'plotly.graph_objects',
    'plotly.io',
    'bokeh',
    'altair',
    
    # Machine Learning
    'xgboost',
    'shap',
    
    # Data handling
    'pyarrow',
    'openpyxl',
    
    # AI/LLM
    'openai',
    'anthropic',
    'sentence_transformers',
    'transformers',
    'faiss',
    
    # Utilities
    'cryptography',
    'psutil',
    'requests',
    'httpx',
    'wordcloud',
    'yfinance',
    'geopy',
    'folium',
    'tabulate',
    'python_dotenv',
    'dotenv',
    
    # Network analysis
    'networkx',
    'community',
    
    # Standard library (if needed)
    'datetime',
    'json',
    'os',
    'sys',
    'pathlib',
    're',
    'time',
    'random',
    'typing',
    'io',
    'traceback',
    'logging'
}