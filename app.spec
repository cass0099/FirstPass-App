# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

block_cipher = None

# Create a runtime hook for matplotlib
mpl_hook = """
import matplotlib
matplotlib.use('QtAgg')  # Use the Qt backend
"""

with open('mpl_hook.py', 'w') as f:
    f.write(mpl_hook)

# Additional data files for sentence transformers
sentence_transformer_files = [
    ('fresh_env/Lib/site-packages/sentence_transformers', 'sentence_transformers'),
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=sentence_transformer_files,
    hiddenimports=[
        # Your local modules
        'analyzer',
        'config',
        'constants',
        'data_preview',
        'db_config',
        'eula_dialog',
        'generate_knowledge_base',
        'gui_components',
        'knowledge_management',
        'license_expired_view',
        'license_generator',
        'license_manager',
        'license_ui',
        'llm_handler',
        'rag_integration',
        'reset_knowledge_base',
        'script_runner',
        'storage',
        # Core data science and ML libraries
        'numpy',
        'pandas',
        'scipy',
        'statsmodels',
        'sklearn',
        'sklearn.utils',
        'category_encoders',
        # Plotting libraries
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_agg',
        'seaborn',
        'plotly',
        'plotly.graph_objs',
        'plotly.subplots',
        'plotly.figure_factory',
        'plotly.express',
        'plotly.colors',
        'plotly.io',
        'plotly.utils',
        'bokeh',
        'altair',
        # Additional ML libraries
        'xgboost',
        'shap',
        # Data handling
        'pyarrow',
        'openpyxl',
        # AI/LLM related
        'openai',
        'anthropic',
        'sentence_transformers',
        'sentence_transformers.models',
        'sentence_transformers.cross_encoder',
        'sentence_transformers.readers',
        'sentence_transformers.evaluation',
        'sentence_transformers.util',
        'sentence_transformers.losses',
        'sentence_transformers.datasets',
        'transformers',
        'transformers.models',
        'transformers.tokenization_utils',
        'faiss_cpu',
        # Utility libraries
        'cryptography',
        'psutil',
        'requests',
        'httpx',
        'python-dotenv',
        'dotenv',
        'wordcloud',
        'yfinance',
        'geopy',
        'folium',
        'psycopg2',
        'tabulate',
        # GUI and visualization
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        # Network analysis
        'python_louvain',
        'networkx',
        # Standard library modules that might be needed
        'json',
        'os',
        'sys',
        'datetime',
        'logging',
        'io',
        'typing',
        'pathlib',
        're',
        'time',
        'random',
        'traceback',
        # Additional packages with submodules
        'category_encoders.binary',
        'category_encoders.ordinal',
        'category_encoders.woe',
        'category_encoders.target_encoder',
        'category_encoders.leave_one_out',
        'altair.vegalite',
        'altair.utils',
        'altair.vegalite.v4',
        'community',
        'community.community_louvain',
        # Additional visualization libraries
        'bokeh.plotting',
        'bokeh.layouts',
        'bokeh.models',
        'kaleido',
        'kaleido.scopes.plotly',
        'yfinance.utils',
        'geopy.geocoders',
        'folium.plugins'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['mpl_hook.py'],
    excludes=['PyQt5', 'PyQt5.QtWidgets', 'PyQt5.QtCore', 'PyQt5.QtGui'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)


# Exclude unnecessary files to reduce size
def remove_from_list(source, patterns):
    for file in source[:]:
        for pattern in patterns:
            if pattern in file[0]:
                source.remove(file)
                break

remove_from_list(a.datas, [
    'tests', 
    'testing',
    '.html',
    '.png',
    '.jpg',
    '.json'
])

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FirstPass',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep True for debugging
    icon='app_icon.ico' if Path('app_icon.ico').exists() else None
)