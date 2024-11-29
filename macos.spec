# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
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
        # Qt/GUI related
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'PySide6.QtWebEngineWidgets',
        'PySide6.QtWebEngineCore',
        'PySide6.QtWebChannel',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

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
    console=False
)

# Create app bundle for macOS
app = BUNDLE(
    exe,
    name='FirstPass.app',
    icon=None,
    bundle_identifier=None,
)