import sys
import os
from pathlib import Path
import importlib.util
import sqlite3
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QTabWidget,
    QGroupBox, QComboBox, QCheckBox, QSplitter, QFileDialog,
    QMessageBox, QScrollArea, QDialog, QTableWidget, QTableWidgetItem, QFormLayout, 
    QDialogButtonBox, QSizePolicy, QProgressBar, QSpinBox, QGridLayout, QInputDialog
)

from PySide6.QtCore import Qt, QDateTime, QTimer, Signal, QRect
from PySide6.QtGui import QPalette, QColor, QPainter, QPen, QFont, QTextCharFormat, QSyntaxHighlighter
from dark_theme import setup_dark_theme
import json
from eula_dialog import EULADialog
import logging
import re 
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Import our modules
from analyzer import DataAnalyzer, EnhancedJSONEncoder
from llm_handler import LLMHandler
from script_runner import ScriptRunner
from storage import StorageManager
from config import ALLOWED_PACKAGES
from rag_integration import RAGAssistant

# Import new license-related modules
from license_manager import LicenseManager
from license_ui import LicenseDialog, LicenseInvalidDialog
from db_config import DatabaseConfig, LicenseDatabase

# Import our custom components
from data_preview import DataPreviewTab
from knowledge_management import DataDictionaryTab, BusinessKnowledgeTab

import asyncio
from functools import partial

def setup_logging(storage_manager=None):
    """Configure comprehensive logging"""
    # Determine log directory
    if storage_manager:
        log_dir = storage_manager.logs_dir
    else:
        # Default fallback path for initial setup
        if os.name == 'nt':  # Windows
            base_path = Path(os.environ['LOCALAPPDATA']) / 'firstpass'
        else:  # macOS and Linux
            base_path = Path.home() / '.firstpass'
        log_dir = base_path / "logs"
    
    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    log_file = log_dir / f"firstpass_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Reset any existing handlers
    logging.root.handlers = []
    
    # File handler with immediate flush
    file_handler = logging.FileHandler(log_file, 'a', 'utf-8')
    file_handler.setLevel(logging.DEBUG)  # Set to DEBUG level
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set root logger to DEBUG
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Ensure all modules are at DEBUG level
    loggers_to_debug = [
        'rag_integration',
        'llm_handler',
        'knowledge_management',
        'main',
        '__main__',
        'storage',
        'license_manager'
    ]
    
    for logger_name in loggers_to_debug:
        module_logger = logging.getLogger(logger_name)
        module_logger.setLevel(logging.DEBUG)
        module_logger.propagate = True
    
    # Test logging levels
    logger = logging.getLogger(__name__)
    logger.debug("Debug test message")
    logger.info("Info test message")
    logger.warning("Warning test message")
    logger.error("Error test message")
    
    return logger

class ScrollableWidget(QWidget):
    """Widget that supports scrolling"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

class LoadingOverlay(QWidget):
    """Loading overlay that matches app theme"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Make sure overlay covers entire parent
        self.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        # Create container widget with fixed size
        self.container = QWidget()
        self.container.setObjectName("container")
        self.container.setFixedSize(250, 50)  # Much smaller fixed size
        
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(10, 5, 10, 5)  # Reduced margins
        container_layout.setAlignment(Qt.AlignCenter)
        
        # Message label
        self.message_label = QLabel("Processing...")
        self.message_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(self.message_label)
        
        # Add container to main layout
        layout.addWidget(self.container)
        
        # Style to match app theme
        self.setStyleSheet("""
            LoadingOverlay {
                background-color: rgba(0, 0, 0, 150);
            }
            QWidget#container {
                background-color: #2c2c2c;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
            }
            QLabel {
                color: #f0f0f0;
                font-size: 12px;
                background: transparent;
            }
        """)
        
        self.hide()
    
    def showEvent(self, event):
        """Center in parent when shown"""
        super().showEvent(event)
        if self.parentWidget():
            # Set the overlay to cover the entire parent window
            self.setGeometry(self.parentWidget().rect())
            
            # Center the container within the overlay
            container_x = (self.width() - self.container.width()) // 2
            container_y = (self.height() - self.container.height()) // 2
            self.container.move(container_x, container_y)
    
    def set_message(self, message: str):
        """Update the loading message"""
        self.message_label.setText(message)
    
    def hideEvent(self, event):
        super().hideEvent(event)
    
    def set_message(self, message: str):
        """Update the loading message"""
        self.message_label.setText(message)

class PythonSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        # Keywords - Bright Pink
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#FF69B4"))
        keyword_format.setFontWeight(QFont.Bold)
        keywords = [
            "and", "as", "assert", "break", "class", "continue", "def",
            "del", "elif", "else", "except", "False", "finally", "for",
            "from", "global", "if", "import", "in", "is", "lambda", "None",
            "nonlocal", "not", "or", "pass", "raise", "return", "True",
            "try", "while", "with", "yield"
        ]
        for word in keywords:
            self.highlighting_rules.append((f"\\b{word}\\b", keyword_format))

        # Built-in functions - Bright Blue
        builtin_format = QTextCharFormat()
        builtin_format.setForeground(QColor("#00BFFF"))
        builtin_format.setFontWeight(QFont.Bold)
        builtins = [
            "abs", "all", "any", "bin", "bool", "bytearray", "bytes", "chr",
            "classmethod", "compile", "complex", "delattr", "dict", "dir",
            "divmod", "enumerate", "eval", "exec", "filter", "float",
            "format", "frozenset", "getattr", "globals", "hasattr", "hash",
            "help", "hex", "id", "input", "int", "isinstance", "issubclass",
            "iter", "len", "list", "locals", "map", "max", "memoryview",
            "min", "next", "object", "oct", "open", "ord", "pow", "print",
            "property", "range", "repr", "reversed", "round", "set", "setattr",
            "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple",
            "type", "vars", "zip"
        ]
        for word in builtins:
            self.highlighting_rules.append((f"\\b{word}\\b", builtin_format))

        # Strings - Bright Green
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#50FA7B"))
        self.highlighting_rules.append((r'"[^"\\]*(\\.[^"\\]*)*"', string_format))
        self.highlighting_rules.append((r"'[^'\\]*(\\.[^'\\]*)*'", string_format))

        # Numbers - Bright Orange
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#FFB86C"))
        self.highlighting_rules.append(("\\b[0-9]+\\b", number_format))

        # Comments - Bright Purple
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#BD93F9"))
        self.highlighting_rules.append(("#[^\n]*", comment_format))

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            for match in re.finditer(pattern, text):
                start = match.start()
                length = match.end() - match.start()
                self.setFormat(start, length, format)

class OutputTab(QWidget):
    """Tab for displaying script output and visualizations"""
    save_analysis_requested = Signal()
    
    def __init__(self, storage_manager: StorageManager, parent=None):
        super().__init__(parent)
        self.storage_manager = storage_manager
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create horizontal button layout
        button_layout = QHBoxLayout()
        
        # Add clear button
        clear_btn = QPushButton("Clear All Outputs")
        clear_btn.clicked.connect(self.clear_all)
        button_layout.addWidget(clear_btn)
        
        # Add save analysis button
        save_analysis_btn = QPushButton("Save this analysis to improve future outputs")
        save_analysis_btn.clicked.connect(self.save_analysis_requested.emit)
        button_layout.addWidget(save_analysis_btn)
        
        # Make buttons take up equal width
        for btn in (clear_btn, save_analysis_btn):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumHeight(40)
        
        layout.addLayout(button_layout)
        
        # Add output text area
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setPlaceholderText("Script output will appear here...")
        
        # Set monospace font for better table alignment
        font = QFont("DejaVu Sans Mono", 10)  # or "Fira Code"
        if not font.exactMatch():
            font.setFamily("Liberation Mono")  # Fallback option
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(7)  # Small font size
        font.setFixedPitch(True)  # Ensure fixed width
        self.text_output.setFont(font)
        
        # Optimize display settings
        self.text_output.setLineWrapMode(QTextEdit.NoWrap)  # Prevent line wrapping
        self.text_output.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.text_output.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Set dark theme colors using the application's default dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #2c2c2c;
                color: #f0f0f0;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #f0f0f0;
                border: 1px solid #3c3c3c;
                padding: 4px;
                line-height: 1.1;
            }
        """)
        
        layout.addWidget(self.text_output)
        self.setLayout(layout)

    def clear_all(self):
        """Clear all outputs"""
        self.text_output.clear()
        plt.close('all')

class CustomModelDialog(QDialog):
    def __init__(self, provider: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Add Custom {provider} Model")
        self.setModal(True)
        layout = QFormLayout(self)

        # Model name input
        self.name_input = QLineEdit()
        layout.addRow("Model Name:", self.name_input)
        
        # Add helper text
        if provider == "Anthropic":
            helper_text = """
Examples:
- claude-3-5-sonnet-20240620
- claude-3-opus-20240229
- claude-3-haiku-20240307
            """
        else:  # OpenAI
            helper_text = """
Examples:
- gpt-4o-2024-05-13
- gpt-4o-mini-2024-07-18
- o1-preview-2024-09-12
            """
        
        helper_label = QLabel(helper_text)
        helper_label.setStyleSheet("color: gray; font-size: 10pt;")
        layout.addRow("", helper_label)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FirstPass - AI-Assisted Local Data Analysis")

        # Core initialization
        screen = QApplication.primaryScreen().geometry()
        window_width = min(max(int(screen.width() * 0.8), 800), 1600)
        window_height = min(max(int(screen.height() * 0.8), 600), 1000)
        self.setMinimumSize(800, 600)
        self.resize(window_width, window_height)

        app = QApplication.instance()
        font = app.font()
        font.setPointSize(7)
        app.setFont(font)

        # Initialize core managers
        self.storage = StorageManager()
        self.logger = logging.getLogger(__name__)

        # Add EULA check right after storage initialization but before other components
        if not self.check_eula_acceptance():
            # This will exit the application if EULA is declined
            QTimer.singleShot(0, lambda: QApplication.instance().quit())
            return

        # Initialize knowledge base before other components
        self.initialize_knowledge_base()
        self.license_manager = LicenseManager(self.storage)
        self.loading_overlay = LoadingOverlay(self)
        
        # Initialize RAG Assistant
        self.rag_assistant = RAGAssistant(
            storage_manager=self.storage,
            enabled=True
        )

        # Initialize data structures
        self.csv_path = None
        self.metadata = None
        self.conversation_history = []
        self.llm_handler = LLMHandler(self.storage)
        self.script_runner = ScriptRunner(self)
        
        # Initialize model registries
        self.model_registries = {
            "Anthropic": {
                "anthropic://claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet (Latest)",
                "anthropic://claude-3-5-sonnet-20240620": "Claude 3.5 Sonnet",
                "anthropic://claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
                "anthropic://claude-3-opus-20240229": "Claude 3 Opus",
                "anthropic://claude-3-sonnet-20240229": "Claude 3 Sonnet",
                "anthropic://claude-3-haiku-20240307": "Claude 3 Haiku"
            },
            "OpenAI": {
                "openai://gpt-4o": "GPT-4o (Latest)",
                "openai://gpt-4o-mini": "GPT-4o Mini",
                "openai://o1-preview": "O1 Preview",
                "openai://o1-mini": "O1 Mini"
            }
        }

        # Initialize UI
        self.init_ui()
        self.load_saved_settings()

        # Test RAG logging
        self.logger.info("Testing RAG logging...")
        test_result = self.rag_assistant.test_logging()
        self.logger.info(f"RAG logging test result: {test_result}")

        # Critical license check
        if not self.check_license():
            self.logger.warning("No valid license found - exiting application")
            QTimer.singleShot(0, lambda: QApplication.instance().quit())
            return
        
        # Initialize license check timer
        self.license_check_timer = QTimer(self)
        self.license_check_timer.timeout.connect(self.check_license)
        self.license_check_timer.start(86400000)  # Check every 24 hours

    def check_eula_acceptance(self):
        """Check EULA acceptance and show dialog if needed"""
        if not EULADialog.check_acceptance(self.storage):
            dialog = EULADialog(self.storage, self)
            if dialog.exec() != QDialog.Accepted:
                # User declined EULA
                QMessageBox.critical(
                    self,
                    "EULA Not Accepted",
                    "You must accept the EULA to use this application."
                )
                return False
        return True

    def initialize_knowledge_base(self):
        """Initialize the knowledge base"""
        try:
            # Define knowledge base path in the data directory
            kb_path = self.storage.data_dir / "base_knowledge.db"
            
            if not kb_path.exists():
                self.logger.info("Base knowledge database not found. Starting without base knowledge.")
                # Create an empty database with required tables
                conn = sqlite3.connect(str(kb_path))
                cursor = conn.cursor()
                
                # Create essential tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analysis_patterns (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS column_definitions (
                        id INTEGER PRIMARY KEY,
                        column_name TEXT,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS business_rules (
                        id INTEGER PRIMARY KEY,
                        rule_name TEXT,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                conn.close()
                
                self.logger.info("Created empty knowledge base")
            else:
                self.logger.info("Using existing knowledge base")
                
        except Exception as e:
            self.logger.error(f"Error during knowledge base initialization: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to initialize knowledge base: {str(e)}"
            )

    def show_loading(self, message="Processing..."):
        """Show loading overlay"""
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.set_message(message)
            self.loading_overlay.resize(self.size())
            self.loading_overlay.move(0, 0)  # Ensure it's positioned at top-left
            self.loading_overlay.raise_()    # Bring to front
            self.loading_overlay.show()
            
            # Force immediate update
            self.loading_overlay.repaint()
            QApplication.processEvents()
        
    def hide_loading(self):
        """Hide loading overlay"""
        self.loading_overlay.hide()
        
    # ============ NEW LICENSE MANAGEMENT METHODS ============
    def check_license(self) -> bool:
        """Check license validity"""
        self.logger.debug("Checking license...")
        
        # First check if we have a saved license
        saved_key = self.license_manager.get_active_license_key()
        
        if saved_key:
            # For existing installations, use validate endpoint
            result = self.license_manager.validate_license()
            if not result.get('valid'):
                error_msg = result.get('message')
                self.logger.warning(f"License invalid: {error_msg}")
                dialog = LicenseInvalidDialog(error_msg, self)
                response = dialog.exec()  # Use exec() instead of exec_()
                
                if response == QDialog.Accepted:
                    # Show license input dialog for re-activation
                    self.logger.debug("Showing license input dialog")
                    license_dialog = LicenseDialog(self.license_manager, self)
                    if license_dialog.exec() == QDialog.Accepted:  # Use exec() instead of exec_()
                        self.update_license_status()
                        return True
                    else:
                        return False
                elif response == 2:  # Exit application
                    self.logger.debug("User chose to exit application")
                    self.close()
                    return False
                
                return False
        else:
            # For new installations, show activation dialog
            dialog = LicenseDialog(self.license_manager, self)
            response = dialog.exec()  # Use exec() instead of exec_()
            
            if response == QDialog.Accepted:
                # Check if activation was successful
                result = self.license_manager.validate_license()
                if not result.get('valid'):
                    error_msg = result.get('message', 'License validation failed')
                    self.logger.warning(f"License invalid: {error_msg}")
                    return False
            else:
                return False
                
        self.logger.info("License check passed")
        self.update_license_status()
        return True
    # ====================================================

    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def init_ui(self):
        """Initialize user interface with aligned windows"""
        # Create main scroll area
        main_scroll = QScrollArea()
        main_scroll.setWidgetResizable(True)
        main_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setCentralWidget(main_scroll)

        # Create main widget and layout
        main_widget = QWidget()
        main_scroll.setWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create main tab widget
        self.tab_widget = QTabWidget()
        
        # Create App Settings tab
        settings_tab = self.create_settings_tab()
        self.tab_widget.addTab(settings_tab, "App Settings")

        # Create input tab with scroll area
        input_scroll = QScrollArea()
        input_scroll.setWidgetResizable(True)
        input_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        input_tab = QWidget()
        input_layout = QVBoxLayout(input_tab)
        input_layout.setSpacing(10)
        input_layout.setContentsMargins(5, 5, 5, 5)

        # History Management at the top with reduced padding
        history_group = QGroupBox("Conversation History")
        history_layout = QHBoxLayout()
        history_layout.setContentsMargins(5, 5, 5, 5)
        history_layout.setSpacing(5)
        
        view_history_btn = QPushButton("View History")
        view_history_btn.clicked.connect(self.view_history)
        save_history_btn = QPushButton("Save History")
        save_history_btn.clicked.connect(self.save_history)
        load_history_btn = QPushButton("Load History")
        load_history_btn.clicked.connect(self.load_history)
        clear_history_btn = QPushButton("Clear History")
        clear_history_btn.clicked.connect(self.clear_history)
        
        history_layout.addWidget(view_history_btn)
        history_layout.addWidget(save_history_btn)
        history_layout.addWidget(load_history_btn)
        history_layout.addWidget(clear_history_btn)
        
        history_group.setLayout(history_layout)
        input_layout.addWidget(history_group)

        # File Selection in its own horizontal section
        file_group = QGroupBox("CSV File Selection")
        file_layout = QHBoxLayout()
        file_layout.setContentsMargins(5, 5, 5, 5)
        self.file_path_label = QLabel("No file selected")
        select_file_btn = QPushButton("Select CSV")
        select_file_btn.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(select_file_btn)
        file_group.setLayout(file_layout)
        input_layout.addWidget(file_group)

        # Create main horizontal splitter for two columns
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        
        # Left column container
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Left vertical splitter
        left_splitter = QSplitter(Qt.Vertical)
        
        # Metadata section
        metadata_group = QGroupBox("CSV Metadata")
        metadata_layout = QVBoxLayout()
        metadata_layout.setContentsMargins(5, 5, 5, 5)
        metadata_header = QHBoxLayout()
        clear_metadata_btn = QPushButton("Clear")
        clear_metadata_btn.clicked.connect(lambda: self.metadata_text.clear())
        metadata_header.addStretch()
        metadata_header.addWidget(clear_metadata_btn)
        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        metadata_layout.addLayout(metadata_header)
        metadata_layout.addWidget(self.metadata_text)
        metadata_group.setLayout(metadata_layout)
        left_splitter.addWidget(metadata_group)

        # Error Log section
        error_group = QGroupBox("Error Log (Editable)")
        error_layout = QVBoxLayout()
        error_layout.setContentsMargins(5, 5, 5, 5)
        error_header = QHBoxLayout()
        clear_error_btn = QPushButton("Clear")
        clear_error_btn.clicked.connect(self.clear_error_log)
        error_header.addStretch()
        error_header.addWidget(clear_error_btn)
        self.error_text = QTextEdit()
        self.error_text.setPlaceholderText("""LLMs sometimes produce errors in their generated scripts.

If you run a script and encounter a script error, the error will populate here.

Click 'Generate Python Script' again and the new prompt will instruct the LLM to avoid errors populated here. This can help resolve some but not all errors.

Clear the error log to no longer send the error message as part of your prompt.""")
        error_layout.addLayout(error_header)
        error_layout.addWidget(self.error_text)
        error_group.setLayout(error_layout)
        left_splitter.addWidget(error_group)

        # Add left splitter to left layout
        left_layout.addWidget(left_splitter)
        
        # Right column container
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Create top section for prompt and generate button
        top_section = QWidget()
        top_layout = QVBoxLayout(top_section)

        # LLM Prompt
        prompt_group = QGroupBox("LLM Prompt")
        prompt_layout = QVBoxLayout()
        prompt_layout.setContentsMargins(5, 5, 5, 5)
        prompt_header = QHBoxLayout()
        clear_prompt_btn = QPushButton("Clear")
        clear_prompt_btn.clicked.connect(lambda: self.prompt_input.clear())
        prompt_header.addStretch()
        prompt_header.addWidget(clear_prompt_btn)
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter your analysis request here. The more specific you can be with your request, the more robust the analyses will be.")
        prompt_layout.addLayout(prompt_header)
        prompt_layout.addWidget(self.prompt_input)
        prompt_group.setLayout(prompt_layout)
        top_layout.addWidget(prompt_group)

        # Generate Button
        generate_btn = QPushButton("Generate Python Script")
        generate_btn.clicked.connect(self.generate_script)
        top_layout.addWidget(generate_btn)

        # Bottom section for script and run button
        bottom_section = QWidget()
        bottom_layout = QVBoxLayout(bottom_section)

        # Script Output
        script_group = QGroupBox("Generated Script")
        script_layout = QVBoxLayout()
        script_layout.setContentsMargins(5, 5, 5, 5)
        script_header = QHBoxLayout()
        clear_script_btn = QPushButton("Clear")
        clear_script_btn.clicked.connect(lambda: self.script_output.clear())
        script_header.addStretch()
        script_header.addWidget(clear_script_btn)
        self.script_output = QTextEdit()
        self.script_output.setPlaceholderText("Generated script will appear here...")

        # Set monospace font and styling for script output
        font = QFont("Fira Code", 8)  # or "Fira Code"
        if not font.exactMatch():
            font.setFamily("Liberation Mono")  # Fallback option
        font.setStyleHint(QFont.Monospace)
        font.setFixedPitch(True)
        self.script_output.setFont(font)

        self.script_output.setStyleSheet("""
            QTextEdit {
                background-color: #282A36;
                color: #F8F8F2;
                border: 1px solid #44475A;
                padding: 4px;
            }
        """)

        self.python_highlighter = PythonSyntaxHighlighter(self.script_output.document())
        script_layout.addLayout(script_header)
        script_layout.addWidget(self.script_output)
        script_group.setLayout(script_layout)
        bottom_layout.addWidget(script_group)

        # Run Button
        run_btn = QPushButton("Run Script")
        run_btn.clicked.connect(self.run_script)
        bottom_layout.addWidget(run_btn)

        # Create and add the vertical splitter
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(top_section)
        right_splitter.addWidget(bottom_section)

        # Add splitter to right layout
        right_layout.addWidget(right_splitter)

        # Add widgets to main splitter
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([50, 50])
        input_layout.addWidget(main_splitter)
        
        # Set input tab layout
        input_scroll.setWidget(input_tab)
        
        # Add tabs
        self.tab_widget.addTab(input_scroll, "Create Prompt")
        
        self.data_preview_tab = DataPreviewTab()
        self.tab_widget.addTab(self.data_preview_tab, "Data Preview")
        
        self.output_tab = OutputTab(storage_manager=self.storage)
        self.output_tab.save_analysis_requested.connect(self.add_current_analysis)
        self.tab_widget.addTab(self.output_tab, "Output")

        self.analysis_patterns_tab = self.create_rag_management_tab()
        self.tab_widget.addTab(self.analysis_patterns_tab, "Analysis Patterns")
        
        self.data_dictionary_tab = DataDictionaryTab(self.rag_assistant, self.storage)
        self.tab_widget.addTab(self.data_dictionary_tab, "Data Dictionary")

        # self.business_knowledge_tab = BusinessKnowledgeTab(self.rag_assistant, self.storage)
        # self.tab_widget.addTab(self.business_knowledge_tab, "Business Context")
        
        # Add tab widget to main layout
        layout.addWidget(self.tab_widget)

    def on_metadata_setting_changed(self, state):
        """Handle changes to metadata configuration checkboxes"""
        # Use print for immediate console output
        print(f"\nMetadata setting changed!")
        print(f"Include Stats: {self.include_stats_checkbox.isChecked()}")
        print(f"Include Samples: {self.include_samples_checkbox.isChecked()}\n")
        
        # Log at INFO level instead of DEBUG
        self.logger.info(f"Metadata setting changed - Stats: {self.include_stats_checkbox.isChecked()}, Samples: {self.include_samples_checkbox.isChecked()}")
        
        # Save settings whenever they change
        self.save_settings()

    def create_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()



        # License Status Group
        license_group = QGroupBox("License Status")
        license_layout = QHBoxLayout()
        self.license_status_label = QLabel("Checking license...")
        view_license_btn = QPushButton("View License Details")
        view_license_btn.clicked.connect(self.show_license_details)
        license_layout.addWidget(self.license_status_label)
        license_layout.addWidget(view_license_btn)
        license_group.setLayout(license_layout)
        layout.addWidget(license_group)

        # Add RAG Status Group 
        rag_group = QGroupBox("Retrieval Augmentation Generation (RAG)")
        rag_layout = QVBoxLayout()

        help_label = QLabel("RAG allows you to enhance your prompt with contextual information. If you often use CSVs with column names that don't have clear meanings, you can map an explanation of the column data to the column name in the data dictionary. Enabling the RAG assistant will capture that context.")
        help_label.setWordWrap(True)
        rag_layout.addWidget(help_label)

        # RAG checkbox with tooltip
        checkbox_layout = QHBoxLayout()
        self.rag_enabled_checkbox = QCheckBox("Enable RAG Assistant")
        self.rag_enabled_checkbox.setChecked(True)
        self.rag_enabled_checkbox.stateChanged.connect(self.toggle_rag)

        rag_info = QPushButton("?")
        rag_info.setFixedSize(16, 16)
        rag_info.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                border: none;
                border-radius: 8px;
                color: #ffffff;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #505050;
                color: #ffffff;
            }
        """)
        rag_info.setToolTip("This is currently an experimental feature in FirstPass. For more predictable outputs, keep the RAG assistant turned off.")

        checkbox_layout.addWidget(self.rag_enabled_checkbox)
        checkbox_layout.addWidget(rag_info)
        checkbox_layout.addStretch()
        rag_layout.addLayout(checkbox_layout)

        # Stats label
        self.stats_label = QLabel("Loading stats...")
        self.update_knowledge_base_stats()
        rag_layout.addWidget(self.stats_label)

        rag_group.setLayout(rag_layout)
        layout.addWidget(rag_group)

        # Metadata Configuration Group
        metadata_group = QGroupBox("Metadata Configuration")
        metadata_layout = QVBoxLayout()

        help_label = QLabel("Control what information is sent to the AI model. More information can lead to better analysis but may expose more data to the LLM and increase token usage.")
        help_label.setWordWrap(True)
        metadata_layout.addWidget(help_label)

        # Stats checkbox with tooltip
        stats_layout = QHBoxLayout()
        self.include_stats_checkbox = QCheckBox("Include Summary Statistics")
        stats_info = QPushButton("?")
        stats_info.setFixedSize(16, 16)
        stats_info.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                border: none;
                border-radius: 8px;
                color: #ffffff;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #505050;
                color: #ffffff;
            }
        """)
        stats_info.setToolTip("Include min, max, mean, median, and distribution information")
        stats_layout.addWidget(self.include_stats_checkbox)
        stats_layout.addWidget(stats_info)
        stats_layout.addStretch()
        metadata_layout.addLayout(stats_layout)

        # Samples checkbox with tooltip
        samples_layout = QHBoxLayout()
        self.include_samples_checkbox = QCheckBox("Include Sample Data")
        samples_info = QPushButton("?")
        samples_info.setFixedSize(16, 16)
        samples_info.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                border: none;
                border-radius: 8px;
                color: #a0a0a0;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #505050;
                color: #ffffff;
            }
        """)
        samples_info.setToolTip("Include five example values from each column (excludes sensitive data where detected)")
        samples_layout.addWidget(self.include_samples_checkbox)
        samples_layout.addWidget(samples_info)
        samples_layout.addStretch()
        metadata_layout.addLayout(samples_layout)

        # Apply tooltip styling
        self.setStyleSheet("""
            QToolTip {
                background-color: #2c2c2c;
                color: #e0e0e0;
                border: 1px solid #505050;
                padding: 4px;
            }
        """)

        # Connect checkbox signals
        self.include_stats_checkbox.setChecked(True)
        self.include_samples_checkbox.setChecked(True)
        self.include_stats_checkbox.stateChanged.connect(self.on_metadata_setting_changed)
        self.include_samples_checkbox.stateChanged.connect(self.on_metadata_setting_changed)

        metadata_group.setLayout(metadata_layout)
        layout.addWidget(metadata_group)

        # Fixed providers section
        fixed_group = QGroupBox("API Configuration")
        fixed_layout = QFormLayout()

        key_layout = QHBoxLayout()
        self.anthropic_key = QLineEdit()
        self.anthropic_key.setEchoMode(QLineEdit.Password)
        self.save_anthropic_key = QCheckBox("Save API Key")
        self.save_anthropic_key.stateChanged.connect(self.save_settings)
        key_layout.addWidget(self.anthropic_key)
        key_layout.addWidget(self.save_anthropic_key)
        fixed_layout.addRow("Anthropic API Key:", key_layout)

        openai_layout = QHBoxLayout()
        self.openai_key = QLineEdit()
        self.openai_key.setEchoMode(QLineEdit.Password)
        self.save_openai_key = QCheckBox("Save API Key")
        self.save_openai_key.stateChanged.connect(self.save_settings)
        openai_layout.addWidget(self.openai_key)
        openai_layout.addWidget(self.save_openai_key)
        fixed_layout.addRow("OpenAI API Key:", openai_layout)

        fixed_group.setLayout(fixed_layout)
        layout.addWidget(fixed_group)

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout()

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Anthropic", "OpenAI"])
        model_layout.addRow("Provider:", self.provider_combo)

        self.model_combo = QComboBox()
        self.update_model_choices("Anthropic")
        self.provider_combo.currentTextChanged.connect(self.update_model_choices)
        model_layout.addRow("Model:", self.model_combo)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Add stretch at the bottom
        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def update_model_choices(self, provider: str):
        self.model_combo.clear()
        
        # Add registered models
        for model_id, display_name in self.model_registries[provider].items():
            self.model_combo.addItem(display_name, model_id)
        
        # Add custom model option
        self.model_combo.addItem("Custom Model...", "custom")
        
        # Connect to handle custom model selection
        self.model_combo.currentIndexChanged.connect(self.handle_custom_model)

    def handle_custom_model(self, index):
        """Handle selection of custom model option"""
        if self.model_combo.itemData(index) == "custom":
            provider = self.provider_combo.currentText()
            dialog = CustomModelDialog(provider, self)
            
            if dialog.exec_() == QDialog.Accepted:
                custom_model = dialog.name_input.text().strip()
                if custom_model:
                    # Block signals during updates
                    self.model_combo.blockSignals(True)
                    
                    model_id = f"{provider.lower()}://{custom_model}"
                    display_name = f"{custom_model} (Custom)"
                    self.model_registries[provider][model_id] = display_name
                    
                    # Update combo box
                    self.model_combo.removeItem(index)
                    self.model_combo.addItem(display_name, model_id)
                    self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
                    self.model_combo.addItem("Custom Model...", "custom")
                    
                    # Unblock signals after updates
                    self.model_combo.blockSignals(False)
            else:
                # Block signals during update
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentIndex(max(0, index - 1))
                self.model_combo.blockSignals(False)

    def save_model_selection(self, index):
        """Save model selection when changed"""
        settings = self.storage.load_provider_settings()
        settings.update({
            'provider': self.provider_combo.currentText(),
            'model_id': self.model_combo.itemData(index)
        })
        self.storage.save_provider_settings(settings)
        self.logger.info(f"Saved model selection: {self.model_combo.currentText()}")

    def update_model_choices(self, provider: str):
        """Update model choices based on selected provider"""
        self.model_combo.clear()
        
        # Block signals during updates
        self.model_combo.blockSignals(True)
        
        # Add registered models
        for model_id, display_name in self.model_registries[provider].items():
            self.model_combo.addItem(display_name, model_id)
        
        # Add custom model option
        self.model_combo.addItem("Custom Model...", "custom")
        
        # Unblock signals before connecting
        self.model_combo.blockSignals(False)
        
        # Connect signals after populating
        try:
            self.model_combo.currentIndexChanged.disconnect()
        except TypeError:
            pass  # No connections to disconnect
        
        self.model_combo.currentIndexChanged.connect(self.save_model_selection)
        self.model_combo.currentIndexChanged.connect(self.handle_custom_model)

    def load_saved_settings(self):
        """Load saved settings including API keys, selected model, metadata configuration, and RAG state"""
        settings = self.storage.load_provider_settings()
        keys = self.storage.load_api_keys()
        
        if 'Anthropic' in keys:
            self.anthropic_key.setText(keys['Anthropic']['key'])
            self.save_anthropic_key.setChecked(True)
        if 'OpenAI' in keys:
            self.openai_key.setText(keys['OpenAI']['key'])
            self.save_openai_key.setChecked(True)
        
        # Set saved provider and model if they exist
        saved_provider = settings.get('provider', 'Anthropic')
        saved_model_id = settings.get('model_id', 'anthropic://claude-3-5-sonnet-20241022')
        
        # Set provider
        index = self.provider_combo.findText(saved_provider)
        if index >= 0:
            self.provider_combo.setCurrentIndex(index)
        
        # Update model choices for the provider
        self.update_model_choices(saved_provider)
        
        # Set model
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == saved_model_id:
                self.model_combo.setCurrentIndex(i)
                break
        
        # Load metadata settings
        metadata_settings = settings.get('metadata', {})
        self.include_stats_checkbox.setChecked(metadata_settings.get('include_stats', True))
        self.include_samples_checkbox.setChecked(metadata_settings.get('include_samples', True))
        
        # Load RAG state with a default of False if not found
        rag_enabled = settings.get('rag_enabled', False)
        self.rag_enabled_checkbox.setChecked(rag_enabled)
        self.rag_assistant.enabled = rag_enabled  # Make sure to update the assistant's state too

    def save_settings(self):
        keys = {}
        if self.save_anthropic_key.isChecked() and self.anthropic_key.text():
            keys['Anthropic'] = {'key': self.anthropic_key.text()}
        if self.save_openai_key.isChecked() and self.openai_key.text():
            keys['OpenAI'] = {'key': self.openai_key.text()}
        
        # Get current model selection
        current_model_id = self.model_combo.currentData()
        current_provider = self.provider_combo.currentText()
        
        settings = {
            'provider': current_provider,
            'model_id': current_model_id,
            'model_display_name': self.model_combo.currentText(),
            'metadata': {
                'include_stats': self.include_stats_checkbox.isChecked(),
                'include_samples': self.include_samples_checkbox.isChecked()
            },
            'rag_enabled': self.rag_enabled_checkbox.isChecked()
        }
        
        self.storage.save_api_keys(keys)
        self.storage.save_provider_settings(settings)

    # ============ NEW LICENSE UI METHODS ============
    def show_license_details(self):
        """Show current license details in a formatted dialog"""
        try:
            # Try to validate current license first
            validation_result = self.license_manager.validate_license()
            
            if not validation_result['valid']:
                QMessageBox.warning(self, "License Details", "No valid license found")
                return

            # Get full license info
            license_info = self.license_manager.get_license_info()
            if license_info:
                # Create a formatted message with key information
                msg_parts = []
                
                # Basic status info
                msg_parts.append(f"Status: {license_info['status'].title()}")
                msg_parts.append(f"Type: {license_info['license_type']}")
                
                # Usage information
                msg_parts.append(f"\nActive Installations: {license_info['active_instances']} of {license_info['activation_limit']}")
                
                # Dates
                if license_info.get('expiration'):
                    msg_parts.append(f"\nExpires: {license_info['expiration']}")
                if license_info.get('next_renewal'):
                    msg_parts.append(f"Next Renewal: {license_info['next_renewal']}")
                
                # Product info
                if license_info.get('product'):
                    msg_parts.append(f"\nProduct: {license_info['product']['name']}")
                    if license_info['product'].get('variant'):
                        msg_parts.append(f"Variant: {license_info['product']['variant']}")
                
                # Customer info if available
                if license_info.get('customer'):
                    msg_parts.append(f"\nLicensed to: {license_info['customer'].get('name', 'N/A')}")
                    msg_parts.append(f"Email: {license_info['customer'].get('email', 'N/A')}")
                
                # Machine ID for support purposes
                msg_parts.append(f"\nMachine ID: {self.license_manager._get_machine_id()[:16]}...")
                
                # Join all parts with newlines
                full_message = "\n".join(msg_parts)
                
                # Show in a dialog
                QMessageBox.information(self, "License Details", full_message)
            else:
                QMessageBox.warning(self, "License Details", 
                                "Could not retrieve license details. Please check your internet connection.")
        except Exception as e:
            self.logger.error(f"Error showing license details: {str(e)}")
            QMessageBox.critical(self, "Error", 
                            f"Error retrieving license details: {str(e)}")

    def update_license_status(self):
        """Update license status display"""
        try:
            result = self.license_manager.validate_license()
            if result and result.get('valid'):
                # Format expiration date if available
                expires_at = result.get('expires_at', '')
                if expires_at:
                    expires_at = expires_at.split('T')[0]  # Get just the date part
                    status_text = f"Active • Expires: {expires_at}"
                else:
                    status_text = "Active"
                
                self.license_status_label.setText(status_text)
                self.license_status_label.setStyleSheet("color: #2ecc71; font-weight: bold;")  # Green
            else:
                self.license_status_label.setText("License Required")
                self.license_status_label.setStyleSheet("color: #e74c3c;")  # Red
        except Exception as e:
            self.logger.error(f"Error updating license status: {str(e)}")
            self.license_status_label.setText("License Error")
            self.license_status_label.setStyleSheet("color: #e74c3c;")  # Red
    # =============================================

    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        if event.size().width() < 800:
            self.resize(800, event.size().height())
        if event.size().height() < 600:
            self.resize(event.size().width(), 600)
        # Add this line to handle loading overlay resize
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.resize(self.size())

    def load_saved_api_key(self):
        """Load saved API keys if available"""
        try:
            saved_keys = self.storage.load_api_keys()
            if saved_keys:
                if 'Anthropic' in saved_keys:
                    self.anthropic_key.setText(saved_keys['Anthropic']['key'])
                    self.save_anthropic_key.setChecked(True)
                if 'OpenAI' in saved_keys:
                    self.openai_key.setText(saved_keys['OpenAI']['key'])
                    self.save_openai_key.setChecked(True)
        except Exception as e:
            self.logger.error(f"Error loading API keys: {str(e)}")
            self.log_error(f"Failed to load saved API keys: {str(e)}")

    def select_file(self):
        """Handle file selection and analysis"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Select CSV File",
                "",
                "CSV Files (*.csv)"
            )
            
            if file_path:
                self.csv_path = str(Path(file_path))
                self.file_path_label.setText(os.path.basename(file_path))
                
                # Read and analyze file
                df = pd.read_csv(file_path)
                analyzer = DataAnalyzer()
                self.metadata = analyzer.analyze_file(Path(file_path))
                
                # Update displays
                self.metadata_text.setText(
                    json.dumps(self.metadata, indent=2, cls=EnhancedJSONEncoder)
                )
                self.data_preview_tab.load_data(df)
                
        except Exception as e:
            self.logger.error(f"Error selecting file: {str(e)}")
            self.log_error(str(e))
            QMessageBox.critical(self, "Error", str(e))

    def generate_script(self):
        """Generate Python script using selected LLM provider"""
        if not self.metadata or not self.csv_path:
            QMessageBox.warning(self, "Warning", "Please select and analyze a CSV file first")
            return

        provider = self.provider_combo.currentText()
        model_id = self.model_combo.itemData(self.model_combo.currentIndex())
        
        # Add immediate console output for toggle states
        print("\nGenerating script with metadata settings:")
        print(f"Include Stats: {self.include_stats_checkbox.isChecked()}")
        print(f"Include Samples: {self.include_samples_checkbox.isChecked()}\n")
        
        try:
            # Ensure metadata is a dictionary
            if isinstance(self.metadata, str):
                try:
                    metadata_dict = json.loads(self.metadata)
                except json.JSONDecodeError:
                    raise ValueError("Invalid metadata format")
            else:
                metadata_dict = self.metadata

            # Initialize the appropriate client
            if provider == "Anthropic":
                if not self.anthropic_key.text():
                    QMessageBox.warning(self, "Warning", "Please enter your Anthropic API key")
                    return
                self.llm_handler.initialize("Anthropic", self.anthropic_key.text())
            else:  # OpenAI
                if not self.openai_key.text():
                    QMessageBox.warning(self, "Warning", "Please enter your OpenAI API key")
                    return
                self.llm_handler.initialize("OpenAI", self.openai_key.text())
                
            # Show loading overlay
            self.show_loading("Generating script... please wait")
            QApplication.processEvents()

            # Verbose debugging for RAG status
            self.logger.debug(f"RAG Assistant enabled: {self.rag_assistant.enabled}")
            self.logger.debug(f"RAG Assistant loaded patterns: {len(self.rag_assistant.get_custom_patterns())}")
            
            # Get base prompt
            base_prompt = self.prompt_input.toPlainText()
            self.logger.debug(f"Base prompt: {base_prompt}")

            # Log metadata before enhancement
            self.logger.debug("Current metadata for RAG enhancement:")
            self.logger.debug(json.dumps(self.metadata, indent=2, cls=EnhancedJSONEncoder))

            # Enhance prompt with RAG if enabled
            self.logger.debug("Starting RAG enhancement process")
            
            # Get column definitions
            column_defs = self.rag_assistant.get_column_definitions()
            self.logger.debug(f"Found {len(column_defs)} column definitions")
            
            # Get business knowledge
            business_rules = self.rag_assistant.get_business_knowledge()
            self.logger.debug(f"Found {len(business_rules)} business rules")
            
            enhanced_prompt = self.rag_assistant.enhance_prompt(
                base_prompt,
                self.metadata
            )
            
            # Log the difference between base and enhanced prompts
            self.logger.debug(f"Base prompt length: {len(base_prompt)}")
            self.logger.debug(f"Enhanced prompt length: {len(enhanced_prompt)}")
            self.logger.debug("Enhancement delta:")
            self.logger.debug(enhanced_prompt.replace(base_prompt, "...BASE_PROMPT..."))
            
            # Get provider settings
            settings = self.storage.load_provider_settings()

            # Get current metadata settings with logging
            include_stats = self.include_stats_checkbox.isChecked()
            include_samples = self.include_samples_checkbox.isChecked()
            
            # Log settings at INFO level
            self.logger.info(f"Generating script with metadata settings - Stats: {include_stats}, Samples: {include_samples}")

            # Generate script with provider, model, and metadata settings
            script = self.llm_handler.generate_script(
                prompt=self.prompt_input.toPlainText(),
                metadata=metadata_dict,  # Pass the dictionary version
                model_id=model_id,
                rag_assistant=self.rag_assistant,
                include_stats=include_stats,
                include_samples=include_samples
            )
            
            self.script_output.setText(script)
            
            # Add to conversation history with enhanced metadata
            history_entry = {
                "timestamp": QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss"),
                "provider": provider,
                "model": model_id,
                "request": base_prompt,
                "enhanced_prompt": enhanced_prompt,
                "response": script,
                "metadata": self.metadata,
                "csv_path": self.csv_path,
                "settings_used": settings,
                "rag_enabled": self.rag_assistant.enabled,
                "column_definitions": len(column_defs),
                "business_rules": len(business_rules)
            }
            
            self.conversation_history.append(history_entry)
            self.logger.debug("Added to conversation history")
            
        except Exception as e:
            self.logger.error(f"Error generating script: {str(e)}")
            self.log_error(str(e))
            QMessageBox.critical(self, "Error", str(e))
        finally:
            # Always hide loading overlay
            self.hide_loading()
            QApplication.processEvents()  # Force UI update

    def run_script(self):
        """Run the generated script"""
        if not self.script_output.toPlainText():
            QMessageBox.warning(self, "Warning", "No script to run")
            return
                
        try:
            # Switch to output tab first
            self.tab_widget.setCurrentWidget(self.output_tab)
            self.output_tab.text_output.clear()
            
            # Show loading overlay
            self.show_loading("Running script... please wait")
            QApplication.processEvents()  # Force UI update
            
            result = self.script_runner.run_script(
                script=self.script_output.toPlainText(),
                local_vars={'csv_path': self.csv_path},
                stdout_callback=self.output_tab.text_output.append,
                stderr_callback=self.log_error
            )
            
            if not result['success']:
                # MODIFIED LINES START
                self.log_error(result['stderr'])
                QMessageBox.critical(self, "Error", 
                    "Script execution failed. See error log for details.")
                # MODIFIED LINES END
                    
        except Exception as e:
            error_msg = f"Application Error: {str(e)}"
            self.logger.error(error_msg)
            self.log_error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
        finally:
            # Always hide loading overlay
            self.hide_loading()
            QApplication.processEvents()  # Force UI update

    def log_error(self, error_msg: str):
        """Log error to error display"""
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        formatted_error = f"[{timestamp}] {error_msg}"
        self.error_text.append(formatted_error)

    def clear_error_log(self):
        """Clear the error log"""
        self.error_text.clear()
        self.llm_handler.error_history.clear()

    def view_history(self):
        """View conversation history"""


        if not self.conversation_history:
            QMessageBox.information(self, "History", "No conversation history available")
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("Conversation History")
        dialog.setMinimumSize(800, 600)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Add search box
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        search_input = QLineEdit()
        search_layout.addWidget(search_label)
        search_layout.addWidget(search_input)
        layout.addLayout(search_layout)
        
        # Add text display with scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        text_display = QTextEdit()
        text_display.setReadOnly(True)
        scroll.setWidget(text_display)
        layout.addWidget(scroll)
        
        # Function to update display based on search
        def update_display(search_text=""):
            history_text = ""
            for entry in self.conversation_history:
                if (search_text.lower() in entry['request'].lower() or 
                    search_text.lower() in entry['response'].lower()):
                    history_text += f"\n{'='*80}\n"
                    history_text += f"Timestamp: {entry['timestamp']}\n"
                    history_text += f"\nRequest:\n{entry['request']}\n"
                    history_text += f"\nResponse:\n{entry['response']}\n"
            
            text_display.setPlainText(history_text)
        
        # Connect search box to update function
        search_input.textChanged.connect(update_display)
        
        # Add button layout
        button_layout = QHBoxLayout()
        
        # Add load button
        load_btn = QPushButton("Load Selected")
        def load_selected():
            cursor = text_display.textCursor()
            selected_text = cursor.selectedText()
            if selected_text:
                # Try to find the corresponding entry
                for entry in self.conversation_history:
                    if entry['response'] in selected_text:
                        self.prompt_input.setText(entry['request'])
                        self.script_output.setText(entry['response'])
                        dialog.accept()
                        return
        load_btn.clicked.connect(load_selected)
        button_layout.addWidget(load_btn)
        
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        
        # Initial display
        update_display()
        
        dialog.exec_()

    def save_history(self):
        """Save conversation history to file"""


        if not self.conversation_history:
            QMessageBox.warning(self, "Warning", "No conversation history to save")
            return
            
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Conversation History",
                str(Path.home()),
                "JSON Files (*.json)"
            )
            
            if file_path:
                if not file_path.endswith('.json'):
                    file_path += '.json'
                    
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.conversation_history, f, indent=2, cls=EnhancedJSONEncoder)
                    
                QMessageBox.information(
                    self,
                    "Success",
                    f"Conversation history saved to:\n{file_path}"
                )
                
        except Exception as e:
            self.logger.error(f"Error saving history: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save history: {str(e)}")

    def load_history(self):
        """Load conversation history from file"""


        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Conversation History",
                str(Path.home()),
                "JSON Files (*.json)"
            )
            
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_history = json.load(f)
                
                if not isinstance(loaded_history, list):
                    raise ValueError("Invalid history file format")
                
                # Validate history entries
                for entry in loaded_history:
                    required_fields = ['timestamp', 'request', 'response']
                    if not all(field in entry for field in required_fields):
                        raise ValueError("Invalid history entry format")
                
                reply = QMessageBox.question(
                    self,
                    "Load History",
                    "Do you want to merge with existing history?\nClick 'No' to replace existing history.",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
                )
                
                if reply == QMessageBox.Yes:
                    # Merge histories based on timestamps
                    existing_timestamps = {entry['timestamp'] for entry in self.conversation_history}
                    for entry in loaded_history:
                        if entry['timestamp'] not in existing_timestamps:
                            self.conversation_history.append(entry)
                            
                    # Sort merged history by timestamp
                    self.conversation_history.sort(key=lambda x: x['timestamp'])
                    
                elif reply == QMessageBox.No:
                    self.conversation_history = loaded_history
                else:
                    return
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"Loaded {len(loaded_history)} history entries successfully"
                )
                
        except Exception as e:
            self.logger.error(f"Error loading history: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load history: {str(e)}")

    def clear_history(self):
        """Clear conversation history"""
        if not self.conversation_history:
            QMessageBox.information(self, "Info", "No conversation history to clear")
            return
            
        reply = QMessageBox.question(
            self,
            "Clear History",
            f"Are you sure you want to clear all conversation history ({len(self.conversation_history)} entries)?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.conversation_history.clear()
            QMessageBox.information(self, "Success", "Conversation history cleared")

    # ============ NEW LICENSE EVENT HANDLERS ============
    def closeEvent(self, event):
        """Handle application close event"""
        try:
            # Stop the periodic license check
            if hasattr(self, 'license_check_timer'):
                self.license_check_timer.stop()
                
            # Cleanup license manager
            if hasattr(self, 'license_manager'):
                self.license_manager.cleanup()
                
            # Save any current license state
            if hasattr(self, 'license_manager'):
                self.license_manager.save_license_key()
                
        except Exception as e:
            error_msg = f"Error during cleanup: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            else:
                print(error_msg)
                
        event.accept()
    # =================================================

    def create_rag_management_tab(self):
        """Create tab for managing the RAG knowledge base"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Custom Patterns Group
        patterns_group = QGroupBox("Custom Analysis Patterns")
        patterns_layout = QVBoxLayout()
        
        # Table showing current custom patterns
        self.patterns_table = QTableWidget()
        self.patterns_table.setColumnCount(4)
        self.patterns_table.setHorizontalHeaderLabels(
            ["Name", "Description", "Date Added", "Actions"]
        )
        self.refresh_patterns_table()
        patterns_layout.addWidget(self.patterns_table)
        
        # Button group for actions
        button_layout = QHBoxLayout()
        
        add_pattern_btn = QPushButton("Add Current Analysis")
        add_pattern_btn.clicked.connect(self.add_current_analysis)
        button_layout.addWidget(add_pattern_btn)
        
        #reset_base_btn = QPushButton("Reset to Default Knowledge Base")
        #reset_base_btn.clicked.connect(self.reset_knowledge_base)
        #Sbutton_layout.addWidget(reset_base_btn)
        
        patterns_layout.addLayout(button_layout)
        patterns_group.setLayout(patterns_layout)
        layout.addWidget(patterns_group)
        
        tab.setLayout(layout)
        return tab

    def toggle_rag(self, state):
        """Toggle RAG assistance and save the setting"""
        self.rag_assistant.enabled = bool(state)
        self.logger.info(f"RAG assistant {'enabled' if state else 'disabled'}")
        self.save_settings()  # Save settings when RAG state changes

    def update_knowledge_base_stats(self):
        """Update knowledge base statistics display"""
        try:
            stats = self.rag_assistant.get_knowledge_base_stats()
            stats_text = (
                f"Base Patterns: {stats['base_patterns']}\n"
                f"Custom Patterns: {stats['custom_patterns']}\n"
                f"Last Updated: {stats['last_updated'] or 'Never'}"
            )
            self.stats_label.setText(stats_text)
        except Exception as e:
            self.logger.error(f"Error updating stats: {str(e)}")
            self.stats_label.setText("Error loading stats")

    def refresh_patterns_table(self):
        """Refresh the patterns table display"""
        try:
            patterns = self.rag_assistant.get_custom_patterns()
            self.patterns_table.setRowCount(len(patterns))
            
            for i, pattern in enumerate(patterns):
                # Name
                self.patterns_table.setItem(
                    i, 0, QTableWidgetItem(pattern['name'])
                )
                # Description
                self.patterns_table.setItem(
                    i, 1, QTableWidgetItem(pattern['description'])
                )
                # Date
                self.patterns_table.setItem(
                    i, 2, QTableWidgetItem(str(pattern['created_at']))
                )
                
                # Delete button
                delete_btn = QPushButton("Delete")
                delete_btn.clicked.connect(
                    lambda checked, pid=pattern['id']: self.delete_pattern(pid)
                )
                self.patterns_table.setCellWidget(i, 3, delete_btn)
            
            self.patterns_table.resizeColumnsToContents()
            
        except Exception as e:
            self.logger.error(f"Error refreshing patterns table: {str(e)}")

    def add_current_analysis(self):
            """Add successful analysis to help improve AI suggestions"""
            if not hasattr(self, 'script_output') or not self.script_output.toPlainText():
                QMessageBox.warning(self, "Warning", "No analysis available to save")
                return
                    
            dialog = QDialog(self)
            dialog.setWindowTitle("Help Improve AI Suggestions")
            layout = QVBoxLayout()
            
            # Explanation text at the top
            explanation = QLabel(
                "Would you like to save this successful analysis to help the AI make better "
                "suggestions in the future? When you ask similar questions later, the AI will "
                "learn from this example to provide more accurate and relevant analyses."
            )
            explanation.setWordWrap(True)
            explanation.setStyleSheet("font-size: 11pt; margin-bottom: 15px;")
            layout.addWidget(explanation)
            
            # Form layout for inputs
            form_layout = QFormLayout()
            
            # Name input
            name_input = QLineEdit()
            name_input.setPlaceholderText("Example: Sales Growth Analysis")
            form_layout.addRow("Give this analysis a name:", name_input)
            
            # Brief note input
            description_input = QTextEdit()
            description_input.setPlaceholderText("Example: Calculates monthly sales growth and creates trend charts")
            description_input.setMaximumHeight(100)
            form_layout.addRow("Brief note about what it does:", description_input)
            
            layout.addLayout(form_layout)
            
            # Buttons
            button_box = QDialogButtonBox(
                QDialogButtonBox.Save | QDialogButtonBox.Cancel
            )
            button_box.button(QDialogButtonBox.Save).setText("Save to Improve AI")
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            dialog.setLayout(layout)
            
            if dialog.exec_() == QDialog.Accepted:
                try:
                    # Create a simple tag from the name
                    tags = [tag.strip() for tag in name_input.text().lower().split()]
                    
                    # Save analysis pattern using proper storage paths
                    pattern_data = {
                        'name': name_input.text(),
                        'description': description_input.toPlainText(),
                        'prompt': self.prompt_input.toPlainText(),
                        'code': self.script_output.toPlainText(),
                        'tags': tags,
                        'metadata': self.metadata
                    }

                    # Save to RAG assistant
                    self.rag_assistant.add_custom_pattern(**pattern_data)
                    
                    # Save pattern log
                    timestamp = datetime.now().isoformat()
                    pattern_log = {
                        'timestamp': timestamp,
                        'name': name_input.text(),
                        'description': description_input.toPlainText(),
                        'tags': tags
                    }
                    
                    # Use storage manager path for pattern logs
                    pattern_log_path = self.storage.data_dir / "pattern_logs" / f"pattern_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
                    pattern_log_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(pattern_log_path, 'a', encoding='utf-8') as f:
                        json.dump(pattern_log, f, cls=EnhancedJSONEncoder)
                        f.write('\n')
                    
                    self.refresh_patterns_table()
                    self.update_knowledge_base_stats()
                    QMessageBox.information(
                        self, 
                        "Success", 
                        "Thanks! The AI will learn from this analysis to provide better suggestions for similar questions in the future."
                    )
                except Exception as e:
                    self.logger.error(f"Error saving analysis pattern: {str(e)}")
                    QMessageBox.critical(self, "Error", f"Failed to save analysis: {str(e)}")

    def delete_pattern(self, pattern_id: int):
        """Delete a custom pattern"""
        reply = QMessageBox.question(
            self,
            "Delete Pattern",
            "Are you sure you want to delete this pattern?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Use the correct method name
                self.rag_assistant.delete_custom_pattern(pattern_id)
                self.refresh_patterns_table()
                self.update_knowledge_base_stats()
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to delete pattern: {str(e)}"
                )

#    def reset_knowledge_base(self):
#        """Reset to default knowledge base"""
#        reply = QMessageBox.question(
#            self,
#            "Reset Knowledge Base",
#            "Are you sure you want to reset to the default knowledge base? "
#            "This will remove all custom patterns.",
#            QMessageBox.Yes | QMessageBox.No
#        )
#        
#        if reply == QMessageBox.Yes:
#            try:
#                self.rag_assistant.reset_to_default()
#                self.refresh_patterns_table()
#                self.update_knowledge_base_stats()
#                QMessageBox.information(
#                    self, 
#                    "Success", 
#                    "Successfully reset to default knowledge base"
#                )
#            except Exception as e:
#                QMessageBox.critical(
#                    self,
#                    "Error",
#                    f"Failed to reset knowledge base: {str(e)}"
#                )

if __name__ == "__main__":
    # Create storage manager first
    storage_manager = StorageManager()
    
    # Setup logging with storage manager
    logger = setup_logging(storage_manager)

    app = QApplication(sys.argv)

    # Set up dark theme
    setup_dark_theme(app)
    
    # Modern high DPI handling for Qt6
    if hasattr(Qt.ApplicationAttribute, 'HighDpiScaleFactorRoundingPolicy'):
        app.setAttribute(Qt.ApplicationAttribute.HighDpiScaleFactorRoundingPolicy, 
                        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()

    # Only show window if initialization succeeded (including EULA acceptance)
    if hasattr(window, 'storage'):  # Check if initialization completed
        window.show()
        sys.exit(app.exec())
    else:
        sys.exit(1)



