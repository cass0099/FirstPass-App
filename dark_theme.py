from PySide6.QtGui import QPalette, QColor, QFont

def setup_dark_theme(app):
    """Set up dark theme for the entire application"""
    # Create the palette
    dark_palette = QPalette()
    
    # Set colors
    dark_palette.setColor(QPalette.Window, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.WindowText, QColor(240, 240, 240))
    dark_palette.setColor(QPalette.Base, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(240, 240, 240))
    dark_palette.setColor(QPalette.ToolTipText, QColor(240, 240, 240))
    dark_palette.setColor(QPalette.Text, QColor(240, 240, 240))
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, QColor(240, 240, 240))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
    dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))
    
    # Apply the palette to the application
    app.setPalette(dark_palette)
    
    # Set stylesheet for fine-tuned control
    app.setStyleSheet("""
        QMainWindow, QDialog {
            background-color: #2d2d2d;
        }
        QWidget {
            background-color: #2d2d2d;
            color: #f0f0f0;
        }
        QTabWidget::pane {
            border: 1px solid #3d3d3d;
        }
        QTabBar::tab {
            background-color: #353535;
            color: #f0f0f0;
            padding: 8px 15px;
            border: 1px solid #3d3d3d;
        }
        QTabBar::tab:selected {
            background-color: #424242;
            border-bottom: none;
        }
        QGroupBox {
            border: 1px solid #3d3d3d;
            margin-top: 0.5em;
            padding-top: 0.5em;
        }
        QGroupBox::title {
            color: #f0f0f0;
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
        }
        QTextEdit, QLineEdit {
            background-color: #1e1e1e;
            color: #f0f0f0;
            border: 1px solid #3d3d3d;
            padding: 2px;
        }
        QPushButton {
            background-color: #353535;
            color: #f0f0f0;
            border: 1px solid #3d3d3d;
            padding: 5px 15px;
            border-radius: 2px;
        }
        QPushButton:hover {
            background-color: #424242;
        }
        QPushButton:pressed {
            background-color: #2a2a2a;
        }
        QPushButton:disabled {
            background-color: #2d2d2d;
            color: #808080;
        }
        QComboBox {
            background-color: #353535;
            color: #f0f0f0;
            border: 1px solid #3d3d3d;
            padding: 5px;
            border-radius: 2px;
        }
        QComboBox:drop-down {
            border: none;
        }
        QComboBox::down-arrow {
            border: none;
        }
        QScrollBar:vertical {
            background-color: #2d2d2d;
            width: 12px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background-color: #454545;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #525252;
        }
        QScrollBar:horizontal {
            background-color: #2d2d2d;
            height: 12px;
            margin: 0;
        }
        QScrollBar::handle:horizontal {
            background-color: #454545;
            min-width: 20px;
        }
        QScrollBar::handle:horizontal:hover {
            background-color: #525252;
        }
        QTableWidget {
            background-color: #1e1e1e;
            color: #f0f0f0;
            gridline-color: #3d3d3d;
        }
        QTableWidget QHeaderView::section {
            background-color: #353535;
            color: #f0f0f0;
            padding: 5px;
            border: 1px solid #3d3d3d;
        }
        QTableWidget::item {
            padding: 5px;
        }
        QTableWidget::item:selected {
            background-color: #424242;
        }
        QCheckBox {
            color: #f0f0f0;
        }
        QCheckBox::indicator {
            width: 13px;
            height: 13px;
        }
        QMenuBar {
            background-color: #2d2d2d;
            color: #f0f0f0;
        }
        QMenuBar::item {
            background-color: transparent;
        }
        QMenuBar::item:selected {
            background-color: #353535;
        }
        QMenu {
            background-color: #2d2d2d;
            color: #f0f0f0;
        }
        QMenu::item:selected {
            background-color: #353535;
        }
    """)