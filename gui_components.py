# gui_components.py
from PySide6.QtGui import QFont, QPalette, QColor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt



class OutputTab(QWidget):
    """Tab for displaying script output and visualizations"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Add clear all button
        clear_btn = QPushButton("Clear All Outputs")
        clear_btn.clicked.connect(self.clear_all)
        layout.addWidget(clear_btn)
        
        # Add output text area
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        self.text_output.setPlaceholderText("Script output will appear here...")
        layout.addWidget(self.text_output)
        
        self.setLayout(layout)

    def clear_all(self):
        """Clear all outputs"""
        self.text_output.clear()
        plt.close('all')