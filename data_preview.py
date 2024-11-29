from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QSpinBox, QPushButton, QComboBox, QGroupBox, QScrollArea,
    QSizePolicy, QApplication
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut  # QShortcut moved here
import pandas as pd
import numpy as np

class DataPreviewTab(QWidget):
    """Tab for displaying CSV data preview and statistics"""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize instance variables
        self.df = None
        self.current_page = 0
        self.rows_per_page = 100
        self.summary_label = None
        self.prev_btn = None
        self.next_btn = None
        self.page_label = None
        self.rows_spinbox = None
        self.view_selector = None
        self.table = None
        
        # Initialize UI
        self.init_ui()
        
        # Set initial state
        self.update_summary()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()

        # Data summary section
        summary_group = QGroupBox("Data Summary")
        summary_layout = QVBoxLayout()
        self.summary_label = QLabel("No data loaded")
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        # Data preview controls
        controls_layout = QHBoxLayout()

        # Page controls
        page_layout = QHBoxLayout()
        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.clicked.connect(self.previous_page)
        self.prev_btn.setEnabled(False)
        
        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self.next_page)
        self.next_btn.setEnabled(False)
        
        self.page_label = QLabel("Page: 0/0")
        
        page_layout.addWidget(self.prev_btn)
        page_layout.addWidget(self.page_label)
        page_layout.addWidget(self.next_btn)
        controls_layout.addLayout(page_layout)

        # Rows per page control
        rows_layout = QHBoxLayout()
        rows_layout.addWidget(QLabel("Rows per page:"))
        self.rows_spinbox = QSpinBox()
        self.rows_spinbox.setRange(10, 1000)
        self.rows_spinbox.setValue(100)
        self.rows_spinbox.valueChanged.connect(self.update_rows_per_page)
        rows_layout.addWidget(self.rows_spinbox)
        controls_layout.addLayout(rows_layout)

        # View selector
        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("View:"))
        self.view_selector = QComboBox()
        self.view_selector.addItems(["Data Preview", "Data Types", "Basic Statistics"])
        self.view_selector.currentTextChanged.connect(self.update_view)
        view_layout.addWidget(self.view_selector)
        controls_layout.addLayout(view_layout)

        # Add copy button
        copy_btn = QPushButton("Copy Selected")
        copy_btn.clicked.connect(self.copy_selection)
        copy_btn.setToolTip("Copy selected cells (Ctrl+C)")
        controls_layout.addWidget(copy_btn)

        layout.addLayout(controls_layout)

        # Data table with enhanced selection
        self.table = QTableWidget()
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)  # Allow multiple selection
        self.table.setSelectionBehavior(QTableWidget.SelectItems)    # Can select individual cells

        # Enable copying with Ctrl+C / Cmd+C
        copy_shortcut = QShortcut(QKeySequence.Copy, self.table)
        copy_shortcut.activated.connect(self.copy_selection)

        layout.addWidget(self.table)
        self.setLayout(layout)

    def update_summary(self):
        """Update the data summary information"""
        if self.df is not None:
            summary = (
                f"Rows: {len(self.df):,}\n"
                f"Columns: {len(self.df.columns)}\n"
                f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n"
                f"Missing Values: {self.df.isna().sum().sum():,}"
            )
            self.summary_label.setText(summary)
        else:
            self.summary_label.setText("No data loaded")

    def load_data(self, df: pd.DataFrame):
        """Load new DataFrame and update display"""
        self.df = df
        self.current_page = 0
        self.update_summary()
        self.update_table()

    def update_rows_per_page(self):
        """Update the number of rows displayed per page"""
        self.rows_per_page = self.rows_spinbox.value()
        self.current_page = 0
        self.update_table()

    def previous_page(self):
        """Go to previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_table()

    def next_page(self):
        """Go to next page"""
        if self.df is not None:
            total_pages = len(self.df) // self.rows_per_page
            if self.current_page < total_pages:
                self.current_page += 1
                self.update_table()

    def _format_value(self, value):
        """Format value for display in table"""
        if pd.isna(value):
            return 'NA'
        elif isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        elif isinstance(value, (int, np.integer)):
            return f"{value:,}"
        return str(value)

    def update_table(self):
        """Update the table display based on current view and page"""
        if self.df is None:
            return

        current_view = self.view_selector.currentText()
        
        if current_view == "Data Preview":
            self._show_data_preview()
        elif current_view == "Data Types":
            self._show_data_types()
        elif current_view == "Basic Statistics":
            self._show_statistics()

    def _show_data_preview(self):
        """Show data preview with pagination"""
        start_idx = self.current_page * self.rows_per_page
        end_idx = start_idx + self.rows_per_page
        page_data = self.df.iloc[start_idx:end_idx]

        # Update table
        self.table.setRowCount(len(page_data))
        self.table.setColumnCount(len(self.df.columns))
        self.table.setHorizontalHeaderLabels(self.df.columns)

        # Populate data
        for i, (_, row) in enumerate(page_data.iterrows()):
            for j, value in enumerate(row):
                item = QTableWidgetItem(self._format_value(value))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, j, item)

        # Update navigation
        total_pages = (len(self.df) - 1) // self.rows_per_page
        self.page_label.setText(f"Page: {self.current_page + 1}/{total_pages + 1}")
        self.prev_btn.setEnabled(self.current_page > 0)
        self.next_btn.setEnabled(self.current_page < total_pages)

        # Adjust column widths
        self.table.resizeColumnsToContents()

    def _show_data_types(self):
        """Show column data types"""
        dtypes = self.df.dtypes.reset_index()
        dtypes.columns = ['Column', 'Data Type']

        # Update table
        self.table.setRowCount(len(dtypes))
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Column', 'Data Type'])

        # Populate data
        for i, (_, row) in enumerate(dtypes.iterrows()):
            self.table.setItem(i, 0, QTableWidgetItem(str(row['Column'])))
            self.table.setItem(i, 1, QTableWidgetItem(str(row['Data Type'])))

        # Disable navigation buttons
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.page_label.setText("Data Types View")

        # Adjust column widths
        self.table.resizeColumnsToContents()

    def _show_statistics(self):
        """Show basic statistics for numeric columns"""
        # Calculate statistics for numeric columns
        stats = self.df.describe().round(6)
        stats = stats.reset_index()

        # Update table
        self.table.setRowCount(len(stats))
        self.table.setColumnCount(len(stats.columns))
        self.table.setHorizontalHeaderLabels(stats.columns)

        # Populate data
        for i, (_, row) in enumerate(stats.iterrows()):
            for j, value in enumerate(row):
                item = QTableWidgetItem(self._format_value(value))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, j, item)

        # Disable navigation buttons
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.page_label.setText("Statistics View")

        # Adjust column widths
        self.table.resizeColumnsToContents()

    def update_view(self):
        """Update display when view selector changes"""
        self.current_page = 0
        self.update_table()

    def copy_selection(self):
        """Copy selected cells to clipboard with improved handling"""
        try:
            selected_items = self.table.selectedItems()
            if not selected_items:
                return

            # Get the bounds of selection
            rows = []
            cols = []
            for item in selected_items:
                rows.append(item.row())
                cols.append(item.column())
            
            min_row, max_row = min(rows), max(rows)
            min_col, max_col = min(cols), max(cols)

            # Create a matrix to hold the data
            matrix = [['' for _ in range(min_col, max_col + 1)] 
                      for _ in range(min_row, max_row + 1)]

            # Fill in selected cells
            for item in selected_items:
                r, c = item.row() - min_row, item.column() - min_col
                matrix[r][c] = item.text()

            # Convert to text
            text = '\n'.join(['\t'.join(row) for row in matrix])

            # Copy to clipboard
            clipboard = QApplication.clipboard()
            clipboard.setText(text)

            # Show feedback in status bar if available
            main_window = self.window()
            if main_window and hasattr(main_window, 'statusBar'):
                main_window.statusBar().showMessage("Selection copied to clipboard", 2000)

        except Exception as e:
            # Show error in status bar if available
            main_window = self.window()
            if main_window and hasattr(main_window, 'statusBar'):
                main_window.statusBar().showMessage(f"Error copying selection: {str(e)}", 3000)