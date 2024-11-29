from storage import StorageManager  
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QTableWidget, QTableWidgetItem,
    QPushButton, QDialog, QLineEdit, QTextEdit, QComboBox, QFormLayout,
    QDialogButtonBox, QLabel, QCompleter, QMessageBox, QFileDialog,
    QGroupBox, QRadioButton, QToolTip, QHeaderView
)
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QCursor
import json
import pandas as pd
import logging
from rag_integration import RAGAssistant, KnowledgeType


# Add these new classes for threading
class AddColumnWorker(QThread):
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, rag_assistant, column_data):
        super().__init__()
        self.rag_assistant = rag_assistant
        self.column_data = column_data
    
    def run(self):
        try:
            knowledge_id = self.rag_assistant.add_column_definition(
                column_pattern=self.column_data['pattern'],
                description=self.column_data['description'],
                data_type=self.column_data['data_type'],
                examples=self.column_data['examples'],
                notes=self.column_data['notes']
            )
            if knowledge_id:
                self.finished.emit()
            else:
                self.error.emit("Failed to add column definition")
        except Exception as e:
            self.error.emit(str(e))

class AddBusinessKnowledgeWorker(QThread):
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, rag_assistant, knowledge_data):
        super().__init__()
        self.rag_assistant = rag_assistant
        self.knowledge_data = knowledge_data
    
    def run(self):
        try:
            knowledge_id = self.rag_assistant.add_business_knowledge(
                text=self.knowledge_data['text'],
                summary=self.knowledge_data.get('summary'),
                system_tags=self.knowledge_data['system_tags'],
                custom_tags=self.knowledge_data.get('custom_tags'),
                related_columns=self.knowledge_data.get('related_columns'),
                priority=self.knowledge_data.get('priority', 1)
            )
            if knowledge_id:
                self.finished.emit()
            else:
                self.error.emit("Failed to add business knowledge")
        except Exception as e:
            self.error.emit(str(e))

class RefreshTableWorker(QThread):
    finished = Signal(list)
    error = Signal(str)
    
    def __init__(self, rag_assistant):
        super().__init__()
        self.rag_assistant = rag_assistant
    
    def run(self):
        try:
            definitions = self.rag_assistant.get_column_definitions()
            self.finished.emit(definitions)
        except Exception as e:
            self.error.emit(str(e))

class DataDictionaryTab(QWidget):
    def __init__(self, rag_assistant: RAGAssistant, storage_manager: StorageManager, parent=None):
        super().__init__(parent)
        self.storage_manager = storage_manager
        self.rag_assistant = rag_assistant
        self.logger = logging.getLogger(__name__)
        self.init_ui()
        # Initialize worker threads as None
        self.add_worker = None
        self.refresh_worker = None
        # Initial refresh
        self.refresh_table()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Column dictionary table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ['Column Name', 'Description', 'Data Type', 'Actions']
        )
        
        # Add/Import buttons
        btn_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add Entry")
        add_btn.clicked.connect(self.show_add_dialog)
        
        import_btn = QPushButton("Import from CSV")
        import_btn.clicked.connect(self.import_dictionary)
        
        template_btn = QPushButton("Download Template")
        template_btn.clicked.connect(self.download_template)
        
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(import_btn)
        btn_layout.addWidget(template_btn)
        
        layout.addLayout(btn_layout)
        layout.addWidget(self.table)
        self.setLayout(layout)


    def show_add_dialog(self):
        """Show dialog to add new column definition with enhanced tooltips and pattern selection"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Column Definition")
        layout = QFormLayout()

        # Add help text at the top
        help_text = QLabel(
            "Define how columns in your data should be interpreted. "
            "This information will help generate more accurate analyses."
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color: #666666; font-style: italic; padding-bottom: 10px;")
        layout.addRow(help_text)

        # Add radio button group for pattern type
        pattern_type_group = QGroupBox("Column Matching Type")
        pattern_layout = QVBoxLayout()

        exact_radio = QRadioButton("Exact Column Name")
        exact_radio.setToolTip("Match a single, specific column name exactly as it appears in your data")
        pattern_radio = QRadioButton("Pattern Match (Multiple Columns)")
        pattern_radio.setToolTip(
            "Match multiple similar columns using wildcards (*)\n"
            "Example: 'revenue_*' matches 'revenue_2023', 'revenue_usd', etc."
        )
        exact_radio.setChecked(True)

        pattern_layout.addWidget(exact_radio)
        pattern_layout.addWidget(pattern_radio)
        pattern_type_group.setLayout(pattern_layout)
        layout.addRow(pattern_type_group)

        # Column name/pattern input
        pattern_label = QLabel("Column Name:")  # Will update dynamically
        pattern_label.setToolTip(
            "Enter either:\n"
            "- A specific column name (e.g., 'revenue')\n"
            "- A pattern using * as wildcard (e.g., 'revenue_*' matches 'revenue_2023', 'revenue_usd')"
        )
        pattern_input = QLineEdit()
        pattern_input.setPlaceholderText("e.g., revenue or revenue_*")
        layout.addRow(pattern_label, pattern_input)

        # Add example section that updates based on radio selection
        example_label = QLabel()
        example_label.setStyleSheet("color: #666666; font-style: italic;")
        layout.addRow(QLabel("Examples:"), example_label)

        # Description input
        description_label = QLabel("What does this column represent?")
        description_label.setToolTip(
            "Provide a clear explanation of what this column represents.\n"
            "Include any important details about the data's meaning and usage."
        )
        description_input = QTextEdit()
        description_input.setMaximumHeight(100)
        description_input.setPlaceholderText(
            "Examples:\n"
            "• Monthly revenue in USD before taxes and refunds\n"
            "• User subscription status (active/inactive/pending)\n"
            "• Date when the user first created their account"
        )
        layout.addRow(description_label, description_input)

        # Data type selection
        type_label = QLabel("What kind of data is stored in this column?")
        type_label.setToolTip(
            "Select the expected data type for this column.\n"
            "This helps ensure proper data handling in analyses."
        )
        data_type_input = QComboBox()
        data_type_input.addItems([
            "string - Text values",
            "integer - Whole numbers",
            "float - Decimal numbers",
            "boolean - True/False values", 
            "datetime - Dates and times",
            "category - Limited set of values",
            "other - Special data types"
        ])
        layout.addRow(type_label, data_type_input)

        # Example values
        examples_label = QLabel("What are some typical values found in this column?")
        examples_label.setToolTip(
            "List typical values found in this column.\n"
            "This helps clarify the expected format and range of values."
        )
        examples_input = QLineEdit()
        examples_input.setPlaceholderText("e.g., active, inactive, pending (separate with commas)")
        layout.addRow(examples_label, examples_input)

        # Additional notes
        notes_label = QLabel("Additional Usage Notes (Optional)")
        notes_label.setToolTip(
            "Add any additional context about how this column should be used.\n"
            "Include special considerations, common issues, or related columns."
        )
        notes_input = QTextEdit()
        notes_input.setMaximumHeight(100)
        notes_input.setPlaceholderText(
            "Examples:\n"
            "• Values are case-sensitive\n"
            "• May contain NULL for new users\n"
            "• Should be used together with the 'account_type' column"
        )
        layout.addRow(notes_label, notes_input)

        def update_pattern_ui():
            if exact_radio.isChecked():
                pattern_label.setText("Column Name:")
                pattern_input.setPlaceholderText("e.g., revenue")
                example_label.setText(
                    "Examples:\n"
                    "• revenue\n"
                    "• user_id\n"
                    "• account_status"
                )
            else:
                pattern_label.setText("Column Pattern:")
                pattern_input.setPlaceholderText("e.g., revenue_* or user_*_status")
                example_label.setText(
                    "Examples:\n"
                    "• revenue_* matches: revenue_2023, revenue_usd, revenue_eur\n"
                    "• user_*_status matches: user_account_status, user_email_status\n"
                    "• *_amount matches: refund_amount, payment_amount"
                )

        exact_radio.toggled.connect(update_pattern_ui)
        pattern_radio.toggled.connect(update_pattern_ui)
        update_pattern_ui()  # Set initial state

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addRow(button_box)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            try:
                # Get values
                column_data = {
                    'pattern': pattern_input.text().strip(),
                    'description': description_input.toPlainText().strip(),
                    'data_type': data_type_input.currentText().split(' - ')[0],  # Get just the type, not description
                    'examples': [x.strip() for x in examples_input.text().split(",") if x.strip()],
                    'notes': notes_input.toPlainText().strip()
                }

                if not column_data['pattern'] or not column_data['description']:
                    raise ValueError("Column name/pattern and description are required")

                # Create and start worker thread
                self.add_worker = AddColumnWorker(self.rag_assistant, column_data)
                self.add_worker.finished.connect(self.on_add_complete)
                self.add_worker.error.connect(self.on_add_error)
                self.add_worker.start()

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to add column definition: {str(e)}"
                )


    def on_add_complete(self):
        """Handle successful addition"""
        self.refresh_table()
        QMessageBox.information(
            self,
            "Success",
            "Column definition added successfully"
        )

    def on_add_error(self, error_msg):
        """Handle addition error"""
        QMessageBox.critical(
            self,
            "Error",
            f"Failed to add column definition: {error_msg}"
        )


    def import_dictionary(self):
        """Import column definitions from CSV with flexible column naming"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Import Column Definitions",
                "",
                "CSV Files (*.csv)"
            )

            if file_path:
                # Read CSV
                df = pd.read_csv(file_path)
                
                # Map various possible column names to expected names
                column_mappings = {
                    'column_pattern': ['column_pattern', 'Column Name', 'column name', 'ColumnName', 'column'],
                    'description': ['description', 'Description', 'desc', 'Definition'],
                    'data_type': ['data_type', 'Data Type', 'type', 'Type'],
                    'examples': ['examples', 'Examples', 'sample values', 'Sample Values'],
                    'notes': ['notes', 'Notes', 'additional notes', 'Additional Notes']
                }
                
                # Find actual column names in the CSV
                df_columns = df.columns
                column_map = {}
                
                for expected_col, possible_names in column_mappings.items():
                    found = False
                    for name in possible_names:
                        if name in df_columns:
                            column_map[expected_col] = name
                            found = True
                            break
                    if not found and expected_col in ['column_pattern', 'description']:
                        raise ValueError(f"Could not find a column matching {expected_col}")
                
                # Rename columns to expected names
                df = df.rename(columns={v: k for k, v in column_map.items()})

                # Process each row
                success_count = 0
                error_count = 0
                for idx, row in df.iterrows():
                    try:
                        # Parse examples if present
                        examples = None
                        if 'examples' in df.columns and pd.notna(row['examples']):
                            examples = [x.strip() for x in str(row['examples']).split(',')]

                        # Add to knowledge base
                        self.rag_assistant.add_column_definition(
                            column_pattern=str(row['column_pattern']).strip(),
                            description=str(row['description']).strip(),
                            data_type=str(row['data_type']) if 'data_type' in df.columns and pd.notna(row['data_type']) else None,
                            examples=examples,
                            notes=str(row['notes']) if 'notes' in df.columns and pd.notna(row['notes']) else None
                        )
                        success_count += 1
                    except Exception as e:
                        error_count += 1
                        self.logger.error(f"Error processing row {idx}: {str(e)}")

                # Refresh display
                self.refresh_table()

                # Show results
                QMessageBox.information(
                    self,
                    "Import Complete",
                    f"Successfully imported {success_count} definitions.\n"
                    f"Failed to import {error_count} definitions."
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to import definitions: {str(e)}"
            )

    def refresh_table(self):
        """Refresh the table display"""
        try:
            # Create and start worker thread
            self.refresh_worker = RefreshTableWorker(self.rag_assistant)
            self.refresh_worker.finished.connect(self.update_table_data)
            self.refresh_worker.error.connect(self.on_refresh_error)
            self.refresh_worker.start()
        except Exception as e:
            self.logger.error(f"Error starting refresh: {str(e)}")

    def on_refresh_error(self, error_msg):
        """Handle refresh error"""
        QMessageBox.critical(
            self,
            "Error",
            f"Failed to refresh table: {error_msg}"
        )

    def delete_definition(self, definition_id: int):
        """Delete a column definition with threading"""
        reply = QMessageBox.question(
            self,
            "Delete Definition",
            "Are you sure you want to delete this column definition?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Create and start worker thread for deletion
                self.delete_worker = QThread()
                self.delete_worker.run = lambda: self.rag_assistant.delete_knowledge(
                    definition_id,
                    KnowledgeType.COLUMN  # Use KnowledgeType directly
                )
                self.delete_worker.finished.connect(self.refresh_table)
                self.delete_worker.start()
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to delete definition: {str(e)}"
                )

    def download_template(self):
        """Download CSV template for column dictionary import with improved examples"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Template CSV",
                "column_dictionary_template.csv",
                "CSV Files (*.csv)"
            )
            
            if file_path:
                if not file_path.endswith('.csv'):
                    file_path += '.csv'
                
                # Create template dataframe with business-focused examples
                template_df = pd.DataFrame({
                    'Column Name': [
                        'customer_id',
                        'revenue_*',
                        'booking_date',
                        'customer_segment',
                        'price_*',
                        'satisfaction_score'
                    ],
                    'Description': [
                        'Unique identifier for each customer in the system',
                        'Any revenue-related columns (monthly, annual, recurring)',
                        'Date when the customer made their booking',
                        'Customer segmentation category (e.g., Business, Leisure)',
                        'Any price-related columns (base price, final price, discount price)',
                        'Customer satisfaction rating on a scale of 1-5'
                    ],
                    'Data Type': [
                        'string',
                        'float',
                        'datetime',
                        'category',
                        'float',
                        'integer'
                    ],
                    'Examples': [
                        'CUST123, CUST456',
                        '1000.50, 2500.75, 750.25',
                        '2024-01-01, 2024-12-31',
                        'Business, Leisure',
                        '99.99, 149.99, 199.99',
                        '1, 2, 3, 4, 5'
                    ],
                    'Notes': [
                        'Used as primary key in customer database',
                        'All revenue values in USD, pre-tax',
                        'ISO format dates only (YYYY-MM-DD)',
                        'Used for segment-specific analysis and reporting',
                        'All prices in USD, includes tax',
                        'Higher score indicates greater satisfaction'
                    ]
                })
                
                # Save template
                template_df.to_csv(file_path, index=False)
                
                # Show helpful message
                QMessageBox.information(
                    self,
                    "Success",
                    f"Template saved to:\n{file_path}\n\n"
                    f"Template includes examples showing:\n"
                    f"- Exact column matching (e.g., 'customer_id')\n"
                    f"- Pattern matching (e.g., 'revenue_*' matches 'revenue_monthly', 'revenue_annual')\n"
                    f"- Common data types and formats\n"
                    f"- Business-relevant examples and descriptions\n"
                    f"- Helpful usage notes and context\n\n"
                    f"Instructions:\n"
                    f"1. Fill in your column definitions\n"
                    f"2. Use patterns with '*' to match multiple similar columns\n"
                    f"3. Keep descriptions clear and business-focused\n"
                    f"4. Include typical example values\n"
                    f"5. Add helpful usage notes"
                )
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save template: {str(e)}"
            )

    def update_table_data(self, definitions):
        """Update table with new data"""
        try:
            self.table.setRowCount(len(definitions))
            
            for i, definition in enumerate(definitions):
                # Column pattern
                self.table.setItem(
                    i, 0, 
                    QTableWidgetItem(definition['column_pattern'])
                )
                
                # Description
                self.table.setItem(
                    i, 1, 
                    QTableWidgetItem(definition['description'])
                )
                
                # Data type
                self.table.setItem(
                    i, 2, 
                    QTableWidgetItem(definition.get('data_type', ''))
                )
                
                # Actions (Delete button)
                delete_btn = QPushButton("Delete")
                delete_btn.clicked.connect(
                    lambda checked, d=definition: self.delete_definition(d['id'])
                )
                self.table.setCellWidget(i, 3, delete_btn)
            
            self.table.resizeColumnsToContents()
            
        except Exception as e:
            self.logger.error(f"Error updating table: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to update table: {str(e)}"
            )

class BusinessKnowledgeTab(QWidget):
    def __init__(self, rag_assistant: RAGAssistant, storage_manager: StorageManager, parent=None):
        super().__init__(parent)
        self.storage_manager = storage_manager
        self.rag_assistant = rag_assistant
        self.logger = logging.getLogger(__name__)
        self.init_ui()
        # Initialize worker threads as None
        self.add_worker = None
        self.refresh_worker = None
        self.delete_worker = None
        # Initial refresh
        self.refresh_table()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Knowledge entry table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            'Description', 'System Tags', 'Custom Tags', 'Priority', 'Actions'
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        
        # Add button
        add_btn = QPushButton("Add Knowledge")
        add_btn.clicked.connect(self.show_add_dialog)
        
        layout.addWidget(add_btn)
        layout.addWidget(self.table)
        self.setLayout(layout)

    def refresh_table(self):
        """Refresh the business knowledge table with threading"""
        try:
            class RefreshWorker(QThread):
                finished = Signal(list)
                error = Signal(str)
                
                def __init__(self, rag_assistant):
                    super().__init__()
                    self.rag_assistant = rag_assistant
                
                def run(self):
                    try:
                        entries = self.rag_assistant.get_business_knowledge()
                        self.finished.emit(entries)
                    except Exception as e:
                        self.error.emit(str(e))
            
            self.refresh_worker = RefreshWorker(self.rag_assistant)
            self.refresh_worker.finished.connect(self.update_table_data)
            self.refresh_worker.error.connect(self.on_refresh_error)
            self.refresh_worker.start()
            
        except Exception as e:
            self.logger.error(f"Error starting refresh: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to start refresh: {str(e)}"
            )

    def update_table_data(self, entries):
        """Update table with new data"""
        try:
            print("\nDEBUG: Starting table update")
            print(f"DEBUG: Received entries: {entries}")

            if entries is None:
                print("DEBUG: Entries is None")
                self.table.setRowCount(0)
                return

            # Filter out empty/invalid entries
            valid_entries = [
                e for e in entries 
                if isinstance(e, dict) and 
                e.get('id') is not None and 
                e.get('text') and 
                len(str(e['text']).strip()) >= 3
            ]
            
            print(f"DEBUG: Valid entries: {valid_entries}")
            
            self.table.setRowCount(len(valid_entries))
            
            for i, entry in enumerate(valid_entries):
                print(f"\nDEBUG: Processing entry {i}: {entry}")
                
                # Description column - get text or summary, ensuring we have a string
                text = entry.get('summary') or entry.get('text', '')
                if text and len(str(text)) > 100:
                    text = str(text)[:97] + "..."
                self.table.setItem(i, 0, QTableWidgetItem(str(text)))
                print(f"DEBUG: Set text: {text}")
                
                # System Tags
                try:
                    if entry.get('system_tags'):
                        system_tags = json.loads(entry['system_tags'])
                        if not isinstance(system_tags, list):
                            system_tags = []
                    else:
                        system_tags = []
                    print(f"DEBUG: Parsed system tags: {system_tags}")
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"DEBUG: Error parsing system tags: {e}")
                    system_tags = []
                self.table.setItem(i, 1, QTableWidgetItem(', '.join(map(str, system_tags))))
                
                # Custom Tags
                try:
                    if entry.get('custom_tags'):
                        custom_tags = json.loads(entry['custom_tags'])
                        if not isinstance(custom_tags, list):
                            custom_tags = []
                    else:
                        custom_tags = []
                    print(f"DEBUG: Parsed custom tags: {custom_tags}")
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"DEBUG: Error parsing custom tags: {e}")
                    custom_tags = []
                self.table.setItem(i, 2, QTableWidgetItem(', '.join(map(str, custom_tags))))
                
                # Priority
                priority = entry.get('priority', 1)
                self.table.setItem(i, 3, QTableWidgetItem(str(priority)))
                print(f"DEBUG: Set priority: {priority}")
                
                # Delete button
                delete_btn = QPushButton("Delete")
                # Ensure we capture entry_id in the lambda's scope
                entry_id = entry['id']
                delete_btn.clicked.connect(lambda checked, eid=entry_id: self.delete_entry(eid))
                self.table.setCellWidget(i, 4, delete_btn)
                print(f"DEBUG: Added delete button for id: {entry_id}")
            
            self.table.resizeColumnsToContents()
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            print("DEBUG: Finished updating table")
            
        except Exception as e:
            self.logger.error(f"Error updating table: {str(e)}")
            print(f"DEBUG: Error in update_table_data: {str(e)}")
            print(f"DEBUG: Error type: {type(e)}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to update table: {str(e)}")

    def on_refresh_error(self, error_msg):
        """Handle refresh error"""
        QMessageBox.critical(
            self,
            "Error",
            f"Failed to refresh table: {error_msg}"
        )

    def delete_entry(self, entry_id: int):
        """Delete a business knowledge entry with proper thread handling"""
        reply = QMessageBox.question(
            self,
            "Delete Entry",
            "Are you sure you want to delete this knowledge entry?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Create worker thread
                class DeleteWorker(QThread):
                    finished = Signal()
                    error = Signal(str)
                    
                    def __init__(self, rag_assistant, entry_id):
                        super().__init__()
                        self.rag_assistant = rag_assistant
                        self.entry_id = entry_id
                        self._is_running = False
                    
                    def run(self):
                        try:
                            self._is_running = True
                            self.rag_assistant.delete_knowledge(
                                self.entry_id,
                                KnowledgeType.BUSINESS
                            )
                            self.finished.emit()
                        except Exception as e:
                            self.error.emit(str(e))
                        finally:
                            self._is_running = False
                            
                    def stop(self):
                        self._is_running = False
                        self.wait()
                
                # Store worker as instance variable
                self.delete_worker = DeleteWorker(self.rag_assistant, entry_id)
                
                # Connect signals
                self.delete_worker.finished.connect(self._on_delete_complete)
                self.delete_worker.error.connect(self._on_delete_error)
                
                # Start worker
                self.delete_worker.start()
                
            except Exception as e:
                self.logger.error(f"Failed to start delete worker: {str(e)}")
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to delete entry: {str(e)}"
                )

        def _on_delete_complete(self):
            """Handle successful deletion"""
            self.refresh_table()
            try:
                if hasattr(self, 'delete_worker'):
                    self.delete_worker.stop()
                    self.delete_worker.deleteLater()
                    self.delete_worker = None
            except Exception as e:
                self.logger.error(f"Error cleaning up delete worker: {str(e)}")

        def _on_delete_error(self, error_msg: str):
            """Handle deletion error"""
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to delete entry: {error_msg}"
            )
            try:
                if hasattr(self, 'delete_worker'):
                    self.delete_worker.stop()
                    self.delete_worker.deleteLater()
                    self.delete_worker = None
            except Exception as e:
                self.logger.error(f"Error cleaning up delete worker: {str(e)}")

    def show_add_dialog(self):
        """Show improved dialog to add new business knowledge"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Business Knowledge")
        layout = QFormLayout()
        
        # Add help text at the top
        help_text = QLabel(
            "Add business rules and knowledge that should be considered during data analysis. "
            "This could include calculation rules, business definitions, or important analysis considerations."
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("color: #666666; font-style: italic; margin-bottom: 15px;")
        layout.addRow(help_text)
        
        # Main knowledge input
        knowledge_label = QLabel("Business Rule or Knowledge: *")  # Added asterisk for required
        knowledge_label.setToolTip(
            "Enter the complete business rule or knowledge.\n\n"
            "Good examples:\n"
            "• For revenue calculations: Exclude refunds and include recurring fees\n"
            "• For customer segmentation: Enterprise customers must have >500 seats\n"
            "• For analysis requirements: Always split results by region and customer tier"
        )
        text_input = QTextEdit()
        text_input.setMinimumHeight(100)
        text_input.setPlaceholderText(
            "Enter the complete business rule or knowledge here.\n\n"
            "Examples:\n"
            "- Monthly revenue must exclude refunds but include recurring subscription fees\n"
            "- Enterprise customers are defined as organizations with more than 500 seats\n"
            "- All pricing analysis must be segmented by region and customer tier"
        )
        layout.addRow(knowledge_label, text_input)
        
        # Brief description (renamed from Quick Reference Summary)
        brief_desc_label = QLabel("Brief Description (Optional):")
        brief_desc_label.setToolTip(
            "A shorter version of your business rule.\n"
            "This will be used in table previews and when the full text is too long."
        )
        brief_desc_input = QLineEdit()
        brief_desc_input.setPlaceholderText("A short description of your rule (e.g., 'Revenue calculation excluding refunds')")
        layout.addRow(brief_desc_label, brief_desc_input)
        
        # Business Domain
        domain_label = QLabel("Business Area: *")  # Added asterisk for required
        domain_label.setToolTip(
            "Select the main business area this knowledge applies to.\n"
            "This helps organize and find relevant rules during analysis."
        )
        domain_input = QComboBox()
        domain_input.addItems([
            "finance - Financial rules and metrics",
            "sales - Sales processes and targets",
            "marketing - Marketing campaigns and metrics",
            "product - Product features and usage",
            "support - Customer support guidelines",
            "operations - Business operations",
            "legal - Compliance and regulations",
            "hr - Human resources policies"
        ])
        layout.addRow(domain_label, domain_input)
        
        # Custom tags (renamed from Categories)
        tags_label = QLabel("Additional Tags (Optional):")
        tags_label.setToolTip(
            "Add your own tags to help find this knowledge later.\n"
            "Separate multiple tags with commas."
        )
        tags_input = QLineEdit()
        tags_input.setPlaceholderText("pricing, enterprise, north-america")
        layout.addRow(tags_label, tags_input)
        
        # Related columns
        columns_label = QLabel("Related Data Columns (Optional):")
        columns_label.setToolTip(
            "List the data columns this knowledge applies to.\n"
            "This helps the system know when to apply this rule.\n"
            "Separate multiple columns with commas."
        )
        columns_input = QLineEdit()
        columns_input.setPlaceholderText("revenue, refunds, customer_tier")
        layout.addRow(columns_label, columns_input)
        
        # Priority with better descriptions
        priority_label = QLabel("Rule Priority: *")  # Added asterisk for required
        priority_label.setToolTip(
            "How important is this rule when analyzing data?\n\n"
            "Normal: General guidelines that are good to know\n"
            "Important: Should be considered in most analyses\n"
            "Critical: Must always be followed - core business rules"
        )
        priority_input = QComboBox()
        priority_input.addItems([
            "1 - Normal (General guidelines)",
            "2 - Important (Key business rules)",
            "3 - Critical (Must always follow)"
        ])
        layout.addRow(priority_label, priority_input)
        
        # Required fields note
        required_note = QLabel("* Required fields")
        required_note.setStyleSheet("color: #666666; font-style: italic;")
        layout.addRow(required_note)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addRow(button_box)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            try:
                # Get values
                knowledge_data = {
                    'text': text_input.toPlainText().strip(),
                    'summary': brief_desc_input.text().strip() or None,
                    'system_tags': [domain_input.currentText().split(' - ')[0]],  # Get just the tag name
                    'custom_tags': [t.strip() for t in tags_input.text().split(",") if t.strip()],
                    'related_columns': [c.strip() for c in columns_input.text().split(",") if c.strip()],
                    'priority': int(priority_input.currentText()[0])
                }
                
                if not knowledge_data['text']:
                    raise ValueError("Business rule or knowledge is required")
                
                # Create and start worker thread
                self.add_worker = AddBusinessKnowledgeWorker(self.rag_assistant, knowledge_data)
                self.add_worker.finished.connect(self.on_add_complete)
                self.add_worker.error.connect(self.on_add_error)
                self.add_worker.start()
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to add business knowledge: {str(e)}"
                )

    def on_add_complete(self):
        """Handle successful addition"""
        self.refresh_table()
        QMessageBox.information(
            self,
            "Success",
            "Business knowledge added successfully"
        )

    def on_add_error(self, error_msg):
        """Handle addition error"""
        QMessageBox.critical(
            self,
            "Error",
            f"Failed to add business knowledge: {error_msg}"
        )