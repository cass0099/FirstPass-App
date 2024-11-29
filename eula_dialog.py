from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QTextEdit, QCheckBox,
    QPushButton, QLabel, QScrollArea, QWidget,
    QDialogButtonBox
)
from PySide6.QtCore import Qt
import json
import logging
from pathlib import Path
from datetime import datetime

class EULADialog(QDialog):
    def __init__(self, storage, parent=None):
        super().__init__(parent)
        self.storage = storage
        self.logger = logging.getLogger(__name__)
        self.accepted = False
        print("EULA Dialog initialized")
        
        # Define EULA configuration file path
        self.eula_config_file = self.storage.config_dir / "eula_config.json"
        self.eula_text_file = self.storage.data_dir / "eula_text.txt"
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("FirstPass License Agreement and Terms of Service")
        self.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(self)
        
        # Header label
        header = QLabel("Please read and accept the following License Agreement and Terms of Service:")
        header.setWordWrap(True)
        layout.addWidget(header)
        
        # Scrollable EULA text
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        eula_text = QTextEdit()
        eula_text.setReadOnly(True)
        eula_text.setText(self.get_eula_text())
        
        eula_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #f0f0f0;
                border: 1px solid #3c3c3c;
                padding: 10px;
            }
        """)
        
        scroll_layout.addWidget(eula_text)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        # Acceptance checkbox
        self.accept_checkbox = QCheckBox("I have read and agree to the License Agreement and Terms of Service")
        self.accept_checkbox.setStyleSheet("color: #f0f0f0;")
        layout.addWidget(self.accept_checkbox)
        
        # Use QDialogButtonBox for buttons instead
        button_box = QDialogButtonBox()
        self.accept_button = button_box.addButton("Accept", QDialogButtonBox.AcceptRole)
        self.accept_button.setEnabled(False)
        self.decline_button = button_box.addButton("Decline", QDialogButtonBox.RejectRole)
        
        # Connect the QDialogButtonBox signals
        button_box.accepted.connect(self.handle_accept)
        button_box.rejected.connect(self.reject)
        
        layout.addWidget(button_box)
        
        # Connect checkbox - Changed to directly compare with Qt.Checked
        def handle_checkbox_state(state):
            print(f"Checkbox state changed to: {state}")
            is_checked = (state == 2)  # Qt.Checked is 2
            print(f"Setting accept button enabled to: {is_checked}")
            self.accept_button.setEnabled(is_checked)
            print(f"Accept button is now enabled: {self.accept_button.isEnabled()}")
            
        self.accept_checkbox.stateChanged.connect(handle_checkbox_state)
        print("UI initialized")
        
        # Dialog styling
        self.setStyleSheet("""
            QDialog {
                background-color: #2c2c2c;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: #f0f0f0;
                border: 1px solid #505050;
                padding: 5px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4c4c4c;
            }
            QPushButton:disabled {
                background-color: #2c2c2c;
                color: #808080;
            }
        """)
    
    def handle_accept(self):
        """Handle accept button click"""
        print("Accept button clicked")
        try:
            self.accept_eula()
            print("EULA accepted successfully")
            self.accept()
        except Exception as e:
            print(f"Error in handle_accept: {str(e)}")
            raise
    
    def get_eula_text(self):
        """Get EULA text from file or return default"""
        try:
            if self.eula_text_file.exists():
                with open(self.eula_text_file, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            self.logger.error(f"Error reading EULA text file: {str(e)}")
        
        # Return default EULA text if file doesn't exist or can't be read
        return """END USER LICENSE AGREEMENT AND TERMS OF SERVICE

Terms of Service
Effective Date: 11/27/2024
⚠️ BETA SOFTWARE NOTICE
This is prototype/beta software. Features may change without notice. Service interruptions may occur.

1. Acceptance of Terms
By using this beta app, you agree to these Terms of Service. If you do not agree, do not use the app.

2. License Validation
Use of the app requires a valid license. Licenses are issued upon subscription and remain active only while the subscription is current.

3. Permissible Use
The app is designed to generate Python code using an integrated LLM. By using the app, you agree:
- To only execute Python code generated by the app
- Not to run external Python scripts through the app

4. Limitations of Liability
This is beta software provided "as is". We are not responsible for any loss, damage, or harm arising from your use. We expressly disclaim all warranties. Use at your own risk.

5. Updates and Termination
We may update the app and these Terms of Service at any time. Continued use after updates constitutes acceptance of revised terms.

End-User License Agreement (EULA)
Effective Date: 11/27/2024
⚠️ BETA SOFTWARE NOTICE
This license applies to prototype software. All terms are subject to change during the beta period.

1. Grant of License
We grant you a non-exclusive, non-transferable license to use this beta app, subject to these terms.

2. Restrictions
You agree not to:
- Reverse engineer, decompile, or disassemble the app
- Share or distribute your license key
- Use the app for any purpose not explicitly permitted
- Use the beta software in production environments

3. Ownership
This app and all associated intellectual property rights are owned by FirstPass. This agreement transfers no ownership.

4. Data Handling
The app collects only the minimal required data for license validation purposes. See Privacy Policy for details.

5. Termination
This license terminates if:
- Your subscription expires or is canceled
- You violate any terms
- Upon termination, you must uninstall and cease all use of the app.

6. Warranty Disclaimer
This beta software is provided "as is" without warranties of any kind. We make no guarantees about functionality, reliability, or fitness for any purpose.



"""
    
    def accept_eula(self):
        """Handle EULA acceptance"""
        print("Saving EULA acceptance")
        try:
            self.accepted = True
            
            # Save acceptance data using storage's config directory
            acceptance_data = {
                "accepted": True,
                "acceptance_date": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            # Ensure config directory exists
            self.storage.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Save acceptance data
            with open(self.eula_config_file, 'w', encoding='utf-8') as f:
                json.dump(acceptance_data, f, indent=2)
            
            self.logger.info("EULA accepted and saved")
            print("EULA acceptance saved successfully")
            
        except Exception as e:
            print(f"Error saving EULA acceptance: {str(e)}")
            self.logger.error(f"Error saving EULA acceptance: {str(e)}")
            raise
    
    @staticmethod
    def check_acceptance(storage) -> bool:
        """Check if EULA has been accepted"""
        try:
            eula_config_file = storage.config_dir / "eula_config.json"
            
            if eula_config_file.exists():
                with open(eula_config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get("accepted", False)
            return False
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error checking EULA acceptance: {str(e)}")
            return False