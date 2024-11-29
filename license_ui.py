from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QMessageBox, QDialog, QApplication, QMainWindow, QDialogButtonBox, QProgressBar
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QDateTime
import logging
import webbrowser
import sys
import traceback
from typing import Optional, Dict, Any

# Get module logger without configuration
logger = logging.getLogger(__name__)

class LicenseValidationThread(QThread):
    """Thread for handling license validation"""
    validation_complete = Signal(dict)
    
    def __init__(self, license_manager, license_key):
        super().__init__()
        self.license_manager = license_manager
        self.license_key = license_key
    
    def run(self):
        logger.debug(f"Validation thread starting for key: {self.license_key[:8] if self.license_key else 'None'}...")
        try:
            result = self.license_manager.validate_license(self.license_key)
            logger.debug(f"Validation thread result: {result}")
            self.validation_complete.emit(result)
        except Exception as e:
            logger.error(f"Validation thread error: {str(e)}", exc_info=True)
            self.validation_complete.emit({
                'valid': False,
                'message': f"Validation error: {str(e)}"
            })

class LicenseDialog(QDialog):
    """Dialog for license key input and validation"""
    
    def __init__(self, license_manager, parent=None):
        super().__init__(parent)
        logger.debug("Initializing LicenseDialog")
        self.license_manager = license_manager
        self.setWindowTitle("License Validation")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        self.validation_thread = None

        # Set window flags to force stay on top
        self.setWindowFlags(
            Qt.Dialog |
            Qt.WindowStaysOnTopHint |
            Qt.WindowTitleHint |
            Qt.CustomizeWindowHint |
            Qt.WindowCloseButtonHint |
            Qt.MSWindowsFixedSizeDialogHint
        )

        # Force always on top
        self.setAttribute(Qt.WA_AlwaysStackOnTop)
        
        # Create focus check timer
        self.focus_timer = QTimer(self)
        self.focus_timer.timeout.connect(self.check_focus)
        self.focus_timer.start(100)  # Check every 100ms

        self.init_ui()
        logger.debug("LicenseDialog initialization complete")

    def check_focus(self):
        """Ensure window stays on top and focused"""
        if not self.isActiveWindow():
            self.activateWindow()
            self.raise_()

    def init_ui(self):
        """Initialize the license dialog UI"""
        logger.debug("Initializing UI components")
        layout = QVBoxLayout()
        
        # Add info label
        info_label = QLabel("Please enter your license key to activate the application.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # License input
        input_layout = QHBoxLayout()
        self.license_input = QLineEdit()
        self.license_input.setPlaceholderText("Enter your license key")
        self.license_input.setMinimumWidth(300)
        self.license_input.returnPressed.connect(self.validate_license)
        input_layout.addWidget(QLabel("License Key:"))
        input_layout.addWidget(self.license_input)
        layout.addLayout(input_layout)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.validate_btn = QPushButton("Validate License")
        self.validate_btn.clicked.connect(self.validate_license)
        cancel_btn = QPushButton("Exit Application")
        cancel_btn.clicked.connect(self.exit_application)
        button_layout.addWidget(self.validate_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        logger.debug("UI initialization complete")

    def validate_license(self):
        """Handle license validation"""
        logger.debug("Starting license validation process")
        try:
            license_key = self.license_input.text().strip()
            
            if not license_key:
                QMessageBox.warning(self, "Error", "Please enter a license key")
                return

            self.license_input.setEnabled(False)
            self.validate_btn.setEnabled(False)
            self.status_label.setText("Activating license...")
            self.status_label.setStyleSheet("color: blue")
            QApplication.processEvents()

            # First try to activate the license
            result = self.license_manager.activate_license(license_key)
            logger.debug(f"Activation result: {result}")
            
            if result.get('message'):
                self.status_label.setText(result['message'])
                self.status_label.setStyleSheet("color: red")
            
            self.handle_validation_result(result)

        except Exception as e:
            logger.error(f"Error in validate_license: {str(e)}")
            self.handle_validation_result({
                'valid': False,
                'message': f"Error: {str(e)}"
            })

    def handle_validation_result(self, result):
        """Handle the validation result"""
        logger.debug(f"Handling validation result: {result}")
        try:
            # Re-enable input
            self.license_input.setEnabled(True)
            self.validate_btn.setEnabled(True)
            
            if result.get('valid', False):
                self.status_label.setText("License validated successfully!")
                self.status_label.setStyleSheet("color: green")
                logger.debug("License validation successful")
                
                # Show success message and close after short delay
                QTimer.singleShot(1500, lambda: self.accept())
                
                # Update parent window status if applicable
                if isinstance(self.parent(), QMainWindow):
                    logger.debug("Updating parent window status")
                    self.parent().update_license_status()
                else:
                    logger.warning(f"Parent is not QMainWindow: {type(self.parent())}")
            else:
                error_message = result.get('message', 'License validation failed')
                self.status_label.setText(f"Error: {error_message}")
                self.status_label.setStyleSheet("color: red")
                logger.warning(f"License validation failed: {error_message}")

        except Exception as e:
            logger.error(f"Error handling validation result: {str(e)}")
            logger.error(traceback.format_exc())
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: red")

    def exit_application(self):
        """Exit the entire application"""
        logger.debug("Exiting application from license dialog")
        if hasattr(self, 'focus_timer'):
            self.focus_timer.stop()
        self.reject()
        QApplication.instance().quit()
    
    def closeEvent(self, event):
        """Handle close event"""
        logger.debug("License dialog close event triggered")
        if hasattr(self, 'focus_timer'):
            self.focus_timer.stop()
        self.exit_application()
        event.accept()

    def showEvent(self, event):
        """Override show event to center and activate window"""
        super().showEvent(event)
        # Center on screen using the modern approach
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )
        self.activateWindow()
        self.raise_()
        self.license_input.setFocus()
        logger.debug("Dialog shown and centered")

class LicenseInvalidDialog(QDialog):
    """Dialog shown when license is invalid or expired"""
    
    def __init__(self, message: str, parent=None):
        super().__init__(parent)
        logger.debug("Initializing LicenseInvalidDialog")
        self.setWindowTitle("License Required")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        # Set window flags to force stay on top
        self.setWindowFlags(
            Qt.Dialog |
            Qt.WindowStaysOnTopHint |
            Qt.WindowTitleHint |
            Qt.CustomizeWindowHint |
            Qt.WindowCloseButtonHint |
            Qt.MSWindowsFixedSizeDialogHint
        )
        
        # Force always on top
        self.setAttribute(Qt.WA_AlwaysStackOnTop)
        
        # Create focus check timer
        self.focus_timer = QTimer(self)
        self.focus_timer.timeout.connect(self.check_focus)
        self.focus_timer.start(100)

        self.message = message
        self.init_ui()
        logger.debug("LicenseInvalidDialog initialization complete")

    def check_focus(self):
        """Ensure window stays on top and focused"""
        if not self.isActiveWindow():
            self.activateWindow()
            self.raise_()

    def init_ui(self):
        """Initialize the invalid license dialog UI"""
        layout = QVBoxLayout()
        
        # Error message
        msg_label = QLabel(self.message)
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)
        
        # Purchase button
        purchase_btn = QPushButton("Purchase License")
        purchase_btn.clicked.connect(self.open_purchase_page)
        layout.addWidget(purchase_btn)
        
        # Enter key button
        enter_key_btn = QPushButton("Enter License Key")
        enter_key_btn.clicked.connect(self.on_enter_key)
        layout.addWidget(enter_key_btn)
        
        # Exit button
        exit_btn = QPushButton("Exit Application")
        exit_btn.clicked.connect(self.exit_application)
        layout.addWidget(exit_btn)
        
        self.setLayout(layout)
        logger.debug("LicenseInvalidDialog UI initialized")

    def on_enter_key(self):
        """Handle enter key button click"""
        logger.debug("Enter key button clicked")
        if hasattr(self, 'focus_timer'):
            self.focus_timer.stop()
        self.accept()

    def showEvent(self, event):
        """Override show event to center and activate window"""
        super().showEvent(event)
        # Center on screen using the modern approach
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )
        self.activateWindow()
        self.raise_()
    
    def closeEvent(self, event):
        """Handle close event"""
        logger.debug("Invalid license dialog close event triggered")
        if hasattr(self, 'focus_timer'):
            self.focus_timer.stop()
        self.exit_application()
        event.accept()

    def exit_application(self):
        """Properly exit the entire application"""
        logger.debug("Exiting application from invalid license dialog")
        if hasattr(self, 'focus_timer'):
            self.focus_timer.stop()
        self.done(2)  # Use custom return code
        QApplication.instance().quit()
    
    def open_purchase_page(self):
        """Open the license purchase webpage"""
        logger.debug("Opening purchase page")
        webbrowser.open('https://usefirstpass.com')

    def __del__(self):
        """Cleanup on deletion"""
        try:
            if hasattr(self, 'focus_timer'):
                self.focus_timer.stop()
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")