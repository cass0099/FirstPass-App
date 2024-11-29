# license_expired_view.py

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, 
    QApplication, QStyle, QHBoxLayout, QSpacerItem, 
    QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QIcon
import webbrowser

class LicenseExpiredView(QWidget):
    """Full screen view shown when license expires"""
    # Signals for handling user actions
    license_renewed = Signal()  # Emitted when user enters new valid license
    quit_requested = Signal()   # Emitted when user chooses to quit

    def __init__(self, parent=None, message="Your license has expired"):
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.message = message
        self.init_ui()

    def init_ui(self):
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create central widget with white background
        content_widget = QWidget()
        content_widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 8px;
            }
        """)
        content_layout = QVBoxLayout()
        content_layout.setAlignment(Qt.AlignCenter)
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(40, 40, 40, 40)

        # Warning icon
        icon_label = QLabel()
        warning_icon = QApplication.style().standardIcon(QStyle.SP_MessageBoxWarning)
        icon_label.setPixmap(warning_icon.pixmap(64, 64))
        content_layout.addWidget(icon_label, alignment=Qt.AlignCenter)

        # Main message
        message_label = QLabel("License Expired")
        message_label.setFont(QFont("Arial", 18, QFont.Bold))
        content_layout.addWidget(message_label, alignment=Qt.AlignCenter)

        # Detailed message
        detail_label = QLabel(self.message)
        detail_label.setWordWrap(True)
        detail_label.setStyleSheet("color: #666;")
        content_layout.addWidget(detail_label, alignment=Qt.AlignCenter)

        # Spacer
        content_layout.addSpacing(20)

        # Action buttons
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)

        # Renew button with prominent styling
        renew_btn = QPushButton("Renew License")
        renew_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        renew_btn.clicked.connect(self.open_renewal_page)
        button_layout.addWidget(renew_btn)

        # Enter license key button
        enter_key_btn = QPushButton("Enter License Key")
        enter_key_btn.setStyleSheet("""
            QPushButton {
                border: 1px solid #2196F3;
                color: #2196F3;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                background-color: white;
            }
            QPushButton:hover {
                background-color: #E3F2FD;
            }
        """)
        enter_key_btn.clicked.connect(self.license_renewed.emit)
        button_layout.addWidget(enter_key_btn)

        # Quit button
        quit_btn = QPushButton("Quit Application")
        quit_btn.setStyleSheet("""
            QPushButton {
                color: #666;
                padding: 10px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        quit_btn.clicked.connect(self.quit_requested.emit)
        button_layout.addWidget(quit_btn)

        content_layout.addLayout(button_layout)
        content_widget.setLayout(content_layout)

        # Add content widget to main layout
        layout.addWidget(content_widget)
        self.setLayout(layout)

    def open_renewal_page(self):
        """Open the license renewal webpage"""
        try:
            webbrowser.open('https://your-domain.com/renew')
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Error",
                "Could not open the renewal page. Please visit our website manually.",
                QMessageBox.Ok
            )