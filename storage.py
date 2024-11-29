# storage.py
import json
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from base64 import b64encode, b64decode
import logging
from typing import Optional, Dict, List
import platform
import tkinter as tk
from tkinter import messagebox
    
class StorageManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Determine if we're running as packaged app or in development
        if getattr(sys, 'frozen', False):
            # Running as packaged executable
            exe_path = Path(sys.executable)
            app_name = "FirstPass"
            
            # Create parent directory if executable is not already in it
            if exe_path.parent.name != app_name:
                # Get the desktop path
                if os.name == 'nt':  # Windows
                    parent_dir = Path(os.path.expanduser("~/Desktop")) / app_name
                else:  # macOS and Linux
                    parent_dir = Path(os.path.expanduser("~/Desktop")) / app_name
                
                # Create parent directory if it doesn't exist
                parent_dir.mkdir(parents=True, exist_ok=True)
                
                # If this is the first run (exe not in app directory)
                if exe_path.parent != parent_dir:
                    try:
                        # Copy executable to new location
                        new_exe_path = parent_dir / exe_path.name
                        if not new_exe_path.exists():  # Only copy if doesn't exist
                            shutil.copy2(exe_path, new_exe_path)
                            self.logger.info(f"Copied executable to: {new_exe_path}")
                            
                            # Create a message to show the user
                            msg = (f"FirstPass has been installed to:\n{parent_dir}\n\n"
                                  f"Please run FirstPass from that location.\n"
                                  f"You can delete this temporary copy.")
                            
                            # Show message using native dialog
                            if os.name == 'nt':
                                import ctypes
                                ctypes.windll.user32.MessageBoxW(0, msg, "FirstPass Installation", 0x40)
                            else:
                                root = tk.Tk()
                                root.withdraw()
                                messagebox.showinfo("FirstPass Installation", msg)
                            
                            # Exit this instance
                            sys.exit(0)
                    except Exception as e:
                        self.logger.error(f"Error setting up application directory: {str(e)}")
            
                base_path = parent_dir / "firstpass_data"
            else:
                # Already in correct parent directory
                base_path = exe_path.parent / "firstpass_data"
                
            self.logger.info(f"Running as packaged app, using base path: {base_path}")
        else:
            # Running in development - use C:\FirstPassData
            base_path = Path('C:/FirstPassData')
            self.logger.info(f"Running in development, using base path: {base_path}")
        
        # Set up directory structure
        self.app_dir = base_path
        self.config_dir = self.app_dir / "config"
        self.data_dir = self.app_dir / "data"
        self.cache_dir = self.app_dir / "cache"
        self.venv_dir = self.app_dir / "venv"
        self.script_environments_dir = self.app_dir / "script_environments"
        self.packages_dir = self.app_dir / "packages"
        self.logs_dir = self.app_dir / "logs"
        self.prompts_dir = self.data_dir / "prompts"  # Fixed attribute name
        
        # Set up file paths
        self.api_key_file = self.config_dir / "api_key.enc"
        self.license_file = self.config_dir / "license.enc"
        self.conversation_history_file = self.data_dir / "conversation_history.json"
        self.config_path = self.config_dir / "config.json"
        
        # Initialize storage
        self._create_directories()
        self._setup_encryption()
    
    def _setup_encryption(self):
        """Initialize encryption for secure data storage"""
        salt = b'AppDataEncryptionSalt'
        machine_id = self._get_machine_id().encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(machine_id))
        self.fernet = Fernet(key)
    
    def _get_machine_id(self) -> str:
        """Get unique machine identifier for encryption"""
        system_info = {
            'platform': platform.system(),
            'node': platform.node(),
            'processor': platform.processor()
        }
        return '-'.join(str(v) for v in system_info.values())
    
    def _create_directories(self):
        """Create necessary directory structure"""
        dirs = [
            self.app_dir,
            self.config_dir,
            self.data_dir,
            self.cache_dir,
            self.venv_dir,
            self.script_environments_dir,
            self.packages_dir,
            self.logs_dir,
            self.prompts_dir
        ]
        
        for directory in dirs:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {directory}")
            except Exception as e:
                self.logger.error(f"Error creating directory {directory}: {str(e)}")
                raise

    def save_prompt_log(self, entry: Dict) -> None:
        try:
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            log_file = self.prompts_dir / f"prompt_{timestamp}.txt"

            # Format log content
            log_content = [
                f"Timestamp: {entry.get('timestamp', datetime.now().isoformat())}",
                f"Provider: {entry.get('provider', '')}",
                f"Model: {entry.get('model', '')}",
                f"RAG Enabled: {entry.get('rag_enabled', 'false')}",
                "",
                "=== USER PROMPT ===",
                entry.get('user_prompt', '').strip(),
                "",
                "=== BASE PROMPT ===",
                entry.get('base_prompt', 'Not provided').strip()
            ]

            if entry.get('rag_enabled') and entry.get('enhanced_prompt'):
                log_content.extend(["", "=== ENHANCED PROMPT ===", entry['enhanced_prompt'].strip()])

            if entry.get('metadata'):
                log_content.extend(["", "=== METADATA ===", json.dumps(entry['metadata'], indent=2)])

            log_content.extend(["", "=== LLM RESPONSE ===", entry.get('response', '').strip()])

            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(log_content))

            self._clean_old_prompt_logs()

        except Exception as e:
            self.logger.error(f"Error saving prompt log: {str(e)}")

    def get_prompt_logs(self, days: int = 7) -> List[Dict]:
        try:
            if not self.prompts_dir.exists():
                return []

            logs = []
            cutoff_date = datetime.now() - timedelta(days=days)

            for log_file in self.prompts_dir.glob("prompt_*.txt"):
                try:
                    file_date_str = log_file.stem.split('_')[1]
                    file_date = datetime.strptime(file_date_str, '%Y%m%d')

                    if file_date >= cutoff_date:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            logs.append(f.read())
                except ValueError:
                    continue

            return sorted(logs, reverse=True)

        except Exception as e:
            self.logger.error(f"Error getting prompt logs: {str(e)}")
            return []

    def _clean_old_prompt_logs(self, keep_days: int = 30) -> None:
        try:
            if not self.prompts_dir.exists():
                return
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            
            for log_file in self.prompts_dir.glob("prompt_*.txt"):
                try:
                    file_date_str = log_file.stem.split('_')[1]
                    file_date = datetime.strptime(file_date_str, '%Y%m%d')
                    if file_date < cutoff_date:
                        log_file.unlink()
                except ValueError:
                    continue
        except Exception as e:
            self.logger.error(f"Error cleaning prompt logs: {str(e)}")

    
    def save_api_keys(self, keys: dict):
        """Save encrypted API keys to config file"""
        try:
            config = {}
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            
            # Encrypt sensitive data
            encrypted_keys = {}
            for provider, data in keys.items():
                encrypted_keys[provider] = {
                    'key': b64encode(self.fernet.encrypt(data['key'].encode())).decode()
                }
            
            config['api_keys'] = encrypted_keys
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving API keys: {e}")

    def load_api_keys(self):
        """Load and decrypt API keys"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    encrypted_keys = config.get('api_keys', {})
                    
                    # Decrypt keys
                    decrypted_keys = {}
                    for provider, data in encrypted_keys.items():
                        encrypted = b64decode(data['key'].encode())
                        decrypted_keys[provider] = {
                            'key': self.fernet.decrypt(encrypted).decode()
                        }
                    return decrypted_keys
            return {}
        except Exception as e:
            self.logger.error(f"Error loading API keys: {e}")
            return {}

    def save_provider_settings(self, settings: dict):
        """Save provider settings to config file"""
        try:
            config = {}
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            
            config['provider_settings'] = settings
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving provider settings: {e}")

    def load_provider_settings(self) -> dict:
        """Load provider settings from config file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('provider_settings', {})
            return {}
        except Exception as e:
            self.logger.error(f"Error loading provider settings: {e}")
            return {}
    
    def save_conversation_history(self, history: list):
        """Save conversation history"""
        try:
            with open(self.conversation_history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
            self.logger.info("Conversation history saved")
        except Exception as e:
            self.logger.error(f"Error saving conversation history: {str(e)}")
            raise
    
    def load_conversation_history(self) -> list:
        """Load conversation history"""
        try:
            if not self.conversation_history_file.exists():
                return []
            with open(self.conversation_history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading conversation history: {str(e)}")
            return []
    
    def get_venv_path(self, environment_name: str) -> Path:
        """Get the path for a virtual environment"""
        return self.venv_dir / environment_name
    
    def get_script_environment_path(self, environment_name: str) -> Path:
        """Get the path for a script environment"""
        return self.script_environments_dir / environment_name
    
    def get_package_path(self, package_name: str) -> Path:
        """Get the path for a package"""
        return self.packages_dir / package_name
    
    def clear_all_data(self):
        """Clear all stored application data"""
        try:
            # Only remove files, keep directory structure
            for file in [self.api_key_file, self.license_file, self.conversation_history_file]:
                if file.exists():
                    file.unlink()
            
            # Clear directories but keep them
            for directory in [self.cache_dir, self.script_environments_dir, self.prompts_dir]:
                if directory.exists():
                    shutil.rmtree(directory)
                    directory.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("All application data cleared")
        except Exception as e:
            self.logger.error(f"Error clearing data: {str(e)}")
            raise
    
    def get_storage_info(self) -> Dict:
        """Get information about stored data"""
        info = {
            "app_directory": str(self.app_dir),
            "config_directory": str(self.config_dir),
            "data_directory": str(self.data_dir),
            "cache_directory": str(self.cache_dir),
            "venv_directory": str(self.venv_dir),
            "script_environments_directory": str(self.script_environments_dir),
            "packages_directory": str(self.packages_dir),
            "has_api_key": self.api_key_file.exists(),
            "has_conversation_history": self.conversation_history_file.exists(),
            "has_license": self.license_file.exists(),
            "total_size": self._get_directory_size(self.app_dir)
        }
        
        # Add prompt log info - Updated to look for .json files instead of .jsonl
        if self.prompts_dir.exists():  # Use consistent attribute name
            log_files = list(self.prompts_dir.glob("prompt_*.json"))
            info.update({
                "prompts_directory": str(self.prompts_dir),
                "prompt_files": len(log_files),
                "latest_prompt": max(log_files).name if log_files else None
            })
            
        return info
    
    def _get_directory_size(self, directory: Path) -> int:
        """Calculate total size of a directory in bytes"""
        return sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())