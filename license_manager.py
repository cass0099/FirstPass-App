# license_manager.py
import requests
import json
import hashlib
import platform
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
import logging
import uuid
import os
from pathlib import Path
from cryptography.fernet import Fernet
from base64 import b64encode, b64decode

class LicenseManager:
    def __init__(self, storage_manager):
        self.logger = logging.getLogger(__name__)
        self.storage_manager = storage_manager  # Store reference
        self.api_url = "https://firstpass-production.up.railway.app"
        self.validated_key = None
        self.validated_features = None
        self.storage_path = storage_manager.config_dir / "license.json"
        self.logger.info(f"LicenseManager initialized with API URL: {self.api_url}")
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.load_saved_license()


    def _get_machine_id(self) -> str:
        """Generate unique machine identifier"""
        self.logger.debug("Generating machine ID...")
        try:
            system_info = {
                'platform': platform.system(),
                'processor': platform.processor(),
                'machine': platform.machine(),
                'node': platform.node()
            }
            machine_id = '-'.join(str(v) for v in system_info.values() if v)
            hashed_id = hashlib.sha256(machine_id.encode()).hexdigest()
            self.logger.debug(f"Generated machine ID: {hashed_id[:8]}...")
            return hashed_id
        except Exception as e:
            self.logger.error(f"Error generating machine ID: {str(e)}")
            fallback_id = str(uuid.uuid4())
            self.logger.warning(f"Using fallback machine ID: {fallback_id}")
            return fallback_id

    def save_license_key(self):
        try:
            if self.validated_key:
                data = {
                    'license_key': self.validated_key,  # Already encrypted, don't encrypt again
                    'features': self.validated_features,
                    'saved_at': datetime.now(timezone.utc).isoformat(),
                    'machine_id': self._get_machine_id()
                }
                
                with open(self.storage_path, 'w') as f:
                    json.dump(data, f)
                        
        except Exception as e:
            self.logger.error(f"Error saving license key: {str(e)}")

    def _is_license_file_valid(self, data: dict) -> bool:
        """Check if the saved license file is valid and not too old"""
        try:
            # Check all required fields exist
            required_fields = ['license_key', 'saved_at', 'machine_id']
            if not all(field in data for field in required_fields):
                self.logger.warning("Saved license file missing required fields")
                return False

            # Parse saved timestamp
            saved_at = datetime.fromisoformat(data['saved_at'])
            
            # Check if the save is not too old (e.g., more than 30 days)
            max_age = timedelta(days=30)
            if datetime.now(timezone.utc) - saved_at > max_age:
                self.logger.warning("Saved license file is too old")
                return False

            # Verify machine ID hasn't changed
            if data['machine_id'] != self._get_machine_id():
                self.logger.warning("Machine ID mismatch in saved license")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Error validating license file: {str(e)}")
            return False

    def load_saved_license(self):
        try:
            if not self.storage_path.exists():
                return
                
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
            if not self._is_license_file_valid(data):
                return
            
            # Keep encrypted key
            self.validated_key = data['license_key']
            self.validated_features = data.get('features', {})
            
            # Validate with server but don't delete file on failure
            try:
                encrypted = b64decode(self.validated_key.encode())
                decrypted_key = self.storage_manager.fernet.decrypt(encrypted).decode()
                validation_result = self.validate_license(decrypted_key)
                
                if validation_result.get('valid'):
                    return  # Success - keep everything as is
                
                # On validation failure, clear memory but keep file
                self.validated_key = None
                self.validated_features = None
                    
            except Exception as e:
                self.logger.error(f"Error during validation: {str(e)}")
                self.validated_key = None
                self.validated_features = None
                
        except Exception as e:
            self.logger.error(f"Error loading license: {str(e)}")
            self.validated_key = None
            self.validated_features = None

    def activate_license(self, license_key: str) -> Dict:
        try:
            machine_id = self._get_machine_id()
            self.logger.info(f"Activating license with machine ID: {machine_id}")
            
            activation_url = f"{self.api_url}/licenses/activate-new"
            payload = {
                "license_key": license_key,
                "machine_id": machine_id
            }
            
            response = requests.post(activation_url, json=payload, timeout=10)
            self.logger.info(f"Raw response: {response.text}")
            
            try:
                response_data = response.json()
                self.logger.info(f"Parsed response data: {response_data}")
            except json.JSONDecodeError:
                self.logger.error("Failed to parse JSON response")
                return {'valid': False, 'message': 'Error processing server response'}

            if response.status_code == 200 and response_data.get('success'):
                encrypted = self.storage_manager.fernet.encrypt(license_key.encode())
                self.validated_key = b64encode(encrypted).decode()
                self.validated_features = response_data.get('features', {})
                self.save_license_key()
                return {
                    'valid': True,
                    'status': response_data.get('status'),
                    'message': 'License activated successfully'
                }
            
            error_message = response_data.get('detail')
            if error_message:
                return {'valid': False, 'message': error_message}
                
            if "already activated" in response.text:
                return self.validate_license(license_key)

            return {'valid': False, 'message': 'License activation failed'}

        except Exception as e:
            self.logger.error(f"Error activating license: {str(e)}")
            return {'valid': False, 'message': str(e)}

    def validate_license(self, license_key: Optional[str] = None) -> Dict:
        try:
            # Only decrypt if no key provided
            if license_key is None:
                if not self.validated_key:
                    return {'valid': False, 'message': 'No license key available'}
                # Decrypt stored key
                try:
                    encrypted = b64decode(self.validated_key.encode())
                    license_key = self.storage_manager.fernet.decrypt(encrypted).decode()
                except Exception as e:
                    self.validated_key = None
                    return {'valid': False, 'message': 'Invalid stored key'}

            # Validate with server
            machine_id = self._get_machine_id()
            response = requests.post(
                f"{self.api_url}/licenses/validate",
                json={
                    "license_key": license_key,
                    "machine_id": machine_id
                },
                timeout=10
            )
            
            result = response.json()
            if response.status_code == 200 and result.get('valid', False):
                # Clear any existing encrypted key
                self.validated_key = None 
                
                # Encrypt the raw key and store
                encrypted = self.storage_manager.fernet.encrypt(license_key.encode())
                self.validated_key = b64encode(encrypted).decode()
                self.validated_features = result.get('features', {})
                self.save_license_key()
                return {
                    'valid': True,
                    'status': result.get('status'),
                    'expires_at': result.get('expires_at')
                }
            
            self.validated_key = None
            return {'valid': False, 'message': result.get('message', 'License validation failed')}

        except Exception as e:
            self.validated_key = None
            return {'valid': False, 'message': str(e)}

    def invalidate_license(self):
        """Clear license state from memory only"""
        self.validated_key = None
        self.validated_features = None
        self.logger.debug("Cleared license from memory")

    def get_license_info(self) -> Optional[Dict]:
        if not self.validated_key:
            return None
            
        try:
            # Decrypt key before sending to server
            encrypted = b64decode(self.validated_key.encode())
            decrypted_key = self.storage_manager.fernet.decrypt(encrypted).decode()
            
            response = requests.get(
                f"{self.api_url}/licenses/info/{decrypted_key}",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                data['valid'] = True
                if data.get('type') == 'subscription':
                    data['license_type'] = 'Subscription'
                    if data.get('renews_at'):
                        data['next_renewal'] = data['renews_at'].split('T')[0]
                else:
                    data['license_type'] = 'Perpetual'
                
                if data.get('expires_at'):
                    data['expiration'] = data['expires_at'].split('T')[0]
                
                return data
            elif response.status_code == 404:
                self.logger.warning("License not found in database")
                return None
            else:
                self.logger.error(f"Failed to get license info: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting license info: {str(e)}")
            return None

    def get_active_license_key(self) -> Optional[str]:
        """Get the currently active license key"""
        if self.validated_key:
            try:
                # Decrypt key before returning
                encrypted = b64decode(self.validated_key.encode())
                return self.storage_manager.fernet.decrypt(encrypted).decode()
            except Exception as e:
                self.logger.error(f"Error decrypting license key: {str(e)}")
                return None
        return None

    def get_features(self) -> Dict:
        """Get the features of the current license"""
        return self.validated_features or {}

    def cleanup(self):
        """Cleanup method to be called on application shutdown"""
        self.logger.debug("Performing license manager cleanup")
        if self.validated_key:
            self.save_license_key()  # Save final state