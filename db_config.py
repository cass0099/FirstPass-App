# db_config.py
from typing import Dict, Optional
import os
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import dotenv
from enum import Enum

class Environment(Enum):
    """Application environment types"""
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"

class DatabaseConfig:
    """Database configuration manager"""
    
    def __init__(self, env_file: str = ".env"):
        self.logger = logging.getLogger(__name__)
        self._load_env(env_file)
        self._init_connection_params()
    
    def _load_env(self, env_file: str):
        """Load environment variables"""
        env_path = Path(env_file)
        if env_path.exists():
            self.logger.info(f"Loading environment from {env_path.absolute()}")
            dotenv.load_dotenv(env_path)
            # Add debug logging
            self.logger.info(f"LOCAL_DB_PORT from env: {os.getenv('LOCAL_DB_PORT', 'not set')}")
            self.logger.info(f"LOCAL_DB_HOST from env: {os.getenv('LOCAL_DB_HOST', 'not set')}")
        else:
            self.logger.warning(f"Environment file not found: {env_path.absolute()}")
        
        self.environment = Environment(os.getenv("APP_ENV", "local").lower())
        self.logger.info(f"Running in {self.environment.value} environment")
    
    def _init_connection_params(self):
        """Initialize database connection parameters based on environment"""
        # Default local settings
        self.db_params = {
            Environment.LOCAL: {
                "dbname": os.getenv("LOCAL_DB_NAME", "license_manager"),
                "user": os.getenv("LOCAL_DB_USER", "postgres"),
                "password": os.getenv("LOCAL_DB_PASSWORD", ""),
                "host": os.getenv("LOCAL_DB_HOST", "localhost"),
                "port": os.getenv("LOCAL_DB_PORT", "5432")  # Changed from DB_PORT to LOCAL_DB_PORT
            },
            Environment.STAGING: {
                "dbname": os.getenv("STAGING_DB_NAME", ""),
                "user": os.getenv("STAGING_DB_USER", ""),
                "password": os.getenv("STAGING_DB_PASSWORD", ""),
                "host": os.getenv("STAGING_DB_HOST", ""),
                "port": os.getenv("STAGING_DB_PORT", "5432")
            },
            Environment.PRODUCTION: {
                "dbname": os.getenv("PROD_DB_NAME", ""),
                "user": os.getenv("PROD_DB_USER", ""),
                "password": os.getenv("PROD_DB_PASSWORD", ""),
                "host": os.getenv("PROD_DB_HOST", ""),
                "port": os.getenv("PROD_DB_PORT", "5432")
            }
        }
    
    def get_connection_params(self) -> Dict:
        """Get current environment's database parameters"""
        return self.db_params[self.environment]

class LicenseDatabase:
    """Database operations for license management"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._init_db()
    
    def _get_connection(self):
        """Create database connection"""
        return psycopg2.connect(
            **self.config.get_connection_params(),
            cursor_factory=RealDictCursor
        )
    
    def _init_db(self):
        """Initialize database schema"""
        schema = """
        -- Enable UUID extension
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        
        -- License tiers
        CREATE TABLE IF NOT EXISTS license_tiers (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50) NOT NULL UNIQUE,
            description TEXT,
            max_users INT NOT NULL DEFAULT 1,
            max_machines INT NOT NULL DEFAULT 1,
            duration_days INT NOT NULL,
            cost_usd DECIMAL(10,2) NOT NULL,
            features JSONB NOT NULL DEFAULT '{}',
            is_active BOOLEAN NOT NULL DEFAULT true,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Licenses
        CREATE TABLE IF NOT EXISTS licenses (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            license_key UUID NOT NULL UNIQUE DEFAULT uuid_generate_v4(),
            tier_id INTEGER REFERENCES license_tiers(id),
            user_email VARCHAR(255) NOT NULL,
            user_name VARCHAR(255),
            status VARCHAR(20) NOT NULL DEFAULT 'active',
            issued_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            last_validated TIMESTAMP WITH TIME ZONE,
            validation_count INTEGER DEFAULT 0,
            metadata JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Machine registrations
        CREATE TABLE IF NOT EXISTS machine_registrations (
            id SERIAL PRIMARY KEY,
            license_id UUID REFERENCES licenses(id),
            machine_id VARCHAR(255) NOT NULL,
            hostname VARCHAR(255),
            first_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT true,
            metadata JSONB NOT NULL DEFAULT '{}',
            UNIQUE(license_id, machine_id)
        );
        
        -- Validation log
        CREATE TABLE IF NOT EXISTS validation_log (
            id SERIAL PRIMARY KEY,
            license_id UUID REFERENCES licenses(id),
            machine_id VARCHAR(255) NOT NULL,
            validated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            is_valid BOOLEAN NOT NULL,
            error_message TEXT,
            metadata JSONB NOT NULL DEFAULT '{}'
        );
        """
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(schema)
                    conn.commit()
        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            raise

    def validate_license(self, license_key: str, machine_id: str) -> Dict:
        """Validate a license"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Get license information
                    cur.execute("""
                        SELECT 
                            l.*, t.max_machines, t.features
                        FROM licenses l
                        JOIN license_tiers t ON l.tier_id = t.id
                        WHERE l.license_key = %s
                    """, (license_key,))
                    
                    license = cur.fetchone()
                    if not license:
                        return {
                            'valid': False,
                            'message': 'Invalid license key'
                        }
                    
                    # Check if license is active
                    if license['status'] != 'active':
                        return {
                            'valid': False,
                            'message': f"License is {license['status']}"
                        }
                    
                    # Check expiration
                    if license['expires_at'] < datetime.now(license['expires_at'].tzinfo):
                        return {
                            'valid': False,
                            'message': 'License has expired'
                        }
                    
                    # Check machine count
                    cur.execute("""
                        SELECT COUNT(*) as count
                        FROM machine_registrations
                        WHERE license_id = %s AND is_active = true
                    """, (license['id'],))
                    
                    machine_count = cur.fetchone()['count']
                    
                    # Get this machine's registration if it exists
                    cur.execute("""
                        SELECT id, is_active
                        FROM machine_registrations
                        WHERE license_id = %s AND machine_id = %s
                    """, (license['id'], machine_id))
                    
                    registration = cur.fetchone()
                    
                    if not registration:
                        if machine_count >= license['max_machines']:
                            return {
                                'valid': False,
                                'message': f"Maximum number of machines ({license['max_machines']}) reached"
                            }
                        
                        # Register new machine
                        cur.execute("""
                            INSERT INTO machine_registrations (
                                license_id, machine_id
                            ) VALUES (%s, %s)
                        """, (license['id'], machine_id))
                    elif not registration['is_active']:
                        return {
                            'valid': False,
                            'message': 'Machine registration is inactive'
                        }
                    
                    # Update license validation info
                    cur.execute("""
                        UPDATE licenses 
                        SET 
                            last_validated = CURRENT_TIMESTAMP,
                            validation_count = validation_count + 1
                        WHERE id = %s
                    """, (license['id'],))
                    
                    # Log validation
                    cur.execute("""
                        INSERT INTO validation_log (
                            license_id, machine_id, is_valid
                        ) VALUES (%s, %s, true)
                    """, (license['id'], machine_id))
                    
                    conn.commit()
                    
                    return {
                        'valid': True,
                        'message': 'License validated successfully',
                        'expires_at': license['expires_at'].isoformat(),
                        'features': license['features']
                    }
                    
        except Exception as e:
            self.logger.error(f"Error validating license: {str(e)}")
            return {
                'valid': False,
                'message': 'License validation error'
            }

    def get_license_info(self, license_key: str) -> Optional[Dict]:
        """Get detailed license information"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            l.license_key,
                            l.user_email,
                            l.user_name,
                            l.status,
                            l.issued_at,
                            l.expires_at,
                            l.last_validated,
                            l.validation_count,
                            t.name as tier_name,
                            t.max_machines,
                            t.features
                        FROM licenses l
                        JOIN license_tiers t ON l.tier_id = t.id
                        WHERE l.license_key = %s
                    """, (license_key,))
                    
                    return cur.fetchone()
        except Exception as e:
            self.logger.error(f"Error getting license info: {str(e)}")
            return None