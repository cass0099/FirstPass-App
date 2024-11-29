# license_generator.py
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Optional, List
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta, date
import calendar
import csv
import uuid
from dataclasses import dataclass
from enum import Enum

class LicenseTier(Enum):
    BASIC = "Basic"
    PROFESSIONAL = "Professional"
    ENTERPRISE = "Enterprise"

@dataclass
class LicenseInfo:
    tier: LicenseTier
    user_email: str
    user_name: Optional[str] = None
    subscription_id: Optional[str] = None  # Stripe subscription ID
    metadata: Optional[Dict] = None

class LicenseGenerator:
    """Utility for generating and managing subscription-based licenses"""
    
    def __init__(self, db_params: Dict):
        self.db_params = db_params
        self.logger = logging.getLogger(__name__)
    
    def _get_connection(self):
        """Create database connection"""
        return psycopg2.connect(**self.db_params, cursor_factory=RealDictCursor)
    
    def _calculate_next_billing_date(self, from_date: Optional[datetime] = None) -> datetime:
        """Calculate next billing date (same day next month)"""
        if from_date is None:
            from_date = datetime.now()
            
        year = from_date.year
        month = from_date.month + 1
        day = from_date.day
        
        # Handle year rollover
        if month > 12:
            month = 1
            year += 1
            
        # Handle edge cases like 31st of month
        _, last_day = calendar.monthrange(year, month)
        if day > last_day:
            day = last_day
            
        next_date = datetime(year, month, day, 
                           from_date.hour, from_date.minute, 
                           from_date.second, from_date.microsecond,
                           from_date.tzinfo)
        
        return next_date
    
    def generate_license(self, license_info: LicenseInfo) -> Dict:
        """Generate a new license"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Calculate expiration date (next billing date)
                    expires_at = self._calculate_next_billing_date()
                    
                    # Get tier ID
                    cur.execute("""
                        SELECT id 
                        FROM license_tiers 
                        WHERE name = %s AND is_active = true
                    """, (license_info.tier.value,))
                    
                    tier_result = cur.fetchone()
                    if not tier_result:
                        raise ValueError(f"Invalid tier: {license_info.tier.value}")
                    
                    tier_id = tier_result['id']
                    
                    # Create license with subscription info
                    cur.execute("""
                        INSERT INTO licenses (
                            tier_id,
                            user_email,
                            user_name,
                            expires_at,
                            metadata
                        ) VALUES (
                            %s, %s, %s, %s, %s
                        ) RETURNING id, license_key, expires_at
                    """, (
                        tier_id,
                        license_info.user_email,
                        license_info.user_name,
                        expires_at,
                        json.dumps({
                            **(license_info.metadata or {}),
                            'subscription_id': license_info.subscription_id,
                            'billing_cycle_anchor': expires_at.isoformat()
                        })
                    ))
                    
                    result = cur.fetchone()
                    
                    # Get tier features
                    cur.execute("""
                        SELECT features
                        FROM license_tiers
                        WHERE id = %s
                    """, (tier_id,))
                    
                    features = cur.fetchone()['features']
                    
                    return {
                        'license_key': result['license_key'],
                        'expires_at': result['expires_at'],
                        'features': features
                    }
                    
        except Exception as e:
            self.logger.error(f"Error generating license: {str(e)}")
            raise
    
    def renew_subscription_license(self, subscription_id: str) -> Dict:
        """Renew a license based on Stripe subscription"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Find license by subscription ID
                    cur.execute("""
                        SELECT id, expires_at
                        FROM licenses
                        WHERE metadata->>'subscription_id' = %s
                        AND status = 'active'
                    """, (subscription_id,))
                    
                    license = cur.fetchone()
                    if not license:
                        raise ValueError(f"No active license found for subscription {subscription_id}")
                    
                    # Calculate next expiration from current expiration
                    current_expires = license['expires_at']
                    next_expires = self._calculate_next_billing_date(current_expires)
                    
                    # Update license expiration
                    cur.execute("""
                        UPDATE licenses
                        SET 
                            expires_at = %s,
                            metadata = jsonb_set(
                                metadata,
                                '{billing_cycle_anchor}',
                                %s::jsonb
                            )
                        WHERE id = %s
                        RETURNING license_key, expires_at
                    """, (
                        next_expires,
                        json.dumps(next_expires.isoformat()),
                        license['id']
                    ))
                    
                    return cur.fetchone()
                    
        except Exception as e:
            self.logger.error(f"Error renewing license: {str(e)}")
            raise
    
    def cancel_subscription(self, subscription_id: str) -> bool:
        """Cancel a subscription license"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE licenses
                        SET status = 'expired'
                        WHERE metadata->>'subscription_id' = %s
                        AND status = 'active'
                        RETURNING id
                    """, (subscription_id,))
                    
                    return cur.fetchone() is not None
                    
        except Exception as e:
            self.logger.error(f"Error cancelling subscription: {str(e)}")
            raise

    def get_licenses_expiring_soon(self, days_threshold: int = 7) -> List[Dict]:
        """Get licenses approaching expiration"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            license_key,
                            user_email,
                            user_name,
                            expires_at,
                            metadata->>'subscription_id' as subscription_id
                        FROM licenses
                        WHERE status = 'active'
                        AND expires_at < CURRENT_TIMESTAMP + interval '%s days'
                        AND expires_at > CURRENT_TIMESTAMP
                        ORDER BY expires_at
                    """, (days_threshold,))
                    
                    return cur.fetchall()
                    
        except Exception as e:
            self.logger.error(f"Error getting expiring licenses: {str(e)}")
            raise

# Example usage in your Stripe webhook handler:
"""
# When a subscription is created
license_info = LicenseInfo(
    tier=LicenseTier.PROFESSIONAL,
    user_email="user@example.com",
    user_name="John Doe",
    subscription_id=subscription.id,
    metadata={
        "stripe_customer_id": customer.id,
        "product_id": product.id
    }
)
license = generator.generate_license(license_info)

# When a subscription payment succeeds
generator.renew_subscription_license(subscription.id)

# When a subscription is cancelled
generator.cancel_subscription(subscription.id)
"""