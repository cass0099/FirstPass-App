# analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import re
from pathlib import Path
from datetime import datetime
import logging
from scipy import stats

class EnhancedJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling non-serializable types"""
    def default(self, obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8, np.uint16,
            np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super(EnhancedJSONEncoder, self).default(obj)

class DataAnalyzer:
    """
    Enhanced data structure analyzer that extracts comprehensive metadata
    while maintaining data privacy.
    """
    
    PII_SEMANTIC_TYPES = {
        'email', 'phone', 'id', 'name', 'address'
    }
    
    @staticmethod
    def detect_semantic_type(column_name: str, sample_values: pd.Series) -> str:
        """Detect semantic type of column based on name and values"""
        try:
            name_lower = column_name.lower()
            patterns = {
                'email': (r'^.*e[-_]?mail.*$', r'^.*@.*\..+$'),
                'phone': (r'^.*phone.*$', r'^\+?1?\d{9,15}$'),
                'date': (r'^.*(date|dt|timestamp).*$', None),
                'id': (r'^.*(_id|id_|identifier).*$', None),
                'name': (r'^.*(name|title).*$', None),
                'address': (r'^.*(address|location|city|state|country|zip|postal).*$', None),
                'currency': (r'^.*(price|cost|revenue|sales|amount).*$', None),
                'percentage': (r'^.*(percentage|ratio|rate|pct).*$', None),
                'quantity': (r'^.*(quantity|count|num|number|amt).*$', None),
                'url': (r'^.*(url|link|website).*$', r'^https?://'),
                'boolean': (r'^.*(is_|has_|flag|indicator).*$', None)
            }
            
            for semantic_type, (name_pattern, value_pattern) in patterns.items():
                if re.match(name_pattern, name_lower):
                    if value_pattern is None:
                        return semantic_type
                    if not sample_values.empty and sample_values.notna().any():
                        first_valid = str(sample_values[sample_values.notna()].iloc[0])
                        if re.match(value_pattern, first_valid):
                            return semantic_type
            return 'unknown'
        except Exception as e:
            logging.error(f"Error in detect_semantic_type: {str(e)}")
            return 'unknown'

    @staticmethod
    def analyze_column_patterns(column: pd.Series) -> Dict:
        """Analyze patterns in column values"""
        try:
            non_null_values = column.dropna()
            if len(non_null_values) == 0:
                return {"pattern_type": "empty"}
            
            sample = str(non_null_values.iloc[0])
            patterns = {
                "numeric_only": r'^\d+$',
                "alphanumeric": r'^[a-zA-Z0-9]+$',
                "email_pattern": r'^[^@]+@[^@]+\.[^@]+$',
                "date_pattern": r'^\d{4}-\d{2}-\d{2}$',
                "timestamp_pattern": r'^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}',
                "uuid_pattern": r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
                "url_pattern": r'^https?://',
                "phone_pattern": r'^\+?1?\d{9,15}$',
                "postal_code": r'^\d{5}(-\d{4})?$'
            }
            
            for pattern_name, regex in patterns.items():
                if re.match(regex, sample, re.IGNORECASE):
                    return {"pattern_type": pattern_name}
            return {"pattern_type": "free_form"}
        except Exception as e:
            logging.error(f"Error in analyze_column_patterns: {str(e)}")
            return {"pattern_type": "error", "error": str(e)}

    @staticmethod
    def analyze_value_distribution(column: pd.Series) -> Dict:
        """Analyze value distribution statistics"""
        try:
            distribution_info = {
                "distinct_ratio": round(column.nunique() / len(column) if len(column) > 0 else 0, 4),
                "is_unique": column.is_unique,
                "constant": column.nunique() == 1 if len(column) > 0 else False,
                "has_duplicates": column.duplicated().any(),
            }
            
            if pd.api.types.is_numeric_dtype(column):
                clean_column = column.dropna()
                distribution_info.update({
                    "has_negatives": (clean_column < 0).any() if len(clean_column) > 0 else False,
                    "has_zeros": (clean_column == 0).any() if len(clean_column) > 0 else False,
                    "min": float(clean_column.min()) if len(clean_column) > 0 else None,
                    "max": float(clean_column.max()) if len(clean_column) > 0 else None,
                    "mean": float(clean_column.mean()) if len(clean_column) > 0 else None,
                    "median": float(clean_column.median()) if len(clean_column) > 0 else None,
                    "std": float(clean_column.std()) if len(clean_column) > 0 else None
                })
                
                if len(clean_column) >= 4:
                    try:
                        z_scores = np.abs(stats.zscore(clean_column))
                        distribution_info["has_outliers"] = bool(z_scores.max() > 3)
                        distribution_info["outlier_count"] = int((z_scores > 3).sum())
                    except Exception as e:
                        logging.warning(f"Could not calculate z-scores: {str(e)}")
                        distribution_info["has_outliers"] = None
            
            return distribution_info
        except Exception as e:
            logging.error(f"Error in analyze_value_distribution: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    def analyze_file(file_path: str | Path) -> Dict:
        """Analyze a CSV file with robust encoding detection and handling"""
        # List of encodings to try
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        last_error = None
        
        for encoding in encodings:
            try:
                logging.info(f"Attempting to read file with {encoding} encoding")
                df = pd.read_csv(file_path, encoding=encoding)
                
                structure = {
                    "metadata": {
                        "rows": len(df),
                        "columns": len(df.columns),
                        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                        "file_name": Path(file_path).name,
                        "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "encoding": encoding,
                        "total_missing_cells": int(df.isna().sum().sum()),
                        "total_missing_percentage": round(100 * df.isna().sum().sum() / (df.shape[0] * df.shape[1]), 2)
                    },
                    "columns": {}
                }
                
                for column in df.columns:
                    logging.info(f"Analyzing column: {column}")
                    col_data = df[column]
                    missing_count = col_data.isnull().sum()
                    
                    # Determine column type
                    if pd.api.types.is_datetime64_any_dtype(col_data):
                        col_type = "datetime"
                    elif pd.api.types.is_numeric_dtype(col_data):
                        if col_data.dtype in [np.int64, np.int32]:
                            col_type = "integer"
                        else:
                            col_type = "float"
                        if col_data.nunique() <= 10:
                            col_type = "categorical_numeric"
                    elif pd.api.types.is_bool_dtype(col_data):
                        col_type = "boolean"
                    else:
                        col_type = "categorical" if col_data.nunique() <= 20 else "text"
                    
                    # Get semantic type
                    semantic_type = DataAnalyzer.detect_semantic_type(column, col_data)
                    
                    # Build column analysis
                    col_analysis = {
                        "technical_type": {
                            "python_dtype": str(col_data.dtype),
                            "general_type": col_type,
                            "nullable": col_data.hasnans,
                        },
                        "semantic_type": semantic_type,
                        "constraints": {
                            "unique_values": int(col_data.nunique()),
                            "missing_count": int(missing_count),
                            "missing_percentage": round(100 * missing_count / len(df), 2),
                            "is_primary_key_candidate": col_data.is_unique and not col_data.isnull().any()
                        },
                        "patterns": DataAnalyzer.analyze_column_patterns(col_data),
                        "distribution": DataAnalyzer.analyze_value_distribution(col_data)
                    }
                    
                    # Only include sample values for non-PII columns
                    if semantic_type not in DataAnalyzer.PII_SEMANTIC_TYPES:
                        col_analysis["sample_values"] = col_data.head().tolist() if len(col_data) > 0 else []
                    
                    structure["columns"][column] = col_analysis
                    
                    # Add relationship information for potential primary keys
                    if col_analysis["constraints"]["is_primary_key_candidate"]:
                        structure["columns"][column]["relationships"] = {
                            "is_primary_key_candidate": True,
                            "potential_foreign_key": False
                        }
                
                logging.info(f"File analysis completed successfully using {encoding} encoding")
                return structure
                
            except UnicodeDecodeError as e:
                last_error = e
                logging.warning(f"Failed to read with {encoding} encoding: {str(e)}")
                continue
            except Exception as e:
                logging.error(f"Error analyzing file: {str(e)}")
                raise
        
        # If we've tried all encodings and none worked, raise the last error
        error_msg = f"Unable to read the CSV file with any of the supported encodings ({', '.join(encodings)}). Last error: {str(last_error)}"
        logging.error(error_msg)
        raise ValueError(error_msg)