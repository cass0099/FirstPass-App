from typing import Dict, Optional, Any
import logging
from anthropic import Anthropic
from config import ALLOWED_PACKAGES
import json
from pathlib import Path
from datetime import datetime, date
import time
import re
import random
import httpx
from openai import OpenAI
import numpy as np  # Make sure this is added for numpy type handling

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle special data types"""
    def default(self, obj):
        if isinstance(obj, bool):
            return str(obj).lower()  # Convert boolean to string
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, np.bool_):
            return str(obj).lower()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)  # Convert any other unknown types to string

class PromptLogger:
    """Dedicated logger for LLM prompts and responses"""
    
    def __init__(self, storage_manager):
        self.storage_manager = storage_manager
        self.logger = logging.getLogger(__name__)
        
    def _format_response(self, response: str) -> str:
        """Format the response code with line breaks and indentation"""
        try:
            lines = response.strip().split('\n')
            return '\n'.join(lines)
        except:
            return response
            
    def _format_metadata(self, metadata: Dict) -> Dict:
        """Format metadata for better readability"""
        if not metadata:
            return {}
            
        formatted = {}
        try:
            for key, value in metadata.items():
                if isinstance(value, dict):
                    formatted[key] = self._format_metadata(value)
                elif isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                    try:
                        cleaned = value.replace("'", '"').replace("np.float64(", "").replace("np.True_", "true").replace(")", "")
                        parsed = json.loads(cleaned)
                        formatted[key] = json.dumps(parsed, indent=2, cls=CustomJSONEncoder)
                    except:
                        formatted[key] = str(value)
                else:
                    formatted[key] = value
            return formatted
        except:
            return metadata

    def log_interaction(self, 
                       prompt: str,
                       response: str,
                       provider: str,
                       model: str,
                       metadata: Optional[Dict] = None,
                       rag_enabled: bool = False,
                       enhanced_prompt: Optional[str] = None,
                       base_prompt: Optional[str] = None) -> None:
        """Log a single prompt-response interaction with improved formatting."""
        try:
            # Create log entry with formatted sections
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "provider": str(provider),
                "model": str(model),
                "rag_enabled": str(rag_enabled).lower(),
                
                "user_prompt": f"""
=== USER PROMPT ===
{prompt.strip()}
""",
                "base_prompt": f"""
=== BASE PROMPT ===
{base_prompt.strip() if base_prompt else 'Not provided'}
"""
            }
            
            # Only add enhanced prompt if RAG was enabled and provided something
            if rag_enabled and enhanced_prompt and enhanced_prompt.strip():
                log_entry["enhanced_prompt"] = f"""
=== ENHANCED PROMPT ===
{enhanced_prompt.strip()}
"""
            
            # Add metadata in its own section
            if metadata:
                try:
                    formatted_metadata = self._format_metadata(metadata)
                    metadata_str = json.dumps(formatted_metadata, indent=2, cls=CustomJSONEncoder)
                    log_entry["metadata"] = f"""
=== METADATA ===
{metadata_str}
"""
                except Exception as e:
                    self.logger.warning(f"Error formatting metadata: {str(e)}")
                    log_entry["metadata"] = str(metadata)
            
            # Add LLM response
            log_entry["response"] = f"""
=== LLM RESPONSE ===
{self._format_response(response)}
"""
            
            # Save the formatted log
            self.storage_manager.save_prompt_log(log_entry)
            self.logger.info("Successfully logged prompt interaction")
                
        except Exception as e:
            self.logger.error(f"Error logging prompt interaction: {str(e)}", exc_info=True)

class LLMHandler:
    """Handles LLM interactions with clear package restrictions and RAG integration"""
    
    def __init__(self, storage_manager):
        self.anthropic_client = None
        self.openai_client = None
        self.error_history = []
        self.logger = logging.getLogger(__name__)
        self.max_retries = 3
        self.base_delay = 2  # Base delay in seconds
        self.storage_manager = storage_manager
        self.prompt_logger = PromptLogger(storage_manager)

    def _get_error_context(self) -> str:
        """Get formatted error context from history"""
        return "\n".join(self.error_history) if self.error_history else ""

    def initialize(self, provider: str, api_key: str):
        """Initialize LLM client based on provider"""
        if provider == "Anthropic":
            self.anthropic_client = Anthropic(api_key=api_key)
        else:  # OpenAI
            self.openai_client = OpenAI(api_key=api_key)

    def _filter_metadata(self, metadata: Dict, include_stats: bool = True, include_samples: bool = True) -> Dict:
        """Filter metadata based on settings"""
        # Ensure metadata is a dictionary
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid metadata format: {str(e)}")

        if not isinstance(metadata, dict):
            raise ValueError(f"Expected dict but got {type(metadata)}")

        # Create a deep copy for filtering
        filtered_metadata = json.loads(json.dumps(metadata, cls=CustomJSONEncoder))

        # Filter metadata based on settings
        if 'columns' in filtered_metadata:
            for col_name, col_data in filtered_metadata['columns'].items():
                # Remove sample values if disabled
                if not include_samples and 'sample_values' in col_data:
                    del col_data['sample_values']
                    
                # Remove distribution stats if disabled
                if not include_stats and 'distribution' in col_data:
                    # Keep only basic distribution info, remove statistical metrics
                    basic_dist = {
                        'distinct_ratio': col_data['distribution'].get('distinct_ratio'),
                        'is_unique': col_data['distribution'].get('is_unique'),
                        'constant': col_data['distribution'].get('constant'),
                        'has_duplicates': col_data['distribution'].get('has_duplicates')
                    }
                    col_data['distribution'] = basic_dist

        return filtered_metadata

    def _get_base_prompt(self, user_prompt: str, metadata: Dict, model_id: str, error_context: str = "") -> str:
        metadata_str = json.dumps(metadata, indent=2, cls=CustomJSONEncoder)
        
        prompt = f"""Given the following CSV metadata:
{metadata_str}

User request: {user_prompt}

CRITICAL REQUIREMENTS:
1. ONLY use the following allowed packages (DO NOT use any other packages):
{sorted(ALLOWED_PACKAGES)}

2. The CSV file path will be provided as the variable 'csv_path'. ALWAYS use pandas (not numpy) to read the CSV file:
Example:
```python
# Use pandas to read CSV, NOT numpy
df = pd.read_csv(csv_path)
```

3. Include proper error handling for all operations
4. Add clear comments explaining the code
5. Use pandas efficiently:
   - Use proper data types
   - Handle missing values appropriately
   - Include error handling for data operations
   - Use pandas.read_csv() NOT numpy.read_csv()
   - Use appropriate visualizations when needed
   - Add clear labels and titles to plots

ADDITIONAL FORMAT REQUIREMENTS:
1. All code must be properly indented using 4 spaces (not tabs)
2. All function definitions must have an indented body
3. All if/for/try blocks must be properly indented
4. Do not include any path assignments (csv_path will be provided)
5. Do not include commented-out code or function calls
6. Include error handling using try/except blocks
7. Ensure all parentheses are properly balanced
8. End the script with a single function call using the csv_path variable

Your code MUST follow this EXACT structure:
```python
import pandas as pd
import matplotlib.pyplot as plt

def main(csv_path):
    try:
        df = pd.read_csv(csv_path)
        
        # Your analysis code here
        
    except Exception as e:
        print(f"Error: {{str(e)}}")

main(csv_path)
```

DO NOT:
- Use any other function name besides 'main'
- Add any example paths or commented code
- Modify the csv_path variable
- Add explanations before or after the code"""

        if error_context:
            prompt += f"\n\nPrevious errors to fix:\n{error_context}"
        
        return prompt

    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = self.base_delay * (2 ** attempt)
        jitter = random.uniform(0, 0.1 * delay)  # 10% jitter
        return delay + jitter

    def _handle_api_error(self, e: Exception, attempt: int) -> Optional[Dict[str, Any]]:
        """Handle API errors with appropriate logging and retry logic"""
        if isinstance(e, httpx.HTTPStatusError):
            status_code = e.response.status_code
            error_detail = e.response.json() if e.response.content else {}
            
            # Handle specific status codes
            if status_code == 529:  # Server overloaded
                delay = self._exponential_backoff(attempt)
                self.logger.warning(
                    f"API overloaded (attempt {attempt + 1}/{self.max_retries}). "
                    f"Retrying in {delay:.1f} seconds..."
                )
                time.sleep(delay)
                return None  # Signal to retry
                
            elif status_code == 429:  # Rate limit
                delay = float(e.response.headers.get('retry-after', self._exponential_backoff(attempt)))
                self.logger.warning(
                    f"Rate limit exceeded (attempt {attempt + 1}/{self.max_retries}). "
                    f"Retrying in {delay:.1f} seconds..."
                )
                time.sleep(delay)
                return None  # Signal to retry
                
            else:  # Other HTTP errors
                error_msg = f"HTTP {status_code}: {error_detail}"
                self.logger.error(f"API error: {error_msg}")
                raise ValueError(f"API error: {error_msg}")
                
        # Handle non-HTTP errors
        self.logger.error(f"Unexpected error: {str(e)}")
        raise e

    def _get_token_estimate(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: ~4 characters per token
        return len(text) // 4

    def _extract_code(self, response: str) -> str:
        """Extract Python code from response"""
        # Look for code between triple backticks
        import re
        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Fallback: try to find any code between triple backticks
        code_match = re.search(r'```(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # If no code blocks found, raise error
        raise ValueError("No code block found in response")

    def add_error_to_history(self, error):
        if isinstance(error, ScriptError) and error.is_script_error:
            formatted_error = f"Script Error: {error.message}"
            if error.traceback:
                formatted_error += f"\n{error.traceback}"
            self.error_history.append(formatted_error)

    def generate_script(self, prompt: str, metadata: Dict, model_id: str, rag_assistant=None, include_stats: bool = True, include_samples: bool = True) -> str:
        # Extract provider and model name from model_id
        provider, model = model_id.split("://", 1)
        
        # Initialize correct client if not already done
        if provider == "anthropic" and not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")
        if provider == "openai" and not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        try:
            # Create filtered metadata first
            filtered_metadata = self._filter_metadata(
                metadata,
                include_stats=include_stats,
                include_samples=include_samples
            )

            # Generate base prompt with filtered metadata and provider-specific formatting
            base_prompt = self._get_base_prompt(
                prompt, 
                filtered_metadata,
                self._get_error_context()
            )
            full_prompt = base_prompt
            
            # Add RAG context if enabled
            rag_enabled = rag_assistant and hasattr(rag_assistant, 'enabled') and rag_assistant.enabled
            rag_context = None
            if rag_enabled:
                try:
                    rag_context = rag_assistant.enhance_prompt(prompt, filtered_metadata)
                    if rag_context and rag_context != prompt:
                        full_prompt = f"{rag_context}\n\n{base_prompt}"
                except Exception as e:
                    self.logger.error(f"Error getting RAG context: {str(e)}")
            
            # Generate response based on provider
            for attempt in range(self.max_retries):
                try:
                    if provider == "anthropic":
                        response = self.anthropic_client.messages.create(
                            model=model,
                            messages=[{"role": "user", "content": full_prompt}],
                            max_tokens=1500
                        )
                        raw_response = response.content[0].text
                    else:  # openai
                        response = self.openai_client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": full_prompt}],
                            max_tokens=1500
                        )
                        raw_response = response.choices[0].message.content

                    # Extract code from response
                    generated_code = self._extract_code(raw_response)

                    # Log the successful interaction
                    self.prompt_logger.log_interaction(
                        prompt=prompt,
                        response=raw_response,
                        provider=provider,
                        model=model,
                        metadata=filtered_metadata,
                        rag_enabled=rag_enabled,
                        enhanced_prompt=rag_context,
                        base_prompt=base_prompt
                    )

                    return generated_code

                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    retry_result = self._handle_api_error(e, attempt)
                    if retry_result is not None:
                        raise

        except Exception as e:
            error_msg = f"Error generating script with {provider} model {model}: {str(e)}"
            self.logger.error(error_msg)
            if "Script Error:" in str(e):
                self.add_error_to_history(str(e))
            raise ValueError(error_msg)

    def clear_error_history(self):
        """Clear the error history"""
        self.error_history.clear()
        self.logger.info("Error history cleared")

    def get_error_history(self) -> list:
        """Get current error history"""
        return self.error_history.copy()