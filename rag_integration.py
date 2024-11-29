from pathlib import Path
import logging
from typing import Optional, Dict, List, Tuple, Any, Union
import json
import random
from datetime import datetime, date
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss
import sqlite3
import shutil
from dataclasses import dataclass
from enum import Enum
from analyzer import EnhancedJSONEncoder

class EnhancedJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle special data types"""
    def default(self, obj):
        if isinstance(obj, bool):
            return int(obj)  # Convert boolean to 0 or 1
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

class KnowledgeType(Enum):
    """Types of knowledge entries"""
    COLUMN = "column"
    BUSINESS = "business"
    PATTERN = "pattern"

@dataclass
class TokenBudget:
    """Token allocation for different context types"""
    column_dictionary: int
    business_rules: int
    analysis_patterns: int
    general_context: int
    
    @classmethod
    def from_remaining(cls, remaining_tokens: int):
        """Create budget from remaining tokens"""
        return cls(
            column_dictionary=int(remaining_tokens * 0.3),
            business_rules=int(remaining_tokens * 0.3),
            analysis_patterns=int(remaining_tokens * 0.3),
            general_context=int(remaining_tokens * 0.1)
        )

class RAGAssistant:
    """Enhanced RAG assistant with comprehensive knowledge management"""
    
    def __init__(self, storage_manager=None, user_knowledge_dir: Optional[Path] = None, enabled: bool = True):
        """Initialize RAG assistant with storage manager"""
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.debug("RAG Assistant logger initialized")
        
        # Set basic configuration
        self.enabled = enabled
        self.max_tokens = 4000
        
        if storage_manager:
            # Use storage manager paths
            self.base_knowledge_dir = storage_manager.data_dir / "base_knowledge"
            self.user_knowledge_dir = storage_manager.data_dir / "user_knowledge"
            
            # Create directories
            self.base_knowledge_dir.mkdir(parents=True, exist_ok=True)
            self.user_knowledge_dir.mkdir(parents=True, exist_ok=True)
            
            self.base_db_path = self.base_knowledge_dir / "base_knowledge.db"
            self.base_index_path = self.base_knowledge_dir / "base_vectors.index"
            self.user_db_path = self.user_knowledge_dir / "user_knowledge.db"
            self.user_index_path = self.user_knowledge_dir / "user_vectors.index"
        else:
            # Use provided paths or defaults
            self.base_knowledge_dir = Path("base_knowledge")
            self.user_knowledge_dir = user_knowledge_dir if user_knowledge_dir else Path("user_knowledge")
            
            self.base_db_path = self.base_knowledge_dir / "base_knowledge.db"
            self.base_index_path = self.base_knowledge_dir / "base_vectors.index" 
            self.user_db_path = self.user_knowledge_dir / "user_knowledge.db"
            self.user_index_path = self.user_knowledge_dir / "user_vectors.index"
        
        # Add debug checks for database status
        self.logger.debug("\n=== Checking Database Status ===")
        
        # Check if base knowledge exists
        self.logger.debug(f"Base DB path: {self.base_db_path}")
        self.logger.debug(f"Base DB exists: {self.base_db_path.exists()}")
        
        if self.base_db_path.exists():
            try:
                with sqlite3.connect(self.base_db_path) as conn:
                    # Check tables
                    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [table[0] for table in cursor.fetchall()]
                    self.logger.debug(f"Tables in base DB: {tables}")
                    
                    # Check pattern count if table exists
                    if 'analysis_patterns' in tables:
                        cursor = conn.execute("SELECT COUNT(*) FROM analysis_patterns")
                        count = cursor.fetchone()[0]
                        self.logger.debug(f"Number of patterns: {count}")
                    else:
                        self.logger.warning("analysis_patterns table not found in base DB")
            except Exception as e:
                self.logger.error(f"Error checking base DB: {str(e)}")
        
        # Initialize embedding model
        self.logger.debug("Initializing sentence transformer model")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load indices
        self.logger.debug("Loading FAISS indices")
        self.base_index = self._load_index(self.base_index_path)
        self.user_index = None
        
        if self.user_knowledge_dir:
            self.user_index = self._load_index(self.user_index_path)
            self._initialize_user_database()
            self._initialize_system_tags()
            
        if self.enabled:
            self.logger.info(f"RAG Assistant initialized with base knowledge: {self.base_knowledge_dir}")
            if self.user_knowledge_dir:
                self.logger.info(f"User knowledge directory: {self.user_knowledge_dir}")
        else:
            self.logger.warning("RAG Assistant initialized but disabled")

    def _count_base_knowledge(self) -> Dict[str, int]:
        """Count items in base knowledge"""
        try:
            counts = {}
            if self.base_db_path.exists():
                with sqlite3.connect(self.base_db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM analysis_patterns")
                    counts['patterns'] = cursor.fetchone()[0]
            return counts
        except Exception as e:
            self.logger.error(f"Error counting base knowledge: {str(e)}")
            return {'error': str(e)}

    def _count_user_knowledge(self) -> Dict[str, int]:
        """Count items in user knowledge"""
        try:
            counts = {}
            if self.user_db_path and Path(self.user_db_path).exists():
                with sqlite3.connect(self.user_db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM column_dictionary")
                    counts['columns'] = cursor.fetchone()[0]
                    
                    cursor = conn.execute("SELECT COUNT(*) FROM business_knowledge")
                    counts['business'] = cursor.fetchone()[0]
                    
                    cursor = conn.execute("SELECT COUNT(*) FROM custom_patterns")
                    counts['patterns'] = cursor.fetchone()[0]
            return counts
        except Exception as e:
            self.logger.error(f"Error counting user knowledge: {str(e)}")
            return {'error': str(e)}

    def _initialize_user_database(self):
        """Initialize user knowledge database schema"""
        try:
            with sqlite3.connect(self.user_db_path) as conn:
                conn.executescript("""
                    -- Existing patterns table
                    CREATE TABLE IF NOT EXISTS custom_patterns (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        code TEXT NOT NULL,
                        tags TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        vector_id INTEGER NOT NULL
                    );
                    
                    -- Column Dictionary
                    CREATE TABLE IF NOT EXISTS column_dictionary (
                        id INTEGER PRIMARY KEY,
                        column_pattern TEXT NOT NULL,
                        description TEXT NOT NULL,
                        data_type TEXT,
                        examples TEXT,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        vector_id INTEGER
                    );
                    
                    -- System Tags
                    CREATE TABLE IF NOT EXISTS system_tags (
                        id INTEGER PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        category TEXT NOT NULL,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- Business Knowledge
                    CREATE TABLE IF NOT EXISTS business_knowledge (
                        id INTEGER PRIMARY KEY,
                        text TEXT NOT NULL,
                        summary TEXT,
                        system_tags TEXT NOT NULL,
                        custom_tags TEXT,
                        related_columns TEXT,
                        priority INTEGER DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        vector_id INTEGER
                    );
                    
                    -- Knowledge embeddings
                    CREATE TABLE IF NOT EXISTS knowledge_embeddings (
                        id INTEGER PRIMARY KEY,
                        knowledge_id INTEGER NOT NULL,
                        knowledge_type TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(knowledge_id, knowledge_type)
                    );
                """)
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error initializing user database: {str(e)}")
            raise

    def _initialize_system_tags(self):
        """Initialize default system tags"""
        default_tags = {
            'business_domain': [
                'finance', 'sales', 'marketing', 'product', 'support', 
                'operations', 'legal', 'hr'
            ],
            'data_type': [
                'metrics', 'dimensions', 'timestamps', 'categories',
                'identifiers', 'monetary', 'percentage'
            ],
            'time_relevance': [
                'historical', 'current', 'deprecated', 'future'
            ],
            'knowledge_type': [
                'business_rule', 'data_quality', 'calculation', 'context',
                'limitation', 'assumption'
            ]
        }
        
        try:
            with sqlite3.connect(self.user_db_path) as conn:
                for category, tags in default_tags.items():
                    for tag in tags:
                        conn.execute("""
                            INSERT OR IGNORE INTO system_tags (name, category, description)
                            VALUES (?, ?, ?)
                        """, (tag, category, f"Default {category} tag: {tag}"))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error initializing system tags: {str(e)}")
    
    def _load_index(self, index_path: Path) -> Optional[faiss.Index]:
        """Load FAISS index if it exists"""
        try:
            if index_path.exists():
                return faiss.read_index(str(index_path))
            
            # Create new index if it doesn't exist
            dimension = self.model.get_sentence_embedding_dimension()
            index = faiss.IndexFlatL2(dimension)
            faiss.write_index(index, str(index_path))
            return index
            
        except Exception as e:
            self.logger.error(f"Error loading index {index_path}: {str(e)}")
            return None

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: ~4 characters per token
        return len(text) // 4

    def _allocate_token_budget(self, base_tokens: int) -> TokenBudget:
        """Allocate remaining tokens to different context types"""
        remaining = self.max_tokens - base_tokens
        if remaining <= 0:
            return TokenBudget(0, 0, 0, 0)
        return TokenBudget.from_remaining(remaining)

    def _update_embeddings(self, knowledge_id: int, text: str, knowledge_type: KnowledgeType):
        """Update embeddings for knowledge entries with improved locking handling"""
        import time
        max_retries = 5  # Increased from 3
        base_delay = 0.5  # Initial delay in seconds
        
        for attempt in range(max_retries):
            conn = None
            try:
                # Generate embedding first to minimize time with lock
                embedding = self.model.encode([text])[0]
                
                # Use a longer timeout and immediate mode for better concurrency
                conn = sqlite3.connect(
                    self.user_db_path,
                    timeout=60.0,  # Increased timeout
                    isolation_level='IMMEDIATE'  # More aggressive locking
                )
                
                # Execute all operations in a single transaction
                with conn:  # This automatically handles commit/rollback
                    # Store in embeddings table
                    conn.execute("""
                        INSERT OR REPLACE INTO knowledge_embeddings
                        (knowledge_id, knowledge_type, embedding)
                        VALUES (?, ?, ?)
                    """, (knowledge_id, knowledge_type.value, embedding.tobytes()))
                    
                    # Update vector_id in respective table
                    if knowledge_type == KnowledgeType.COLUMN:
                        table = "column_dictionary"
                    elif knowledge_type == KnowledgeType.BUSINESS:
                        table = "business_knowledge"
                    else:
                        table = "custom_patterns"
                        
                    conn.execute(f"""
                        UPDATE {table}
                        SET vector_id = ?
                        WHERE id = ?
                    """, (self.user_index.ntotal if self.user_index else 0, knowledge_id))
                
                # Add to FAISS index outside the transaction
                if self.user_index is not None:
                    self.user_index.add(embedding.reshape(1, -1).astype('float32'))
                    
                return  # Success - exit the retry loop
                    
            except sqlite3.OperationalError as e:
                if conn:
                    try:
                        conn.rollback()
                    except:
                        pass
                        
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    self.logger.warning(f"Database locked, retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue
                raise  # Re-raise if we're out of retries or it's a different error
                
            except Exception as e:
                if conn:
                    try:
                        conn.rollback()
                    except:
                        pass
                self.logger.error(f"Error updating embeddings: {str(e)}")
                raise
                
            finally:
                if conn:
                    try:
                        conn.close()
                    except:
                        pass

        raise Exception("Failed to update embeddings after multiple retries")

    def _score_knowledge_relevance(self,
                                knowledge_entry: Dict,
                                prompt: str,
                                metadata: Dict) -> float:
        """Score knowledge entry relevance with improved exact match scoring"""
        score = 0.0
        prompt_lower = prompt.lower()
        
        # Extract text content to check
        text = knowledge_entry.get('text', '').lower()
        summary = knowledge_entry.get('summary', '').lower()
        description = knowledge_entry.get('description', '').lower()
        
        # 1. Keyword presence in prompt (weight: 1.0)
        significant_words = set()
        for content in [text, summary, description]:
            words = [w for w in content.split() if len(w) > 3]
            significant_words.update(words)
        
        matching_words = sum(1 for word in significant_words if word in prompt_lower)
        word_score = matching_words / max(len(significant_words), 1)
        score += word_score * 1.0
        
        # 2. Column matching with higher weight for exact matches (weight: 5.0)
        if isinstance(metadata.get('columns'), list):
            column_pattern = knowledge_entry.get('column_pattern', '').lower()
            matched_columns = 0
            exact_matches = 0
            
            for col in metadata['columns']:
                col_lower = col.lower()
                if col_lower == column_pattern:
                    exact_matches += 1
                    matched_columns += 1
                elif ('*' in column_pattern and 
                    re.match(f"^{column_pattern.replace('*', '.*')}$", col_lower)):
                    matched_columns += 0.8  # High score for wildcard matches
                elif (re.sub(r'[^a-z0-9]', '', column_pattern) == 
                    re.sub(r'[^a-z0-9]', '', col_lower)):
                    matched_columns += 0.6  # Good score for normalized matches
            
            # Boost score significantly for exact matches
            if exact_matches > 0:
                score += 5.0  # Base boost for exact match
            
            # Add weighted column match score
            column_score = matched_columns / max(len(metadata['columns']), 1)
            score += column_score * 3.0
        
        # 3. Tag matching (weight: 1.5)
        system_tags = json.loads(knowledge_entry.get('system_tags', '[]'))
        custom_tags = json.loads(knowledge_entry.get('custom_tags', '[]'))
        
        tag_score = 0.0
        for tag in system_tags + custom_tags:
            if tag.lower() in prompt_lower:
                tag_score += 1.0
        
        if system_tags or custom_tags:
            tag_score = tag_score / max(len(system_tags) + len(custom_tags), 1)
            score += tag_score * 1.5
        
        # 4. Priority boost (weight: 1.0 per level)
        priority = knowledge_entry.get('priority', 1)
        score += (priority - 1) * 1.0
        
        # 5. Recency boost (max 0.5)
        created_at = datetime.fromisoformat(knowledge_entry['created_at'])
        days_old = (datetime.now() - created_at).days
        if days_old < 30:
            score += (30 - days_old) / 30 * 0.5
        
        return score

    def _select_context(self,
                    entries: List[Dict],
                    token_budget: int,
                    min_score: float = 1.0) -> List[Dict]:
        """Select most relevant context within token budget"""
        selected = []
        tokens_used = 0
        
        # Filter by minimum score and sort
        relevant_entries = [
            entry for entry in entries 
            if entry.get('relevance_score', 0) >= min_score
        ]
        sorted_entries = sorted(
            relevant_entries,
            key=lambda x: (
                x.get('priority', 1),
                x.get('relevance_score', 0),
                x['created_at']
            ),
            reverse=True
        )
        
        print(f"Selected {len(relevant_entries)} entries above minimum score {min_score}")
        
        for entry in sorted_entries:
            # For column definitions, use description
            if 'description' in entry:
                text = entry['description']
            # For business knowledge, use summary if available for tight budgets
            elif tokens_used > token_budget * 0.7 and entry.get('summary'):
                text = entry['summary']
            # Otherwise use full text
            elif 'text' in entry:
                text = entry['text']
            else:
                print(f"Warning: Entry has no usable text content: {entry}")
                continue
                
            entry_tokens = self._estimate_tokens(text)
            
            if tokens_used + entry_tokens > token_budget:
                print(f"Token budget {token_budget} would be exceeded by {tokens_used + entry_tokens} tokens, stopping")
                break
                
            print(f"Adding entry using {entry_tokens} tokens (budget: {tokens_used}/{token_budget})")
            entry['selected_text'] = text
            selected.append(entry)
            tokens_used += entry_tokens
        
        print(f"Selected {len(selected)} entries using {tokens_used} of {token_budget} tokens")
        return selected

    def _get_column_context(self, columns: List[str], prompt: str, token_budget: int) -> str:
        """Get relevant column definitions with improved exact and pattern matching"""
        try:
            print(f"\nAttempting to get column context for columns: {columns}")
            context_entries = []
            
            with sqlite3.connect(self.user_db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Debug: Show all patterns in database
                cursor = conn.execute("SELECT column_pattern, description FROM column_dictionary")
                rows = cursor.fetchall()
                print("\nAvailable patterns in database:")
                for row in rows:
                    print(f"  - {row['column_pattern']}: {row['description']}")
                
                for column in columns:
                    print(f"\nLooking up matches for column: {column}")
                    # First try exact match
                    cursor = conn.execute("""
                        SELECT 
                            id, column_pattern, description, data_type,
                            examples, notes, created_at
                        FROM column_dictionary
                        WHERE lower(column_pattern) = lower(?)
                    """, (column,))
                    
                    matches = [dict(row) for row in cursor.fetchall()]
                    print(f"Found {len(matches)} exact matches")
                    
                    # If no exact match, try pattern matching
                    if not matches:
                        print("No exact matches, trying pattern matching...")
                        cursor = conn.execute("""
                            SELECT 
                                id, column_pattern, description, data_type,
                                examples, notes, created_at
                            FROM column_dictionary
                        """)
                        
                        pattern_matches = []
                        for row in cursor:
                            row_dict = dict(row)
                            pattern = row_dict['column_pattern'].lower()
                            col_lower = column.lower()
                            
                            # Print pattern comparison for debugging
                            print(f"Comparing column '{col_lower}' with pattern '{pattern}'")
                            
                            if (pattern == col_lower or
                                ('*' in pattern and re.match(f"^{pattern.replace('*', '.*')}$", col_lower)) or
                                ('%' in pattern and re.match(f"^{pattern.replace('%', '.*')}$", col_lower)) or
                                (re.sub(r'[^a-z0-9]', '', pattern) == re.sub(r'[^a-z0-9]', '', col_lower))):
                                print(f"Found pattern match: {pattern}")
                                pattern_matches.append(row_dict)
                        
                        matches = pattern_matches
                        print(f"Found {len(matches)} pattern matches")
                    
                    # Process matches for this column
                    for entry in matches:
                        entry['matched_column'] = column
                        relevance_score = self._score_knowledge_relevance(
                            entry, prompt, {'columns': [column]}
                        )
                        entry['relevance_score'] = relevance_score
                        context_entries.append(entry)
                        print(f"Added entry with score {relevance_score}: {entry['column_pattern']}")
            
            # Select entries within token budget
            selected = self._select_context(context_entries, token_budget)
            print(f"\nSelected {len(selected)} entries within token budget")
            
            if not selected:
                print("No entries selected - returning empty string")
                return ""
            
            # Build context string
            context = ""
            by_column = {}
            for entry in selected:
                col = entry['matched_column']
                if col not in by_column:
                    by_column[col] = []
                by_column[col].append(entry)
            
            # Format the output
            for col, entries in by_column.items():
                context += f"\n{col}:\n"
                for entry in entries:
                    context += f"- {entry['description']}\n"
                    if entry.get('notes'):
                        context += f"  Note: {entry['notes']}\n"
                    if entry.get('examples'):
                        examples = json.loads(entry['examples']) if isinstance(entry['examples'], str) else entry['examples']
                        if examples:
                            context += f"  Examples: {', '.join(map(str, examples))}\n"
            
            print(f"\nGenerated context:\n{context}")
            return context.strip()
                
        except Exception as e:
            print(f"Error in _get_column_context: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return ""

    def _get_business_context(self, prompt: str, metadata: Dict, token_budget: int) -> str:
        """Get relevant business knowledge with improved filtering"""
        try:
            self.logger.debug("Getting business context with metadata keys: " + 
                            f"{list(metadata.keys() if metadata else [])}")
            
            with sqlite3.connect(self.user_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT 
                        id, text, summary, system_tags, custom_tags,
                        related_columns, priority, created_at
                    FROM business_knowledge
                    ORDER BY priority DESC, created_at DESC
                """)
                
                entries = [dict(row) for row in cursor.fetchall()]
                self.logger.debug(f"Found {len(entries)} total business knowledge entries")
                
                if not entries:
                    return ""
                    
                context_entries = []
                for entry in entries:
                    try:
                        # Score relevance for each entry
                        relevance_score = self._score_knowledge_relevance(
                            entry, prompt, metadata
                        )
                        
                        # Only include entries with significant relevance
                        if relevance_score >= 2.0:  # Minimum threshold for inclusion
                            entry['relevance_score'] = relevance_score
                            context_entries.append(entry)
                            self.logger.debug(
                                f"Entry scored {relevance_score:.2f}: {entry['text'][:50]}..."
                            )
                        else:
                            self.logger.debug(
                                f"Entry below threshold ({relevance_score:.2f}): {entry['text'][:50]}..."
                            )
                            
                    except Exception as e:
                        self.logger.error(f"Error scoring entry {entry.get('id', 'unknown')}: {str(e)}")
                        continue
                
                # Sort by relevance score and priority
                context_entries.sort(
                    key=lambda x: (x['relevance_score'], x.get('priority', 1)),
                    reverse=True
                )
                
                self.logger.debug(
                    f"Selected {len(context_entries)} entries above relevance threshold"
                )
                
                # Select entries within token budget with minimum relevance
                selected = self._select_context(
                    context_entries,
                    token_budget,
                    min_score=2.0  # Minimum relevance score required
                )
                
                self.logger.debug(f"Selected {len(selected)} entries within token budget")
                
                if not selected:
                    return ""
                
                # Build context string grouped by priority
                grouped_entries = {}
                for entry in selected:
                    priority = entry.get('priority', 1)
                    if priority not in grouped_entries:
                        grouped_entries[priority] = []
                    grouped_entries[priority].append(entry)
                
                context_parts = ["Business Rules:"]
                
                # Add entries by priority group
                for priority in sorted(grouped_entries.keys(), reverse=True):
                    entries = grouped_entries[priority]
                    
                    if priority > 1:
                        context_parts.append(f"\nPriority {priority}:")
                        
                    for entry in entries:
                        text = entry.get('selected_text', entry.get('text', ''))
                        score = entry.get('relevance_score', 0)
                        context_parts.append(f"• [{score:.1f}] {text}")
                
                final_context = "\n".join(context_parts)
                self.logger.debug(f"Generated business context: {final_context}")
                return final_context
                    
        except Exception as e:
            self.logger.error(f"Error getting business context: {str(e)}")
            return ""

    def _get_relevant_patterns(self, prompt: str, metadata: Dict, top_k: int = 3) -> list:
        """Retrieve relevant patterns based on prompt and metadata with detailed logging"""
        if not self.enabled or not self.base_index:
            return []
                    
        try:
            self.logger.debug("\n" + "="*50)
            self.logger.debug("PATTERN MATCHING SEQUENCE")
            self.logger.debug("="*50)
            self.logger.debug(f"Input prompt (truncated): {prompt[:100]}...")
            
            pattern_sequence = {
                'base_pattern_count': 0,
                'user_pattern_count': 0,
                'matches': [],
                'metadata_keys': list(metadata.keys())
            }
            
            # Get prompt embedding
            prompt_embedding = self.model.encode([prompt])[0]
            patterns = []
            
            # Search base knowledge
            if self.base_index:
                self.logger.debug("\n=== Base Pattern Search ===")
                D, I = self.base_index.search(
                    prompt_embedding.reshape(1, -1).astype('float32'),
                    top_k
                )
                pattern_sequence['base_scores'] = D[0].tolist()
                
                with sqlite3.connect(self.base_db_path) as conn:
                    for idx, score in zip(I[0], D[0]):
                        self.logger.debug(f"\nProcessing base pattern {idx} (score: {score:.4f})")
                        row = conn.execute("""
                            SELECT 
                                pattern_name,
                                description,
                                analysis_steps,
                                tags
                            FROM analysis_patterns 
                            WHERE vector_id = ?
                        """, (int(idx),)).fetchone()
                        
                        if row:
                            tags = json.loads(row[3])
                            self.logger.debug(f"Found pattern: {row[0]}")
                            self.logger.debug(f"Tags: {tags}")
                            pattern_sequence['matches'].append({
                                'type': 'base',
                                'name': row[0],
                                'score': float(score),
                                'tags': tags
                            })
                            pattern_sequence['base_pattern_count'] += 1
                            
                            patterns.append({
                                'source': 'base',
                                'pattern_name': row[0],
                                'description': row[1],
                                'analysis_steps': json.loads(row[2]),
                                'tags': tags,
                                'similarity_score': float(score)
                            })
            
            # Search user knowledge
            if self.user_index:
                self.logger.debug("\n=== User Pattern Search ===")
                D_user, I_user = self.user_index.search(
                    prompt_embedding.reshape(1, -1).astype('float32'),
                    top_k
                )
                pattern_sequence['user_scores'] = D_user[0].tolist()
                
                with sqlite3.connect(self.user_db_path) as conn:
                    for idx, score in zip(I_user[0], D_user[0]):
                        self.logger.debug(f"\nProcessing user pattern {idx} (score: {score:.4f})")
                        row = conn.execute("""
                            SELECT 
                                name,
                                description,
                                prompt,
                                tags,
                                metadata
                            FROM custom_patterns 
                            WHERE vector_id = ?
                        """, (int(idx),)).fetchone()
                        
                        if row:
                            tags = json.loads(row[3])
                            metadata = json.loads(row[4]) if row[4] else {}
                            self.logger.debug(f"Found pattern: {row[0]}")
                            self.logger.debug(f"Tags: {tags}")
                            self.logger.debug(f"Original metadata: {metadata}")
                            
                            pattern_sequence['matches'].append({
                                'type': 'user',
                                'name': row[0],
                                'score': float(score),
                                'tags': tags,
                                'metadata': metadata
                            })
                            pattern_sequence['user_pattern_count'] += 1
                            
                            patterns.append({
                                'source': 'user',
                                'pattern_name': row[0],
                                'description': row[1],
                                'analysis_steps': ['Custom pattern based on: ' + row[2]],
                                'tags': tags,
                                'similarity_score': float(score)
                            })
            
            # Sort and log final results
            patterns.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            self.logger.debug("\n=== Pattern Matching Summary ===")
            self.logger.debug(f"""
    Results:
    - Base Patterns Found: {pattern_sequence['base_pattern_count']}
    - User Patterns Found: {pattern_sequence['user_pattern_count']}
    - Total Matches: {len(pattern_sequence['matches'])}

    Top Matches (sorted by score):
    {json.dumps([f"{m['type']}: {m['name']} ({m['score']:.4f})" for m in 
                sorted(pattern_sequence['matches'], key=lambda x: x['score'], reverse=True)], 
            indent=2)}
    """)
            self.logger.debug("="*50 + "\n")
            
            return patterns[:top_k]
                    
        except Exception as e:
            self.logger.error(f"Error in pattern matching: {str(e)}", exc_info=True)
            return []

    def enhance_prompt(self, base_prompt: str, metadata: Dict) -> str:
        """Enhanced prompt generation with strategic context management and detailed logging"""
        print("\n" + "="*50)
        print("Starting prompt enhancement")
        print(f"RAG enabled: {self.enabled}")
        print(f"Base prompt: {base_prompt}")
        print(f"Metadata keys: {list(metadata.keys() if metadata else [])}")
        print("="*50)

        if not self.enabled:
            print("RAG enhancement disabled - returning base prompt")
            return base_prompt
                    
        try:
            print("Beginning prompt enhancement with RAG")
            
            # Test database connection and content
            if self.user_db_path.exists():
                print(f"Database exists at: {self.user_db_path}")
                with sqlite3.connect(self.user_db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM column_dictionary")
                    count = cursor.fetchone()[0]
                    print(f"Found {count} column definitions in database")
                    
                    if count > 0:
                        cursor = conn.execute("""
                            SELECT column_pattern, description 
                            FROM column_dictionary 
                            LIMIT 3
                        """)
                        samples = cursor.fetchall()
                        print("Sample definitions:")
                        for sample in samples:
                            print(f"  - {sample[0]}: {sample[1]}")
            
            # Calculate base prompt tokens
            base_tokens = self._estimate_tokens(base_prompt)
            token_budget = self._allocate_token_budget(base_tokens)
            
            print(f"""
    Token Budget Allocation:
    - Base prompt: {base_tokens} tokens
    - Column dictionary: {token_budget.column_dictionary} tokens
    - Business rules: {token_budget.business_rules} tokens
    - Analysis patterns: {token_budget.analysis_patterns} tokens
    - General context: {token_budget.general_context} tokens
    """)
            
            # Get relevant knowledge
            column_keys = []
            if isinstance(metadata.get('columns'), dict):
                column_keys = list(metadata['columns'].keys())
                print(f"Extracted column keys from metadata: {column_keys}")
            else:
                print(f"Could not extract column keys. Metadata 'columns' type: {type(metadata.get('columns'))}")
            
            # Get column context
            print("\nGetting column context...")
            column_context = self._get_column_context(
                column_keys,
                base_prompt,
                token_budget.column_dictionary
            )
            
            if column_context:
                print("\nGenerated column context:")
                print(column_context)
            else:
                print("No column context was generated")
            
            # Get business context
            print("\nGetting business context...")
            business_context = self._get_business_context(
                base_prompt,
                metadata,
                token_budget.business_rules
            )
            
            # Get analysis patterns
            print("\nGetting analysis patterns...")
            patterns = self._get_relevant_patterns(base_prompt, metadata)
            pattern_text = ""
            if patterns:
                pattern_text = "\n\nRelevant Analysis Patterns:\n"
                for pattern in patterns:
                    pattern_text += f"\n{pattern['pattern_name']} ({pattern['source']}):\n"
                    pattern_text += f"- Description: {pattern['description']}\n"
                    pattern_text += "- Steps:\n"
                    for step in pattern['analysis_steps']:
                        pattern_text += f"  * {step}\n"
            
            # Build enhanced prompt
            enhanced_prompt = base_prompt
            
            if column_context:
                enhanced_prompt += "\n\nColumn Definitions:\n" + column_context
                print("\nAdded column context to prompt")
            
            if business_context:
                enhanced_prompt += "\n\nBusiness Context:\n" + business_context
                print("Added business context to prompt")
            
            if pattern_text:
                enhanced_prompt += pattern_text
                print(f"Added {len(patterns)} patterns to prompt")
            
            # Compare prompts
            original_tokens = self._estimate_tokens(base_prompt)
            enhanced_tokens = self._estimate_tokens(enhanced_prompt)
            print(f"""
    Prompt Enhancement Summary:
    - Original tokens: {original_tokens}
    - Enhanced tokens: {enhanced_tokens}
    - Added content: {enhanced_tokens - original_tokens} tokens
    """)
            
            return enhanced_prompt
                    
        except Exception as e:
            print(f"Error enhancing prompt: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return base_prompt

    def add_column_definition(self, 
                            column_pattern: str,
                            description: str,
                            data_type: Optional[str] = None,
                            examples: Optional[List[str]] = None,
                            notes: Optional[str] = None) -> int:
        """Add or update column definition - simplified version"""
        try:
            # Generate embedding first
            text_to_embed = f"{column_pattern} {description} {notes or ''}"
            embedding = self.model.encode([text_to_embed])[0]
            
            # Single database connection for all operations
            with sqlite3.connect(self.user_db_path) as conn:
                # Insert column definition
                cursor = conn.execute("""
                    INSERT INTO column_dictionary
                    (column_pattern, description, data_type, examples, notes)
                    VALUES (?, ?, ?, ?, ?)
                    RETURNING id
                """, (
                    column_pattern,
                    description,
                    data_type,
                    json.dumps(examples) if examples else None,
                    notes
                ))
                
                knowledge_id = cursor.fetchone()[0]
                
                # Store embedding
                conn.execute("""
                    INSERT INTO knowledge_embeddings
                    (knowledge_id, knowledge_type, embedding)
                    VALUES (?, ?, ?)
                """, (knowledge_id, KnowledgeType.COLUMN.value, embedding.tobytes()))
                
                # Update FAISS index if it exists
                if self.user_index is not None:
                    self.user_index.add(embedding.reshape(1, -1).astype('float32'))
                
                conn.commit()
                return knowledge_id
                
        except Exception as e:
            self.logger.error(f"Error adding column definition: {str(e)}")
            raise

    def add_business_knowledge(self,
                            text: str,
                            system_tags: List[str],
                            custom_tags: Optional[List[str]] = None,
                            related_columns: Optional[List[str]] = None,
                            priority: int = 1,
                            summary: Optional[str] = None) -> int:
        """Add business knowledge entry - simplified version"""
        try:
            # Generate embedding first
            text_to_embed = f"{text} {' '.join(system_tags)} {' '.join(custom_tags or [])}"
            embedding = self.model.encode([text_to_embed])[0]
            
            # Single database connection for all operations
            with sqlite3.connect(self.user_db_path) as conn:
                # Validate system tags
                valid_tags = set(row[0] for row in conn.execute(
                    "SELECT name FROM system_tags"
                ).fetchall())
                
                invalid_tags = set(system_tags) - valid_tags
                if invalid_tags:
                    raise ValueError(f"Invalid system tags: {invalid_tags}")
                
                # Insert business knowledge
                cursor = conn.execute("""
                    INSERT INTO business_knowledge
                    (text, summary, system_tags, custom_tags, related_columns, priority)
                    VALUES (?, ?, ?, ?, ?, ?)
                    RETURNING id
                """, (
                    text,
                    summary,
                    json.dumps(system_tags),
                    json.dumps(custom_tags) if custom_tags else None,
                    json.dumps(related_columns) if related_columns else None,
                    priority
                ))
                
                knowledge_id = cursor.fetchone()[0]
                
                # Store embedding
                conn.execute("""
                    INSERT INTO knowledge_embeddings
                    (knowledge_id, knowledge_type, embedding)
                    VALUES (?, ?, ?)
                """, (knowledge_id, KnowledgeType.BUSINESS.value, embedding.tobytes()))
                
                # Update FAISS index if it exists
                if self.user_index is not None:
                    self.user_index.add(embedding.reshape(1, -1).astype('float32'))
                
                conn.commit()
                return knowledge_id
                
        except Exception as e:
            self.logger.error(f"Error adding business knowledge: {str(e)}")
            raise

    def get_system_tags(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """Get system tags, optionally filtered by category"""
        try:
            with sqlite3.connect(self.user_db_path) as conn:
                if category:
                    cursor = conn.execute(
                        "SELECT category, name FROM system_tags WHERE category = ?",
                        (category,)
                    )
                else:
                    cursor = conn.execute(
                        "SELECT category, name FROM system_tags"
                    )
                
                tags = {}
                for row in cursor:
                    if row[0] not in tags:
                        tags[row[0]] = []
                    tags[row[0]].append(row[1])
                
                return tags
                
        except Exception as e:
            self.logger.error(f"Error getting system tags: {str(e)}")
            return {}

    def add_custom_pattern(self,
                        name: str,
                        description: str,
                        prompt: str,
                        code: str,
                        tags: List[str],
                        metadata: Dict):
        """Add custom analysis pattern with comprehensive serialization handling"""
        try:
            # Generate embedding for the pattern
            text_to_embed = f"{name} {description} {prompt}"
            embedding = self.model.encode([text_to_embed])[0]
            
            # Initialize user index if needed
            if self.user_index is None:
                self.user_index = faiss.IndexFlatL2(embedding.shape[0])
            
            # Add embedding to index
            vector_id = self.user_index.ntotal
            self.user_index.add(embedding.reshape(1, -1).astype('float32'))
            
            # Clean tags
            cleaned_tags = [str(tag).strip() for tag in tags if tag]
            
            # Convert metadata using enhanced JSON encoder
            try:
                metadata_json = json.dumps(metadata, cls=EnhancedJSONEncoder)
            except Exception as e:
                self.logger.warning(f"Error serializing metadata: {str(e)}")
                # Fallback: convert problematic values to strings
                cleaned_metadata = {}
                for key, value in metadata.items():
                    try:
                        # Try to serialize each value individually
                        json.dumps(value)
                        cleaned_metadata[key] = value
                    except:
                        cleaned_metadata[key] = str(value)
                metadata_json = json.dumps(cleaned_metadata)

            # Save to database with explicit type handling
            with sqlite3.connect(self.user_db_path) as conn:
                # Create a database cursor
                cursor = conn.cursor()
                
                try:
                    cursor.execute("""
                        INSERT INTO custom_patterns (
                            name, description, prompt, code, tags,
                            metadata, vector_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(name),
                        str(description),
                        str(prompt),
                        str(code),
                        json.dumps(cleaned_tags),
                        metadata_json,
                        int(vector_id)
                    ))
                    
                    conn.commit()
                    
                    # Save pattern ID for reference
                    pattern_id = cursor.lastrowid
                    
                    # Log success with pattern details
                    self.logger.info(
                        f"Successfully added custom pattern: {name} (ID: {pattern_id})"
                    )
                    
                    # Save updated index
                    faiss.write_index(self.user_index, str(self.user_index_path))
                    
                    return pattern_id
                    
                except sqlite3.Error as e:
                    self.logger.error(f"Database error: {str(e)}")
                    conn.rollback()
                    raise
                
        except Exception as e:
            self.logger.error(f"Error adding custom pattern: {str(e)}")
            # Log the problematic data for debugging
            self.logger.error(f"Problematic metadata: {metadata}")
            raise

    def _safe_serialize(self, value):
        """Safely serialize any value to a JSON-compatible format"""
        try:
            json.dumps(value)
            return value
        except:
            if isinstance(value, (bool, int, float)):
                return value
            return str(value)

    def delete_custom_pattern(self, pattern_id: int):
        """Delete custom analysis pattern"""
        try:
            with sqlite3.connect(self.user_db_path) as conn:
                # Delete from custom_patterns table
                conn.execute("""
                    DELETE FROM custom_patterns 
                    WHERE id = ?
                """, (pattern_id,))
                
                # Delete associated embedding
                conn.execute("""
                    DELETE FROM knowledge_embeddings
                    WHERE knowledge_id = ? AND knowledge_type = ?
                """, (pattern_id, KnowledgeType.PATTERN.value))
                
                conn.commit()
                
                # Note: We don't remove from FAISS index as it would require rebuilding
                # Instead, we'll filter out deleted entries during retrieval
                
                self.logger.info(f"Successfully deleted pattern {pattern_id}")
                
        except Exception as e:
            self.logger.error(f"Error deleting pattern: {str(e)}")
            raise

    def delete_knowledge(self, knowledge_id: int, knowledge_type: KnowledgeType):
        """Delete knowledge entry"""
        try:
            with sqlite3.connect(self.user_db_path) as conn:
                # Determine table name
                if knowledge_type == KnowledgeType.COLUMN:
                    table = "column_dictionary"
                elif knowledge_type == KnowledgeType.BUSINESS:
                    table = "business_knowledge"
                else:
                    table = "custom_patterns"
                
                # Delete from table
                conn.execute(f"DELETE FROM {table} WHERE id = ?", (knowledge_id,))
                
                # Delete embedding
                conn.execute("""
                    DELETE FROM knowledge_embeddings
                    WHERE knowledge_id = ? AND knowledge_type = ?
                """, (knowledge_id, knowledge_type.value))
                
                conn.commit()
                
            # Note: We don't remove from FAISS index as it would require rebuilding
            # Instead, we'll filter out deleted entries during retrieval
            
        except Exception as e:
            self.logger.error(f"Error deleting knowledge: {str(e)}")
            raise

    def search_knowledge(self, 
                        query: str, 
                        knowledge_type: Optional[KnowledgeType] = None,
                        limit: int = 10) -> List[Dict]:
        """Search knowledge base"""
        try:
            # Get query embedding
            query_embedding = self.model.encode([query])[0]
            
            results = []
            with sqlite3.connect(self.user_db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Build query based on knowledge type
                if knowledge_type == KnowledgeType.COLUMN:
                    cursor = conn.execute("""
                        SELECT cd.*, ke.embedding
                        FROM column_dictionary cd
                        JOIN knowledge_embeddings ke 
                            ON ke.knowledge_id = cd.id 
                            AND ke.knowledge_type = ?
                    """, (knowledge_type.value,))
                elif knowledge_type == KnowledgeType.BUSINESS:
                    cursor = conn.execute("""
                        SELECT bk.*, ke.embedding
                        FROM business_knowledge bk
                        JOIN knowledge_embeddings ke 
                            ON ke.knowledge_id = bk.id 
                            AND ke.knowledge_type = ?
                    """, (knowledge_type.value,))
                else:
                    cursor = conn.execute("""
                        SELECT cp.*, ke.embedding
                        FROM custom_patterns cp
                        JOIN knowledge_embeddings ke 
                            ON ke.knowledge_id = cp.id 
                            AND ke.knowledge_type = 'pattern'
                    """)
                
                entries = cursor.fetchall()
                
                # Calculate similarities
                for entry in entries:
                    embedding = np.frombuffer(entry['embedding'])
                    similarity = np.dot(query_embedding, embedding)
                    results.append({
                        'entry': dict(entry),
                        'similarity': similarity
                    })
                
                # Sort by similarity and return top results
                results.sort(key=lambda x: x['similarity'], reverse=True)
                return [r['entry'] for r in results[:limit]]
                
        except Exception as e:
            self.logger.error(f"Error searching knowledge: {str(e)}")
            return []

    def export_knowledge(self, export_path: Path):
        """Export all knowledge to file"""
        try:
            with sqlite3.connect(self.user_db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                export_data = {
                    'column_dictionary': [dict(row) for row in conn.execute(
                        "SELECT * FROM column_dictionary"
                    )],
                    'business_knowledge': [dict(row) for row in conn.execute(
                        "SELECT * FROM business_knowledge"
                    )],
                    'custom_patterns': [dict(row) for row in conn.execute(
                        "SELECT * FROM custom_patterns"
                    )],
                    'system_tags': [dict(row) for row in conn.execute(
                        "SELECT * FROM system_tags"
                    )]
                }
                
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                    
        except Exception as e:
            self.logger.error(f"Error exporting knowledge: {str(e)}")
            raise

    def import_knowledge(self, import_path: Path):
        """Import knowledge from file"""
        try:
            with open(import_path) as f:
                import_data = json.load(f)
            
            with sqlite3.connect(self.user_db_path) as conn:
                # Import system tags
                for tag in import_data.get('system_tags', []):
                    conn.execute("""
                        INSERT OR IGNORE INTO system_tags (name, category, description)
                        VALUES (?, ?, ?)
                    """, (tag['name'], tag['category'], tag['description']))
                
                # Import column dictionary
                for entry in import_data.get('column_dictionary', []):
                    self.add_column_definition(
                        entry['column_pattern'],
                        entry['description'],
                        entry.get('data_type'),
                        json.loads(entry['examples']) if entry.get('examples') else None,
                        entry.get('notes')
                    )
                
                # Import business knowledge
                for entry in import_data.get('business_knowledge', []):
                    self.add_business_knowledge(
                        entry['text'],
                        json.loads(entry['system_tags']),
                        json.loads(entry['custom_tags']) if entry.get('custom_tags') else None,
                        json.loads(entry['related_columns']) if entry.get('related_columns') else None,
                        entry.get('priority', 1),
                        entry.get('summary')
                    )
                
                # Import custom patterns
                for entry in import_data.get('custom_patterns', []):
                    self.add_custom_pattern(
                        entry['name'],
                        entry['description'],
                        entry['prompt'],
                        entry['code'],
                        json.loads(entry['tags']),
                        json.loads(entry['metadata'])
                    )
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error importing knowledge: {str(e)}")
            raise

    def get_knowledge_base_stats(self) -> Dict:
        """Get statistics about current knowledge base"""
        stats = {
            'base_patterns': 0,
            'custom_patterns': 0,
            'last_updated': None
        }
        
        try:
            # Get base pattern count
            if self.base_db_path.exists():
                with sqlite3.connect(self.base_db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM analysis_patterns")
                    stats['base_patterns'] = cursor.fetchone()[0]
            
            # Get custom pattern count and last update
            if self.user_db_path.exists():
                with sqlite3.connect(self.user_db_path) as conn:
                    cursor = conn.execute("""
                        SELECT 
                            COUNT(*) as count,
                            MAX(created_at) as last_updated
                        FROM custom_patterns
                    """)
                    result = cursor.fetchone()
                    stats['custom_patterns'] = result[0]
                    stats['last_updated'] = result[1]
            
        except Exception as e:
            self.logger.error(f"Error getting knowledge base stats: {str(e)}")
        
        return stats

    def get_custom_patterns(self) -> List[Dict]:
        """Get list of all custom patterns"""
        patterns = []
        try:
            if self.user_db_path.exists():
                with sqlite3.connect(self.user_db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute("""
                        SELECT 
                            id, name, description, 
                            created_at, tags
                        FROM custom_patterns
                        ORDER BY created_at DESC
                    """)
                    
                    for row in cursor:
                        patterns.append(dict(row))
                        
        except Exception as e:
            self.logger.error(f"Error getting custom patterns: {str(e)}")
        
        return patterns

    def get_column_definitions(self) -> List[Dict]:
        """Get all column definitions with debug logging"""
        print("\nGetting column definitions...")
        definitions = []
        try:
            if self.user_db_path.exists():
                print(f"Database exists at: {self.user_db_path}")
                with sqlite3.connect(self.user_db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    print("Executing query...")
                    cursor = conn.execute("""
                        SELECT 
                            id, column_pattern, description, data_type,
                            examples, notes, created_at
                        FROM column_dictionary
                        ORDER BY created_at DESC
                    """)
                    
                    print("Processing query results...")
                    rows = cursor.fetchall()
                    print(f"Found {len(rows)} rows")
                    
                    for row in rows:  # Use rows instead of cursor
                        row_dict = dict(row)
                        definitions.append(row_dict)
                        print(f"Added definition for column: {row_dict.get('column_pattern', 'unknown')}")
                    
                    print(f"Retrieved {len(definitions)} total definitions")
                    return definitions
                            
        except Exception as e:
            print(f"Error getting column definitions: {str(e)}")
            return []

    def get_business_knowledge(self) -> List[Dict]:
        """Get business knowledge entries"""
        try:
            with sqlite3.connect(self.user_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM business_knowledge 
                    WHERE text IS NOT NULL AND trim(text) != ''
                    ORDER BY priority DESC, created_at DESC
                """)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Error getting business knowledge: {str(e)}")
            return []

    def test_logging(self):
        """Test method to verify RAG logging is working"""
        self.logger.debug("RAG Debug Test")
        self.logger.info("RAG Info Test")
        self.logger.warning("RAG Warning Test")
        self.logger.error("RAG Error Test")
        return "Logging test completed"

