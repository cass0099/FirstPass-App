import os
from pathlib import Path

def reset_knowledge_base():
    """Reset the knowledge base by removing existing files"""
    base_dir = Path("base_knowledge")
    
    # Remove database file
    db_path = base_dir / "base_knowledge.db"
    if db_path.exists():
        os.remove(db_path)
        print(f"Removed database: {db_path}")
    
    # Remove vector index
    index_path = base_dir / "base_vectors.index"
    if index_path.exists():
        os.remove(index_path)
        print(f"Removed index: {index_path}")
    
    print("Knowledge base reset complete")

if __name__ == "__main__":
    reset_knowledge_base()