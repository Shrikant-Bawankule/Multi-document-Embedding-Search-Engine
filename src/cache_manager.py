import sqlite3
import pickle
import numpy as np
from datetime import datetime
from typing import Optional, Dict
import os


class CacheManager:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.db_path = os.path.join(cache_dir, "embeddings.db")
        self._init_database()
        
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'embeddings_generated': 0
        }
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id TEXT PRIMARY KEY,
                embedding BLOB,
                hash TEXT,
                doc_length INTEGER,
                updated_at TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_cached_embedding(self, doc_id: str, current_hash: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding if hash matches"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT embedding, hash FROM embeddings WHERE doc_id = ?',
            (doc_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            cached_embedding_blob, cached_hash = result
            
            # Check if document has changed
            if cached_hash == current_hash:
                self.stats['cache_hits'] += 1
                return pickle.loads(cached_embedding_blob)
            else:
                self.stats['cache_misses'] += 1
                return None
        else:
            self.stats['cache_misses'] += 1
            return None
    
    def save_embedding(self, doc_id: str, embedding: np.ndarray, 
                      doc_hash: str, doc_length: int):
        """Save embedding to cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_blob = pickle.dumps(embedding)
        timestamp = datetime.now()
        
        cursor.execute('''
            INSERT OR REPLACE INTO embeddings 
            (doc_id, embedding, hash, doc_length, updated_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (doc_id, embedding_blob, doc_hash, doc_length, timestamp))
        
        conn.commit()
        conn.close()
        
        self.stats['embeddings_generated'] += 1
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Retrieve all cached embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT doc_id, embedding FROM embeddings')
        results = cursor.fetchall()
        conn.close()
        
        embeddings_dict = {}
        for doc_id, embedding_blob in results:
            embeddings_dict[doc_id] = pickle.loads(embedding_blob)
        
        return embeddings_dict
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        cached_count = cursor.fetchone()[0]
        conn.close()
        
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cached_embeddings': cached_count,
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate_percent': round(hit_rate, 2),
            'embeddings_generated': self.stats['embeddings_generated']
        }
    
    def clear_cache(self):
        """Clear all cached embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM embeddings')
        conn.commit()
        conn.close()
        print("Cache cleared")


if __name__ == "__main__":
    # Test cache manager
    cache = CacheManager()
    
    # Test embedding
    test_embedding = np.random.rand(384)
    cache.save_embedding("doc_001", test_embedding, "test_hash_123", 500)
    
    # Retrieve
    retrieved = cache.get_cached_embedding("doc_001", "test_hash_123")
    print(f"Retrieved embedding: {retrieved is not None}")
    print(f"Cache stats: {cache.get_cache_stats()}")
