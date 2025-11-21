from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import sys
import os
sys.path.append(os.path.dirname(__file__))
from cache_manager import CacheManager
import time


class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_manager: CacheManager = None):
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.cache_manager = cache_manager or CacheManager()
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """L2 normalize embedding for cosine similarity"""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def generate_embedding(self, text: str, doc_id: str, 
                          doc_hash: str, doc_length: int) -> np.ndarray:
        """Generate embedding with cache check"""
        # Check cache first
        cached_embedding = self.cache_manager.get_cached_embedding(doc_id, doc_hash)
        
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate new embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        embedding = self.normalize_embedding(embedding)
        
        # Save to cache
        self.cache_manager.save_embedding(doc_id, embedding, doc_hash, doc_length)
        
        return embedding
    
    def batch_generate(self, documents: List[str], metadata_list: List[Dict],
                      show_progress: bool = True) -> Dict[str, np.ndarray]:
        """Generate embeddings for multiple documents with caching"""
        embeddings_dict = {}
        
        start_time = time.time()
        
        for idx, (text, meta) in enumerate(zip(documents, metadata_list)):
            if show_progress and (idx + 1) % 20 == 0:
                print(f"Processing document {idx + 1}/{len(documents)}...")
            
            embedding = self.generate_embedding(
                text, 
                meta['doc_id'], 
                meta['hash'], 
                meta['doc_length']
            )
            embeddings_dict[meta['doc_id']] = embedding
        
        elapsed = time.time() - start_time
        
        if show_progress:
            stats = self.cache_manager.get_cache_stats()
            print(f"\nEmbedding generation complete!")
            print(f"Total time: {elapsed:.2f}s")
            print(f"Cache hits: {stats['cache_hits']}")
            print(f"Cache misses: {stats['cache_misses']}")
            print(f"Hit rate: {stats['hit_rate_percent']}%")
        
        return embeddings_dict
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode search query"""
        embedding = self.model.encode(query, convert_to_numpy=True)
        return self.normalize_embedding(embedding)


if __name__ == "__main__":
    from preprocessor import DocumentPreprocessor
    
    # Test embedding generation
    preprocessor = DocumentPreprocessor()
    documents, metadata = preprocessor.load_documents_from_disk()
    
    embedder = EmbeddingGenerator()
    embeddings = embedder.batch_generate(documents[:10], metadata[:10])
    
    print(f"\nGenerated {len(embeddings)} embeddings")
    print(f"Sample shape: {list(embeddings.values())[0].shape}")
