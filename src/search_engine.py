import faiss
import numpy as np
from typing import List, Dict, Tuple
import json
import os
from collections import Counter
import re


class SearchEngine:
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.doc_ids = []
        self.documents = []
        self.metadata = []
        
    def build_index(self, embeddings_dict: Dict[str, np.ndarray], 
                   documents: List[str], metadata_list: List[Dict]):
        """Build FAISS index from embeddings"""
        print("Building FAISS index...")
        
        # Store documents and metadata
        self.doc_ids = list(embeddings_dict.keys())
        self.documents = documents
        self.metadata = metadata_list
        
        # Convert embeddings to matrix
        embedding_matrix = np.array([embeddings_dict[doc_id] for doc_id in self.doc_ids])
        embedding_matrix = embedding_matrix.astype('float32')
        
        # Create FAISS index (IndexFlatIP for cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embedding_matrix)
        
        print(f"Index built with {self.index.ntotal} vectors")
        
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction: lowercase, split, filter short words
        words = re.findall(r'\w+', text.lower())
        words = [w for w in words if len(w) > 3]  # Filter short words
        
        # Get most common words
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common(top_n)]
    
    def explain_ranking(self, query: str, doc_text: str, score: float, 
                       doc_length: int) -> Dict:
        """Explain why document was matched"""
        query_keywords = set(self.extract_keywords(query, top_n=20))
        doc_keywords = set(self.extract_keywords(doc_text, top_n=50))
        
        # Find overlapping keywords
        matched_keywords = list(query_keywords.intersection(doc_keywords))
        
        # Calculate overlap ratio
        overlap_ratio = len(matched_keywords) / len(query_keywords) if query_keywords else 0
        
        # Document length normalization (optional scoring factor)
        length_score = min(doc_length / 500, 1.0)  # Normalize to [0, 1]
        
        return {
            "matched_keywords": matched_keywords[:5],  # Top 5 matches
            "overlap_ratio": round(overlap_ratio, 3),
            "doc_length": doc_length,
            "length_score": round(length_score, 3),
            "embedding_score": round(float(score), 3)
        }
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
              query_text: str = "") -> List[Dict]:
        """Search for similar documents"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Search
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, (distance, doc_idx) in enumerate(zip(distances[0], indices[0])):
            if doc_idx < len(self.doc_ids):  # Valid index
                doc_id = self.doc_ids[doc_idx]
                doc_text = self.documents[doc_idx]
                meta = self.metadata[doc_idx]
                
                # Create preview (first 150 chars)
                preview = doc_text[:150] + "..." if len(doc_text) > 150 else doc_text
                
                # Generate explanation
                explanation = self.explain_ranking(
                    query_text, 
                    doc_text, 
                    distance,
                    meta['doc_length']
                )
                
                results.append({
                    "rank": idx + 1,
                    "doc_id": doc_id,
                    "score": round(float(distance), 4),
                    "preview": preview,
                    "category": meta.get('category', 'unknown'),
                    "doc_length": meta['doc_length'],
                    "explanation": explanation
                })
        
        return results
    
    def save_index(self, filepath: str = "cache/faiss.index"):
        """Save FAISS index to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        faiss.write_index(self.index, filepath)
        
        # Save doc_ids mapping
        mapping_path = filepath.replace('.index', '_mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump({
                'doc_ids': self.doc_ids,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str = "cache/faiss.index"):
        """Load FAISS index from disk"""
        self.index = faiss.read_index(filepath)
        
        # Load doc_ids mapping
        mapping_path = filepath.replace('.index', '_mapping.json')
        with open(mapping_path, 'r') as f:
            data = json.load(f)
            self.doc_ids = data['doc_ids']
            self.embedding_dim = data['embedding_dim']
        
        print(f"Index loaded from {filepath}")


if __name__ == "__main__":
    from preprocessor import DocumentPreprocessor
    from embedder import EmbeddingGenerator
    
    # Load documents
    preprocessor = DocumentPreprocessor()
    documents, metadata = preprocessor.load_documents_from_disk()
    
    # Generate embeddings
    embedder = EmbeddingGenerator()
    embeddings = embedder.batch_generate(documents, metadata)
    
    # Build search index
    search_engine = SearchEngine()
    search_engine.build_index(embeddings, documents, metadata)
    
    # Test search
    query = "computer graphics and image processing"
    query_embedding = embedder.encode_query(query)
    results = search_engine.search(query_embedding, top_k=5, query_text=query)
    
    print(f"\nSearch results for: '{query}'")
    print("=" * 80)
    for result in results:
        print(f"\nRank {result['rank']}: {result['doc_id']} (Score: {result['score']})")
        print(f"Category: {result['category']}")
        print(f"Preview: {result['preview']}")
        print(f"Explanation: {result['explanation']}")
    
    # Save index
    search_engine.save_index()
