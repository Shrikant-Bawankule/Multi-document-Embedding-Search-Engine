from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import os

# Import your modules
import sys
sys.path.append(os.path.dirname(__file__))

from preprocessor import DocumentPreprocessor
from embedder import EmbeddingGenerator
from search_engine import SearchEngine

# Initialize FastAPI app
app = FastAPI(
    title="Embedding Search Engine API",
    description="Multi-document semantic search with caching",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text", min_length=1)
    top_k: int = Field(5, description="Number of results to return", ge=1, le=20)

class SearchResult(BaseModel):
    rank: int
    doc_id: str
    score: float
    preview: str
    category: str
    doc_length: int
    explanation: dict

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    processing_time_ms: float
    total_documents: int

# Global variables (loaded at startup)
search_engine = None
embedder = None
documents = None
metadata = None

@app.on_event("startup")
async def startup_event():
    """Initialize search engine on startup"""
    global search_engine, embedder, documents, metadata
    
    print("Loading search engine...")
    
    # Load preprocessed documents
    preprocessor = DocumentPreprocessor()
    
    # Check if documents exist, if not, create them
    if not os.path.exists("data/docs/metadata.json"):
        print("Documents not found. Preprocessing...")
        docs = preprocessor.load_20newsgroups(n_docs_per_category=10)
        metadata = preprocessor.save_documents(docs)
    
    documents, metadata = preprocessor.load_documents_from_disk()
    
    # Initialize embedder
    embedder = EmbeddingGenerator()
    
    # Generate embeddings (uses cache)
    embeddings = embedder.batch_generate(documents, metadata, show_progress=True)
    
    # Build search index
    search_engine = SearchEngine()
    search_engine.build_index(embeddings, documents, metadata)
    
    print("Search engine ready!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Embedding Search Engine API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search",
            "health": "/health",
            "stats": "/stats"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "index_loaded": search_engine is not None,
        "total_documents": len(documents) if documents else 0
    }

@app.get("/stats")
async def get_stats():
    """Get cache and index statistics"""
    cache_stats = embedder.cache_manager.get_cache_stats()
    
    return {
        "total_documents": len(documents) if documents else 0,
        "index_size": search_engine.index.ntotal if search_engine else 0,
        "embedding_dimension": embedder.embedding_dim,
        "cache_stats": cache_stats
    }

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for similar documents
    
    Example request:
    ```
    {
        "query": "quantum physics basics",
        "top_k": 5
    }
    ```
    """
    if search_engine is None or embedder is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    start_time = time.time()
    
    try:
        # Encode query
        query_embedding = embedder.encode_query(request.query)
        
        # Search
        results = search_engine.search(
            query_embedding, 
            top_k=request.top_k,
            query_text=request.query
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return SearchResponse(
            query=request.query,
            results=results,
            processing_time_ms=round(processing_time, 2),
            total_documents=len(documents)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
