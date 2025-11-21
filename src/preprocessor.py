import os
import re
import hashlib
from typing import List, Dict, Tuple
from sklearn.datasets import fetch_20newsgroups
from bs4 import BeautifulSoup
import json


class DocumentPreprocessor:
    def __init__(self, data_dir: str = "data/docs"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_20newsgroups(self, n_docs_per_category: int = 10) -> List[Dict]:
        """Download 20 newsgroups and select subset"""
        print("Downloading 20 newsgroups dataset...")
        
        # Remove headers, footers, quotes for realistic training
        newsgroups = fetch_20newsgroups(
            subset='all',
            remove=('headers', 'footers', 'quotes'),
            shuffle=True,
            random_state=42
        )
        
        documents = []
        category_counts = {}
        
        for idx, (text, category_id) in enumerate(zip(newsgroups.data, newsgroups.target)):
            category = newsgroups.target_names[category_id]
            
            if category not in category_counts:
                category_counts[category] = 0
            
            if category_counts[category] < n_docs_per_category:
                documents.append({
                    'text': text,
                    'category': category,
                    'original_id': idx
                })
                category_counts[category] += 1
            
            # Stop when we have enough from all categories
            if all(count >= n_docs_per_category for count in category_counts.values()):
                break
        
        print(f"Loaded {len(documents)} documents from {len(category_counts)} categories")
        return documents
    
    def clean_text(self, text: str) -> str:
        """Clean text: lowercase, remove HTML, normalize spaces"""
        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # Lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()
    
    def compute_hash(self, text: str) -> str:
        """Compute SHA-256 hash for cache lookup"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def save_documents(self, documents: List[Dict]) -> List[Dict]:
        """Save documents as .txt files and return metadata"""
        metadata_list = []
        
        for idx, doc in enumerate(documents):
            doc_id = f"doc_{idx:03d}"
            filename = f"{doc_id}.txt"
            filepath = os.path.join(self.data_dir, filename)
            
            # Clean text
            cleaned_text = self.clean_text(doc['text'])
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # Create metadata
            metadata = {
                'doc_id': doc_id,
                'filename': filename,
                'filepath': filepath,
                'doc_length': len(cleaned_text),
                'hash': self.compute_hash(cleaned_text),
                'category': doc['category']
            }
            metadata_list.append(metadata)
        
        # Save metadata
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        print(f"Saved {len(metadata_list)} documents to {self.data_dir}/")
        return metadata_list
    
    def load_documents_from_disk(self) -> Tuple[List[str], List[Dict]]:
        """Load documents from disk"""
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        
        with open(metadata_path, 'r') as f:
            metadata_list = json.load(f)
        
        documents = []
        for meta in metadata_list:
            with open(meta['filepath'], 'r', encoding='utf-8') as f:
                documents.append(f.read())
        
        return documents, metadata_list


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DocumentPreprocessor()
    docs = preprocessor.load_20newsgroups(n_docs_per_category=10)
    metadata = preprocessor.save_documents(docs)
    print(f"\nProcessed {len(metadata)} documents")
    print(f"Sample: {metadata[0]}")
