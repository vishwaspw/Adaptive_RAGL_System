import os
import pickle
import faiss
import numpy as np
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Add the parent directory to path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import (
    GEMINI_API_KEY, 
    VECTOR_DB_PATH, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP,
    TOP_K_RETRIEVAL
)

class AdaptiveVectorDB:
    """An adaptive vector database that can be continuously updated with new knowledge."""
    
    def __init__(self, load_if_exists: bool = True):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=GEMINI_API_KEY,
            model="models/embedding-001"  # Adding the required model parameter
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # FAISS index and metadata storage
        self.index = None
        self.documents = []
        self.document_metadata = []
        self.dimension = None
        
        # Create storage directory if it doesn't exist
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        
        # Paths for saving/loading
        self.index_path = os.path.join(VECTOR_DB_PATH, "faiss_index.bin")
        self.metadata_path = os.path.join(VECTOR_DB_PATH, "metadata.pkl")
        
        # Load existing database if requested and exists
        if load_if_exists and os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.load()
        else:
            # Initialize empty index
            self._initialize_empty_db()
    
    def _initialize_empty_db(self):
        """Initialize an empty FAISS index."""
        # Get dimension from a sample embedding
        sample_embedding = self.embeddings.embed_query("sample text")
        self.dimension = len(sample_embedding)
        
        # Create empty index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.document_metadata = []
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add texts to the vector database."""
        if not texts:
            return []
        
        # Default metadata if none provided
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Ensure metadatas is the same length as texts
        if len(metadatas) != len(texts):
            raise ValueError("Metadatas must be the same length as texts")
        
        # Add timestamp to metadata
        current_time = datetime.now().isoformat()
        for metadata in metadatas:
            metadata["timestamp"] = current_time
        
        # Split texts into chunks
        documents = []
        for i, text in enumerate(texts):
            chunks = self.text_splitter.split_text(text)
            for chunk in chunks:
                doc = Document(page_content=chunk, metadata=metadatas[i].copy())
                documents.append(doc)
        
        # Get embeddings for the documents
        embeddings = [self.embeddings.embed_query(doc.page_content) for doc in documents]
        
        # Add to FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        if self.index is None:
            self.dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index.add(embeddings_array)
        
        # Store documents and metadata
        doc_ids = list(range(len(self.documents), len(self.documents) + len(documents)))
        self.documents.extend(documents)
        self.document_metadata.extend([doc.metadata for doc in documents])
        
        # Save the updated database
        self.save()
        
        return doc_ids
    
    def update_document(self, doc_id: int, new_text: str, metadata_updates: Optional[Dict[str, Any]] = None):
        """Update a document in the vector database."""
        if doc_id < 0 or doc_id >= len(self.documents):
            raise ValueError(f"Document ID {doc_id} out of range")
        
        # Get current document and metadata
        current_doc = self.documents[doc_id]
        current_metadata = self.document_metadata[doc_id].copy()
        
        # Update metadata if provided
        if metadata_updates:
            current_metadata.update(metadata_updates)
        
        # Add update timestamp
        current_metadata["updated_timestamp"] = datetime.now().isoformat()
        
        # Create new document
        new_doc = Document(page_content=new_text, metadata=current_metadata)
        
        # Generate new embedding
        new_embedding = self.embeddings.embed_query(new_text)
        
        # Replace in FAISS index - requires rebuilding the index
        all_embeddings = []
        for i, doc in enumerate(self.documents):
            if i == doc_id:
                all_embeddings.append(new_embedding)
            else:
                all_embeddings.append(self.embeddings.embed_query(doc.page_content))
        
        # Rebuild the index
        embeddings_array = np.array(all_embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_array)
        
        # Update document and metadata
        self.documents[doc_id] = new_doc
        self.document_metadata[doc_id] = current_metadata
        
        # Save the updated database
        self.save()
    
    def similarity_search(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Document]:
        """Perform similarity search on the vector database."""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        query_embedding_array = np.array([query_embedding]).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_embedding_array, min(k, self.index.ntotal))
        
        # Retrieve and return the documents
        retrieved_docs = [self.documents[i] for i in indices[0]]
        
        return retrieved_docs
    
    def save(self):
        """Save the vector database to disk."""
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save metadata and documents
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'document_metadata': self.document_metadata,
                'dimension': self.dimension
            }, f)
    
    def load(self):
        """Load the vector database from disk."""
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Load metadata and documents
        with open(self.metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.document_metadata = data['document_metadata']
            self.dimension = data['dimension']
    
    def get_retriever(self, search_kwargs=None):
        """Create a retriever for use with langchain."""
        if search_kwargs is None:
            search_kwargs = {"k": TOP_K_RETRIEVAL}
            
        return lambda query: self.similarity_search(query, k=search_kwargs.get("k", TOP_K_RETRIEVAL)) 