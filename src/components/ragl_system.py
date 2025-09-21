from typing import Dict, List, Any, Optional
import os
import json
from datetime import datetime
import sys

# Add the parent directory to path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.components.vector_db import AdaptiveVectorDB
from src.components.learner import AdaptiveLearner
from src.utils.model_loader import create_rag_chain

class RAGLSystem:
    """
    Retrieval-Augmented Generation Learning system that continuously 
    curates and updates vector databases with new knowledge.
    """
    
    def __init__(self, init_docs: Optional[List[str]] = None, init_metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the RAGL system.
        
        Args:
            init_docs: Optional list of documents to initialize the system with
            init_metadatas: Optional list of metadata for each document
        """
        # Initialize vector database
        self.vector_db = AdaptiveVectorDB()
        
        # Initialize learner
        self.learner = AdaptiveLearner(self.vector_db)
        
        # Add initial documents if provided
        if init_docs:
            self.vector_db.add_texts(init_docs, init_metadatas)
        
        # Create RAG chain
        self.retriever = self.vector_db.get_retriever()
        self.chain = create_rag_chain(self.retriever)
        
        # Conversation history
        self.conversation_history = []
        
        # Session information
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.learning_events = []
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and return a response along with retrieved documents.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing the response and additional information
        """
        # Retrieve relevant documents for display purposes
        retrieved_docs = self.retriever(query)
        
        # Generate response using the chain
        response = self.chain(query)
        
        # Save to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": query,
            "timestamp": datetime.now().isoformat()
        })
        
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Process the interaction for learning
        if len(self.conversation_history) >= 2:
            knowledge_data = self.learner.process_interaction(query, response)
            
            # If new knowledge was detected, update the vector database
            if knowledge_data:
                doc_id = self.learner.update_vector_db(knowledge_data)
                
                # Record the learning event
                learning_event = {
                    "timestamp": datetime.now().isoformat(),
                    "knowledge": knowledge_data.get("extracted_knowledge", ""),
                    "confidence": knowledge_data.get("confidence", 0.0),
                    "doc_id": doc_id
                }
                self.learning_events.append(learning_event)
                
                # Refresh the retriever and chain after updating the DB
                self.retriever = self.vector_db.get_retriever()
                self.chain = create_rag_chain(self.retriever)
        
        # Return response with relevant information
        return {
            "response": response,
            "retrieved_docs": [{"content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs],
            "learning_occurred": bool(self.learning_events) and self.learning_events[-1]["timestamp"] > self.conversation_history[-2]["timestamp"]
        }
    
    def save_session(self, file_path: Optional[str] = None) -> str:
        """
        Save the current session information.
        
        Args:
            file_path: Optional path to save the session file
            
        Returns:
            Path to the saved session file
        """
        if file_path is None:
            os.makedirs("data/sessions", exist_ok=True)
            file_path = f"data/sessions/session_{self.session_id}.json"
        
        # Create session data
        session_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "conversation_history": self.conversation_history,
            "learning_events": self.learning_events
        }
        
        # Save to file
        with open(file_path, "w") as f:
            json.dump(session_data, f, indent=2)
        
        return file_path
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of documents to add
            metadatas: Optional list of metadata for each document
            
        Returns:
            List of document IDs
        """
        # Add documents to vector database
        doc_ids = self.vector_db.add_texts(documents, metadatas)
        
        # Refresh the retriever and chain after updating the DB
        self.retriever = self.vector_db.get_retriever()
        self.chain = create_rag_chain(self.retriever)
        
        return doc_ids
    
    def provide_feedback(self, doc_id: int, relevance_score: float) -> bool:
        """
        Provide feedback on document relevance to improve retrieval.
        
        Args:
            doc_id: Document ID
            relevance_score: Score between 0 and 1 indicating relevance
            
        Returns:
            Success flag
        """
        # Process feedback
        success = self.learner.process_document_feedback(doc_id, relevance_score)
        
        # Refresh the retriever and chain if update was successful
        if success:
            self.retriever = self.vector_db.get_retriever()
            self.chain = create_rag_chain(self.retriever)
        
        return success 