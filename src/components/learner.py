from typing import List, Dict, Any, Optional, Tuple
import re
import os
import sys
from langchain_google_genai import GoogleGenerativeAI
import json

# Add the parent directory to path to resolve imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import (
    GEMINI_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    AUTO_UPDATE_THRESHOLD
)

class AdaptiveLearner:
    """
    Component that manages the continuous learning process for the RAG system.
    It extracts new knowledge from user interactions and updates the vector database.
    """
    
    def __init__(self, vector_db):
        """
        Initialize the learner with the vector database.
        
        Args:
            vector_db: The vector database instance to update.
        """
        self.vector_db = vector_db
        self.llm = GoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=TEMPERATURE
        )
        self.interaction_history = []
    
    def process_interaction(self, user_query: str, assistant_response: str) -> Optional[Dict[str, Any]]:
        """
        Process a user-assistant interaction to extract potential new knowledge.
        
        Args:
            user_query: The user's query
            assistant_response: The assistant's response
            
        Returns:
            Dictionary containing extracted knowledge, if any
        """
        # Save the interaction to history
        self.interaction_history.append({
            "user_query": user_query,
            "assistant_response": assistant_response
        })
        
        # Only process if we have sufficient history for context
        if len(self.interaction_history) < 2:
            return None
        
        # Try to extract new knowledge
        return self._extract_knowledge(user_query, assistant_response)
    
    def _extract_knowledge(self, user_query: str, assistant_response: str) -> Optional[Dict[str, Any]]:
        """
        Extract new knowledge from the user query and assistant response.
        
        Args:
            user_query: The user's query
            assistant_response: The assistant's response
            
        Returns:
            Dictionary containing extracted knowledge if successful, None otherwise
        """
        # Check if the user's query contains potential new information
        # Create a prompt to analyze the interaction
        prompt = f"""
        Analyze the following user query and assistant response for potential new factual information:
        
        USER QUERY: {user_query}
        
        ASSISTANT RESPONSE: {assistant_response}
        
        Task:
        1. Identify if the user is providing new factual information in their query.
        2. Evaluate if this information is worth storing (factual, objective, and potentially useful for future queries).
        3. Extract this information in a concise, factual statement.
        4. Rate your confidence in this extraction from 0.0 to 1.0.
        
        IMPORTANT: You must respond with a valid JSON object only, using the exact format below. 
        Your response should be parseable by Python's json.loads() function.
        
        Output format:
        {{
          "contains_new_knowledge": true,
          "extracted_knowledge": "Concise factual statement here",
          "confidence": 0.8,
          "topics": ["topic1", "topic2"]
        }}
        
        If no new knowledge is detected, respond with:
        {{
          "contains_new_knowledge": false,
          "extracted_knowledge": "",
          "confidence": 0.0,
          "topics": []
        }}
        
        Provide ONLY the valid JSON object as your response, with no additional text.
        """
        
        # Use the LLM to analyze the interaction
        response = self.llm.invoke(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        # Try to parse the JSON response
        try:
            # Use regex to extract the JSON part
            json_match = re.search(r'({[\s\S]*})', response_text)
            if not json_match:
                return None
                
            json_str = json_match.group(1)
            
            # Replace Python boolean literals with JSON boolean literals
            json_str = json_str.replace("True", "true").replace("False", "false")
            
            # Parse using json instead of eval for safety and proper handling
            knowledge_data = json.loads(json_str)
            
            # Check if knowledge was extracted with sufficient confidence
            if knowledge_data.get("contains_new_knowledge", False) and knowledge_data.get("confidence", 0) > AUTO_UPDATE_THRESHOLD:
                return knowledge_data
            
            return None
            
        except Exception as e:
            print(f"Error extracting knowledge: {e}")
            return None
    
    def update_vector_db(self, knowledge_data: Dict[str, Any]) -> int:
        """
        Update the vector database with new knowledge.
        
        Args:
            knowledge_data: Dictionary containing extracted knowledge
            
        Returns:
            Document ID of the added text
        """
        # Prepare the text and metadata
        text = knowledge_data.get("extracted_knowledge", "")
        metadata = {
            "source": "user_interaction",
            "confidence": knowledge_data.get("confidence", 0.0),
            "topics": knowledge_data.get("topics", [])
        }
        
        # Add to vector database
        doc_ids = self.vector_db.add_texts([text], [metadata])
        
        return doc_ids[0] if doc_ids else -1
    
    def process_document_feedback(self, doc_id: int, relevance_score: float) -> bool:
        """
        Process feedback on document relevance to improve retrieval quality.
        
        Args:
            doc_id: The document ID
            relevance_score: Score between 0 and 1 indicating relevance
            
        Returns:
            Success flag
        """
        if doc_id < 0 or doc_id >= len(self.vector_db.documents):
            return False
        
        # Get the document
        doc = self.vector_db.documents[doc_id]
        
        # Update metadata with relevance feedback
        current_metadata = self.vector_db.document_metadata[doc_id].copy()
        
        # Average with existing feedback if available
        existing_score = current_metadata.get("relevance_score", None)
        if existing_score is not None:
            feedback_count = current_metadata.get("feedback_count", 1)
            new_score = (existing_score * feedback_count + relevance_score) / (feedback_count + 1)
            current_metadata["relevance_score"] = new_score
            current_metadata["feedback_count"] = feedback_count + 1
        else:
            current_metadata["relevance_score"] = relevance_score
            current_metadata["feedback_count"] = 1
        
        # Update the document with new metadata
        self.vector_db.update_document(doc_id, doc.page_content, current_metadata)
        
        return True 