# Create a new file for centralized prompt management

from typing import Dict, Any, List

class RAGPrompts:
    """Collection of prompts for RAG pipelines"""
    
    @staticmethod
    def split_document_prompt(query: str, document: str) -> str:
        """
        Creates a prompt for generating a response based on a single document
        
        Args:
            query: User query
            document: Document content to base the response on
            
        Returns:
            Formatted prompt string
        """
        return f"""
        Answer the following question using only the context provided below:
        
        Context: {document}
        
        Question: {query}
        
        Provide a detailed answer focusing exclusively on the information in this context.
        If the context doesn't contain relevant information, say so clearly.
        """
    
    @staticmethod
    def analytical_prompt(query: str, context: str) -> str:
        """
        Creates a prompt for generating an analytical response
        
        Args:
            query: User query
            context: Combined context from documents
            
        Returns:
            Formatted analytical prompt
        """
        return f"""
        Answer the following question based on the provided context using an analytical, precise style:
        
        Context: {context}
        
        Question: {query}
        
        Provide a detailed, factual answer that is comprehensive and accurate, with a focus on
        technical correctness and precision. Use specific terminology and structured reasoning.
        """
    
    @staticmethod
    def conversational_prompt(query: str, context: str) -> str:
        """
        Creates a prompt for generating a conversational response
        
        Args:
            query: User query
            context: Combined context from documents
            
        Returns:
            Formatted conversational prompt
        """
        return f"""
        Answer the following question based on the provided context using a conversational, friendly style:
        
        Context: {context}
        
        Question: {query}
        
        Provide a warm, accessible answer that feels like a helpful conversation. Use simpler language,
        analogies where appropriate, and a more personal tone. Focus on being engaging and relatable.
        """
