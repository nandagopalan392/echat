"""
Chat Service - Business Logic for Chat Operations

Handles all chat-related business logic including message handling,
RLHF feedback processing, and file uploads.
"""
import os
import json
import logging
import asyncio
import sqlite3
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime

from app.core.rag import get_rag_engine
from app.db import DatabaseConnection
from app.db.repositories import ChatRepository
from app.core.training.rlhf import RLHFManager

logger = logging.getLogger(__name__)


class ChatService:
    """
    Service class for managing chat operations.
    
    Handles message sending, session management, RLHF feedback,
    and file uploads for chat context.
    """
    
    def __init__(self):
        """Initialize chat service with dependencies"""
        self.db = DatabaseConnection()
        self.chat_repo = ChatRepository(self.db)
        self.rag_engine = get_rag_engine()
        self.rlhf_manager = RLHFManager()
        logger.info("ChatService initialized")
    
    async def send_message(
        self,
        content: str,
        session_id: Optional[int],
        username: str
    ) -> Dict[str, Any]:
        """
        Send a chat message and get RLHF responses.
        
        Args:
            content: Message content
            session_id: Optional session ID (creates new if None)
            username: Username of the sender
            
        Returns:
            Dictionary with response options for RLHF
        """
        try:
            logger.info(f"Processing message from user '{username}'")
            
            # Create or use existing session
            if not session_id:
                session_id = self.chat_repo.create_session(username, content)
                logger.info(f"Created new session {session_id}")
            
            # Save user message
            self.chat_repo.save_message(session_id, content, is_user=True)
            
            # Generate two responses with different styles for RLHF
            logger.info(f"Generating RLHF response options for: {content[:50]}...")
            
            # Response A: Conversational style
            response_a_chunks = []
            async for chunk in self.rag_engine.stream_response(
                content,
                style="conversational",
                user_id=username
            ):
                response_a_chunks.append(chunk)
            
            response_a_content, response_a_thinking = self._parse_response_chunks(
                response_a_chunks
            )
            logger.info(f"Response A (conversational): {len(response_a_content)} chars")
            
            # Small delay between requests
            await asyncio.sleep(2)
            
            # Response B: Detailed/analytical style
            response_b_chunks = []
            async for chunk in self.rag_engine.stream_response(
                content,
                style="detailed",
                user_id=username
            ):
                response_b_chunks.append(chunk)
            
            response_b_content, response_b_thinking = self._parse_response_chunks(
                response_b_chunks
            )
            logger.info(f"Response B (detailed): {len(response_b_content)} chars")
            
            # Prepare response options
            response_options = [
                {
                    "thinking": response_a_thinking,
                    "content": response_a_content,
                    "style": "conversational"
                },
                {
                    "thinking": response_b_thinking,
                    "content": response_b_content,
                    "style": "detailed"
                }
            ]
            
            # Save response options for RLHF
            try:
                self.rlhf_manager.save_response_options(
                    session_id=session_id,
                    question=content,
                    response_option_0=response_a_content,
                    response_option_1=response_b_content,
                    username=username
                )
                logger.info(f"Saved RLHF response options for session {session_id}")
            except Exception as e:
                logger.error(f"Failed to save RLHF response options: {str(e)}")
            
            # Return response with options
            return {
                "content": "I've generated two different responses for you to choose from: one conversational and friendly, the other detailed and analytical. Please select your preferred approach:",
                "full_response": "I've generated two different responses for you to choose from: one conversational and friendly, the other detailed and analytical. Please select your preferred approach:",
                "is_final": True,
                "session_id": session_id,
                "response_options": response_options,
                "rlhf_enabled": True,
                "message": "Choose between conversational (friendly explanations) and detailed (comprehensive analysis) responses:",
                "thinking_included": bool(response_a_thinking or response_b_thinking)
            }
            
        except Exception as e:
            logger.error(f"Error in send_message: {str(e)}", exc_info=True)
            raise
    
    def _parse_response_chunks(self, chunks: List[Any]) -> tuple[str, str]:
        """
        Parse response chunks to extract content and thinking.
        
        Args:
            chunks: List of response chunks
            
        Returns:
            Tuple of (content, thinking)
        """
        content = ""
        thinking = ""
        
        for chunk in chunks:
            if isinstance(chunk, str):
                try:
                    chunk_data = json.loads(chunk)
                    if isinstance(chunk_data, dict):
                        if chunk_data.get("thinking"):
                            thinking = chunk_data["thinking"]
                        if chunk_data.get("content"):
                            content = chunk_data["content"]
                    else:
                        content += str(chunk)
                except (json.JSONDecodeError, TypeError):
                    content += str(chunk)
            else:
                content += str(chunk)
        
        # Fallback if no content extracted
        if not content:
            content = "".join(str(c) for c in chunks)
        
        return content, thinking
    
    def get_user_sessions(self, username: str) -> List[Dict[str, Any]]:
        """
        Get all chat sessions for a user.
        
        Args:
            username: Username to fetch sessions for
            
        Returns:
            List of session dictionaries
        """
        try:
            sessions = self.chat_repo.get_user_sessions(username)
            logger.info(f"Retrieved {len(sessions)} sessions for user '{username}'")
            # Convert ChatSession objects to dicts
            return [
                {
                    "id": session.id,
                    "username": session.username,
                    "topic": session.topic,
                    "created_at": session.created_at,
                    "last_updated": session.last_updated
                }
                for session in sessions
            ]
        except Exception as e:
            logger.error(f"Error getting user sessions: {str(e)}")
            raise
    
    def get_session_messages(self, session_id: int, username: str) -> Dict[str, Any]:
        """
        Get all messages for a specific session.
        
        Args:
            session_id: Session ID to fetch messages for
            username: Username (for authorization)
            
        Returns:
            Dictionary with messages and session_id
        """
        try:
            logger.info(f"Fetching messages for session {session_id} by user {username}")
            
            messages = self.chat_repo.get_session_messages(session_id)
            
            # Format messages - convert Message objects to dicts
            formatted_messages = []
            for i, msg in enumerate(messages):
                # Handle timestamp - it's stored as string in DB
                timestamp = msg.timestamp
                if timestamp and hasattr(timestamp, 'isoformat'):
                    timestamp = timestamp.isoformat()
                
                formatted_msg = {
                    "id": f"{session_id}-{i}",
                    "content": msg.content,
                    "isUser": msg.is_user,
                    "timestamp": timestamp
                }
                formatted_messages.append(formatted_msg)
                logger.debug(
                    f"Message {i+1}: isUser={formatted_msg['isUser']}, "
                    f"content='{formatted_msg['content'][:50]}...'"
                )
            
            logger.info(f"Returning {len(formatted_messages)} messages for session {session_id}")
            
            return {
                "messages": formatted_messages,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error getting session messages: {str(e)}")
            raise
    
    async def process_rlhf_feedback(
        self,
        session_id: int,
        chosen_index: int,
        username: str
    ) -> Dict[str, str]:
        """
        Process RLHF feedback and save chosen response.
        
        Args:
            session_id: Session ID
            chosen_index: Index of chosen response (0 or 1)
            username: Username submitting feedback
            
        Returns:
            Success status dictionary
        """
        try:
            logger.info(
                f"Processing RLHF feedback for session {session_id}, "
                f"chosen_index: {chosen_index}"
            )
            
            # Get response options from repository
            chosen_response_content = None
            try:
                response_options = self.rlhf_manager.repository.get_response_options(
                    session_id=session_id
                )
                
                if response_options and len(response_options) > 0:
                    # Get the most recent options
                    latest_options = response_options[0]
                    options_list = [
                        latest_options.get('response_option_0', ''),
                        latest_options.get('response_option_1', '')
                    ]
                    
                    if 0 <= chosen_index < len(options_list):
                        chosen_response_content = options_list[chosen_index]
                        logger.info(f"Found chosen response: {chosen_response_content[:100]}...")
                    else:
                        logger.error(f"Invalid chosen_index {chosen_index}, defaulting to 0")
                        chosen_response_content = options_list[0]
                        chosen_index = 0
                else:
                    logger.error(f"No response options found for session {session_id}")
            
            except Exception as e:
                logger.error(f"Error retrieving response options: {str(e)}")
            
            # Save user's preference
            success = self.rlhf_manager.save_selected_response(
                session_id=session_id,
                chosen_index=chosen_index,
                user_id=username
            )
            
            if not success:
                logger.error(f"Failed to save RLHF preference for session {session_id}")
                raise Exception("Failed to save RLHF feedback")
            
            # Save chosen response to chat history
            if chosen_response_content:
                try:
                    # Check for duplicates and save if not exists
                    with self.db.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            """SELECT COUNT(*) FROM messages 
                               WHERE session_id = ? AND content = ? AND is_user = 0""",
                            (session_id, chosen_response_content)
                        )
                        
                        if cursor.fetchone()[0] == 0:  # Not found, save it
                            # Save using the repository method
                            self.chat_repo.save_message(
                                session_id,
                                chosen_response_content,
                                is_user=False
                            )
                            logger.info(
                                f"✅ Successfully saved chosen RLHF response as chat message "
                                f"for session {session_id}"
                            )
                        else:
                            logger.info(
                                f"✅ Chosen RLHF response already exists in chat history "
                                f"for session {session_id}"
                            )
                            
                except Exception as e:
                    logger.error(f"❌ Error saving chosen response as chat message: {str(e)}")
                    raise Exception("Failed to save response to chat history")
            else:
                logger.error(f"❌ No chosen response content to save for session {session_id}")
                raise Exception("No response content found to save")
            
            logger.info(
                f"✅ RLHF feedback processing completed successfully for session {session_id}"
            )
            
            return {"status": "success", "message": "Feedback received and processed"}
            
        except Exception as e:
            logger.error(f"❌ Error processing RLHF feedback: {str(e)}")
            raise
    
    def update_message(
        self,
        session_id: int,
        content: str,
        username: str
    ) -> Dict[str, Any]:
        """
        Update the latest AI message in a session.
        
        Args:
            session_id: Session ID
            content: New message content
            username: Username (for authorization)
            
        Returns:
            Success status with message_id
        """
        try:
            # Get latest AI message ID
            message_id = self.chat_repo.get_latest_ai_message_id(session_id)
            
            if not message_id:
                raise ValueError("No AI message found in session")
            
            # Update message
            success = self.chat_repo.update_message(message_id, content)
            
            if not success:
                raise Exception("Failed to update message")
            
            logger.info(f"Message {message_id} updated for session {session_id}")
            return {"status": "success", "message_id": message_id}
            
        except Exception as e:
            logger.error(f"Error updating message: {str(e)}")
            raise
    
    async def upload_file(
        self,
        filename: str,
        file_content: bytes,
        username: str
    ) -> Dict[str, str]:
        """
        Upload and process a file for RAG context.
        
        Args:
            filename: Original filename
            file_content: File binary content
            username: Username uploading the file
            
        Returns:
            Success status with file_id
        """
        try:
            file_id = f"upload_{datetime.now().timestamp()}"
            temp_filename = f"temp_{filename}"
            
            # Write file temporarily
            with open(temp_filename, "wb") as f:
                f.write(file_content)
            
            # Process with RAG engine
            success = self.rag_engine.ingest_with_storage(temp_filename, filename)
            
            # Cleanup temp file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            
            if success:
                logger.info(f"File {filename} processed successfully by user {username}")
                return {"message": "File processed successfully", "file_id": file_id}
            else:
                raise Exception("Failed to process file")
                
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            # Cleanup on error
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            raise
