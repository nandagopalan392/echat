"""
WebSocket connection managers for real-time updates
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class BaseConnectionManager:
    """Base WebSocket connection manager with reconnection support"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_info: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept and store a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        # Initialize or update connection info
        if connection_id not in self.connection_info:
            self.connection_info[connection_id] = {
                "connected_at": datetime.utcnow(),
                "reconnect_count": 0
            }
        else:
            # This is a reconnection
            self.connection_info[connection_id]["reconnect_count"] += 1
            self.connection_info[connection_id]["connected_at"] = datetime.utcnow()
            logger.info(f"WebSocket reconnected for {connection_id} (attempt #{self.connection_info[connection_id]['reconnect_count']})")
        
        self.connection_info[connection_id]["last_ping"] = datetime.utcnow()
        logger.info(f"WebSocket connected for {connection_id}")
    
    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            
            # Keep connection info for reconnection tracking
            if connection_id in self.connection_info:
                self.connection_info[connection_id]["disconnected_at"] = datetime.utcnow()
            
            logger.info(f"WebSocket disconnected for {connection_id}")
    
    async def send_message(self, connection_id: str, message: dict) -> bool:
        """Send a JSON message to a specific connection"""
        if connection_id in self.active_connections:
            try:
                # Update last activity timestamp
                if connection_id in self.connection_info:
                    self.connection_info[connection_id]["last_activity"] = datetime.utcnow()
                
                await self.active_connections[connection_id].send_text(json.dumps(message))
                return True
                
            except ConnectionResetError:
                logger.info(f"WebSocket connection reset for {connection_id} - client disconnected")
                self.disconnect(connection_id)
                return False
                
            except Exception as e:
                # Log as info instead of error for common disconnection scenarios
                if "disconnected" in str(e).lower() or "closed" in str(e).lower():
                    logger.info(f"WebSocket connection closed for {connection_id}: {e}")
                else:
                    logger.error(f"Failed to send WebSocket message to {connection_id}: {e}")
                self.disconnect(connection_id)
                return False
        
        return False
    
    def is_connected(self, connection_id: str) -> bool:
        """Check if a connection is active"""
        return connection_id in self.active_connections
    
    def get_connection_info(self, connection_id: str) -> Dict[str, Any]:
        """Get connection metadata"""
        return self.connection_info.get(connection_id, {})
    
    def get_active_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)


class EvaluationConnectionManager(BaseConnectionManager):
    """WebSocket connection manager for evaluation tasks"""
    pass


class FinetuningConnectionManager(BaseConnectionManager):
    """WebSocket connection manager for fine-tuning experiments"""
    pass


# Global connection manager instances
evaluation_manager = EvaluationConnectionManager()
finetuning_manager = FinetuningConnectionManager()


def get_evaluation_manager() -> EvaluationConnectionManager:
    """Get the global evaluation connection manager"""
    return evaluation_manager


def get_finetuning_manager() -> FinetuningConnectionManager:
    """Get the global fine-tuning connection manager"""
    return finetuning_manager
