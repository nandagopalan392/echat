/**
 * Robust WebSocket Service with Auto-Reconnection and Polling Fallback
 * 
 * This service implements the "WebSocket for freshness, polling for correctness" pattern:
 * - WebSocket provides real-time updates when connection is stable
 * - Auto-reconnection handles temporary network issues
 * - Polling fallback ensures state consistency during disconnections
 * - Connection health monitoring via ping/pong
 */

class WebSocketService {
    constructor() {
        this.connections = new Map(); // taskId -> connection info
        this.config = {
            reconnectInterval: 3000,     // Start with 3 seconds
            maxReconnectInterval: 30000, // Max 30 seconds
            reconnectDecay: 1.5,         // Exponential backoff
            timeoutInterval: 5000,       // Connection timeout
            maxReconnectAttempts: 10,    // Max attempts before giving up
            pollingInterval: 30000,      // Poll every 30 seconds
            pingInterval: 45000,         // Ping every 45 seconds
            pongTimeout: 5000            // Wait 5 seconds for pong
        };
    }

    /**
     * Create a new WebSocket connection with auto-reconnection and polling
     */
    connect(taskId, options = {}) {
        const {
            onMessage = () => {},
            onError = () => {},
            onClose = () => {},
            onStatusChange = () => {},
            pollCallback = null,
            enablePolling = true,
            endpointType = 'evaluation' // 'evaluation' or 'qca-dataset'
        } = options;

        // If connection already exists, return it
        if (this.connections.has(taskId)) {
            const existing = this.connections.get(taskId);
            if (existing.ws && existing.ws.readyState === WebSocket.OPEN) {
                return existing;
            }
            // Clean up existing connection
            this.disconnect(taskId);
        }

        const connection = {
            taskId,
            ws: null,
            status: 'connecting',
            reconnectAttempts: 0,
            reconnectTimeoutId: null,
            pollingIntervalId: null,
            pingIntervalId: null,
            pongTimeoutId: null,
            lastPongTime: Date.now(),
            callbacks: { onMessage, onError, onClose, onStatusChange },
            pollCallback,
            enablePolling,
            isManualClose: false,
            endpointType // Store endpoint type for WebSocket URL generation
        };

        this.connections.set(taskId, connection);
        this._createWebSocket(connection);
        
        // Don't start polling immediately - wait for WebSocket to fail
        // Polling will be started by _scheduleReconnect if WebSocket fails to connect

        return connection;
    }

    /**
     * Disconnect and cleanup a WebSocket connection
     */
    disconnect(taskId) {
        const connection = this.connections.get(taskId);
        if (!connection) return;

        connection.isManualClose = true;
        
        // Clear all timers
        if (connection.reconnectTimeoutId) {
            clearTimeout(connection.reconnectTimeoutId);
        }
        if (connection.pollingIntervalId) {
            clearInterval(connection.pollingIntervalId);
        }
        if (connection.pingIntervalId) {
            clearInterval(connection.pingIntervalId);
        }
        if (connection.pongTimeoutId) {
            clearTimeout(connection.pongTimeoutId);
        }

        // Close WebSocket
        if (connection.ws && connection.ws.readyState === WebSocket.OPEN) {
            connection.ws.close(1000, 'Manual disconnect');
        }

        this.connections.delete(taskId);
        this._updateConnectionStatus(connection, 'disconnected');
    }

    /**
     * Get connection status
     */
    getConnectionStatus(taskId) {
        const connection = this.connections.get(taskId);
        return connection ? connection.status : 'disconnected';
    }

    /**
     * Check if connection is active
     */
    isConnected(taskId) {
        const connection = this.connections.get(taskId);
        return connection && connection.ws && connection.ws.readyState === WebSocket.OPEN;
    }

    /**
     * Send message through WebSocket
     */
    send(taskId, message) {
        const connection = this.connections.get(taskId);
        if (connection && connection.ws && connection.ws.readyState === WebSocket.OPEN) {
            try {
                connection.ws.send(JSON.stringify(message));
                return true;
            } catch (error) {
                console.error('Failed to send WebSocket message:', error);
                return false;
            }
        }
        return false;
    }

    /**
     * Force polling update for a specific task
     */
    async forcePoll(taskId) {
        const connection = this.connections.get(taskId);
        if (connection && connection.pollCallback) {
            try {
                await connection.pollCallback(taskId);
            } catch (error) {
                console.error('Polling error:', error);
            }
        }
    }

    /**
     * Get all active connections
     */
    getActiveConnections() {
        return Array.from(this.connections.keys());
    }

    /**
     * Cleanup all connections
     */
    disconnectAll() {
        const taskIds = Array.from(this.connections.keys());
        taskIds.forEach(taskId => this.disconnect(taskId));
    }

    // ===== PRIVATE METHODS =====

    /**
     * Create WebSocket connection
     */
    _createWebSocket(connection) {
        const { taskId, endpointType } = connection;
        
        // Get current origin and convert to WebSocket protocol
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        
        // ✅ UPDATED: No longer using localStorage token
        // Authentication now uses httpOnly cookies automatically
        
        // Generate WebSocket URL based on endpoint type
        let wsUrl;
        if (endpointType === 'qca-dataset') {
            wsUrl = `${protocol}//${host}/api/ws/qca-dataset/${taskId}`;
        } else if (endpointType === 'finetuning') {
            wsUrl = `${protocol}//${host}/api/ws/finetuning/${taskId}`;
        } else {
            // Default to evaluation endpoint
            wsUrl = `${protocol}//${host}/api/evaluation/ws/evaluation/${taskId}`;
        }

        console.log(`🔌 Creating WebSocket connection for task ${taskId} (cookie-based auth)`);
        console.log(`🔗 WebSocket URL: ${wsUrl}`);
        this._updateConnectionStatus(connection, 'connecting');

        try {
            connection.ws = new WebSocket(wsUrl);
            
            connection.ws.onopen = () => {
                console.log(`✅ WebSocket connected for task ${taskId}`);
                connection.reconnectAttempts = 0;
                this._updateConnectionStatus(connection, 'connected');
                this._startPing(connection);
            };

            connection.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    // Handle ping/pong for connection health
                    if (data.type === 'ping') {
                        this._sendPong(connection);
                        return;
                    } else if (data.type === 'pong') {
                        connection.lastPongTime = Date.now();
                        if (connection.pongTimeoutId) {
                            clearTimeout(connection.pongTimeoutId);
                            connection.pongTimeoutId = null;
                        }
                        return;
                    }

                    // Forward message to callback
                    connection.callbacks.onMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                    connection.callbacks.onError(error);
                }
            };

            connection.ws.onerror = (error) => {
                console.error(`❌ WebSocket error for task ${taskId}:`, error);
                this._updateConnectionStatus(connection, 'error');
                connection.callbacks.onError(error);
            };

            connection.ws.onclose = (event) => {
                console.log(`🔒 WebSocket closed for task ${taskId}:`, event.code, event.reason);
                this._stopPing(connection);
                
                // Handle authentication failures (code 1008)
                if (event.code === 1008) {
                    console.error('❌ WebSocket authentication failed:', event.reason);
                    this._updateConnectionStatus(connection, 'auth_failed');
                    connection.callbacks.onError(new Error(`Authentication failed: ${event.reason}`));
                    // Don't attempt to reconnect on auth failures, but start polling as fallback
                    connection.isManualClose = true;
                    
                    // Start polling as fallback for auth-failed connections
                    if (connection.enablePolling && connection.pollCallback && !connection.pollingIntervalId) {
                        console.log(`📊 Starting HTTP polling as fallback (auth failed) for task ${taskId}`);
                        this._startPolling(connection);
                    }
                }
                
                if (!connection.isManualClose) {
                    this._updateConnectionStatus(connection, 'disconnected');
                    this._scheduleReconnect(connection);
                } else {
                    this._updateConnectionStatus(connection, 'disconnected');
                }
                
                connection.callbacks.onClose(event);
            };

        } catch (error) {
            console.error(`Failed to create WebSocket for task ${taskId}:`, error);
            this._updateConnectionStatus(connection, 'error');
            this._scheduleReconnect(connection);
        }
    }

    /**
     * Schedule reconnection with exponential backoff
     */
    _scheduleReconnect(connection) {
        if (connection.isManualClose || connection.reconnectAttempts >= this.config.maxReconnectAttempts) {
            console.log(`🚫 Max reconnection attempts reached for task ${connection.taskId}`);
            this._updateConnectionStatus(connection, 'failed');
            
            // Start polling as fallback when WebSocket has completely failed
            if (connection.enablePolling && connection.pollCallback && !connection.pollingIntervalId) {
                console.log(`📊 Starting HTTP polling as fallback for task ${connection.taskId}`);
                this._startPolling(connection);
            }
            return;
        }

        const delay = Math.min(
            this.config.reconnectInterval * Math.pow(this.config.reconnectDecay, connection.reconnectAttempts),
            this.config.maxReconnectInterval
        );

        console.log(`🔄 Scheduling reconnection for task ${connection.taskId} in ${delay}ms (attempt ${connection.reconnectAttempts + 1})`);
        this._updateConnectionStatus(connection, 'reconnecting');
        
        // Start polling during reconnection attempts as temporary fallback
        if (connection.enablePolling && connection.pollCallback && !connection.pollingIntervalId) {
            console.log(`📊 Starting HTTP polling during reconnection for task ${connection.taskId}`);
            this._startPolling(connection);
        }

        connection.reconnectTimeoutId = setTimeout(() => {
            connection.reconnectAttempts++;
            this._createWebSocket(connection);
        }, delay);
    }

    /**
     * Start polling fallback
     */
    _startPolling(connection) {
        if (!connection.enablePolling || !connection.pollCallback) return;

        console.log(`📊 Starting polling for task ${connection.taskId}`);
        
        connection.pollingIntervalId = setInterval(async () => {
            try {
                await connection.pollCallback(connection.taskId);
            } catch (error) {
                console.error(`Polling error for task ${connection.taskId}:`, error);
            }
        }, this.config.pollingInterval);
    }

    /**
     * Start ping mechanism for connection health
     */
    _startPing(connection) {
        connection.pingIntervalId = setInterval(() => {
            if (connection.ws && connection.ws.readyState === WebSocket.OPEN) {
                this._sendPing(connection);
            }
        }, this.config.pingInterval);
    }

    /**
     * Stop ping mechanism
     */
    _stopPing(connection) {
        if (connection.pingIntervalId) {
            clearInterval(connection.pingIntervalId);
            connection.pingIntervalId = null;
        }
        if (connection.pongTimeoutId) {
            clearTimeout(connection.pongTimeoutId);
            connection.pongTimeoutId = null;
        }
    }

    /**
     * Send ping to check connection health
     */
    _sendPing(connection) {
        try {
            const pingMessage = {
                type: 'ping',
                timestamp: new Date().toISOString()
            };
            connection.ws.send(JSON.stringify(pingMessage));
            
            // Set timeout for pong response
            connection.pongTimeoutId = setTimeout(() => {
                console.warn(`🏓 Pong timeout for task ${connection.taskId}, connection may be dead`);
                if (connection.ws) {
                    connection.ws.close(3001, 'Ping timeout');
                }
            }, this.config.pongTimeout);
            
        } catch (error) {
            console.error(`Failed to send ping for task ${connection.taskId}:`, error);
        }
    }

    /**
     * Send pong response
     */
    _sendPong(connection) {
        try {
            const pongMessage = {
                type: 'pong',
                timestamp: new Date().toISOString()
            };
            connection.ws.send(JSON.stringify(pongMessage));
        } catch (error) {
            console.error(`Failed to send pong for task ${connection.taskId}:`, error);
        }
    }

    /**
     * Update connection status and notify callback
     */
    _updateConnectionStatus(connection, status) {
        const oldStatus = connection.status;
        connection.status = status;
        
        if (oldStatus !== status) {
            console.log(`🔄 Connection status changed for task ${connection.taskId}: ${oldStatus} → ${status}`);
            connection.callbacks.onStatusChange(status, oldStatus);
        }
    }
}

// Create singleton instance
const webSocketService = new WebSocketService();

export default webSocketService;

// Export connection status constants
export const CONNECTION_STATUS = {
    CONNECTING: 'connecting',
    CONNECTED: 'connected',
    DISCONNECTED: 'disconnected',
    RECONNECTING: 'reconnecting',
    ERROR: 'error',
    FAILED: 'failed'
};
