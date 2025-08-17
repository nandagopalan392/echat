/**
 * React Hook for WebSocket connections with auto-reconnection and polling fallback
 * 
 * This hook provides a simple interface to use the WebSocketService in React components
 * with automatic cleanup and state management.
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import webSocketService, { CONNECTION_STATUS } from '../services/websocketService';
import { evaluationApi } from '../services/api';

export const useWebSocketConnection = (taskId, options = {}) => {
    const {
        onMessage = () => {},
        onError = () => {},
        onClose = () => {},
        enablePolling = true,
        autoConnect = true,
        pollCallback = null
    } = options;

    const [connectionStatus, setConnectionStatus] = useState(CONNECTION_STATUS.DISCONNECTED);
    const [lastMessage, setLastMessage] = useState(null);
    const [error, setError] = useState(null);
    const connectionRef = useRef(null);

    // Default poll callback that fetches task status
    const defaultPollCallback = useCallback(async (taskId) => {
        try {
            const status = await evaluationApi.getDetailedTaskStatus(taskId);
            
            // Simulate a WebSocket message format for consistency
            const simulatedMessage = {
                type: 'evaluation_update',
                status: status.status,
                data: status.data || {},
                timestamp: new Date().toISOString(),
                source: 'polling'
            };
            
            onMessage(simulatedMessage);
            return status;
        } catch (error) {
            console.error('Polling failed:', error);
            throw error;
        }
    }, [onMessage]);

    // Connection status change handler
    const handleStatusChange = useCallback((status, oldStatus) => {
        setConnectionStatus(status);
        
        // Clear error when connection is successful
        if (status === CONNECTION_STATUS.CONNECTED) {
            setError(null);
        }
    }, []);

    // Message handler
    const handleMessage = useCallback((message) => {
        setLastMessage(message);
        setError(null); // Clear error on successful message
        onMessage(message);
    }, [onMessage]);

    // Error handler
    const handleError = useCallback((error) => {
        setError(error);
        onError(error);
    }, [onError]);

    // Close handler
    const handleClose = useCallback((event) => {
        onClose(event);
    }, [onClose]);

    // Connect function
    const connect = useCallback(() => {
        if (!taskId) {
            console.warn('Cannot connect: taskId is required');
            return null;
        }

        console.log(`🔌 Connecting to WebSocket for task: ${taskId}`);

        const connection = webSocketService.connect(taskId, {
            onMessage: handleMessage,
            onError: handleError,
            onClose: handleClose,
            onStatusChange: handleStatusChange,
            pollCallback: pollCallback || defaultPollCallback,
            enablePolling
        });

        connectionRef.current = connection;
        return connection;
    }, [
        taskId, 
        handleMessage, 
        handleError, 
        handleClose, 
        handleStatusChange, 
        defaultPollCallback,
        pollCallback,
        enablePolling
    ]);

    // Disconnect function
    const disconnect = useCallback(() => {
        if (taskId) {
            console.log(`🔒 Disconnecting WebSocket for task: ${taskId}`);
            webSocketService.disconnect(taskId);
            connectionRef.current = null;
            setConnectionStatus(CONNECTION_STATUS.DISCONNECTED);
        }
    }, [taskId]);

    // Send message function
    const sendMessage = useCallback((message) => {
        if (taskId) {
            return webSocketService.send(taskId, message);
        }
        return false;
    }, [taskId]);

    // Force polling update
    const forcePoll = useCallback(async () => {
        if (taskId) {
            await webSocketService.forcePoll(taskId);
        }
    }, [taskId]);

    // Check if connected
    const isConnected = useCallback(() => {
        return taskId ? webSocketService.isConnected(taskId) : false;
    }, [taskId]);

    // Auto-connect on mount if enabled
    useEffect(() => {
        if (autoConnect && taskId) {
            connect();
        }

        // Cleanup on unmount
        return () => {
            if (taskId) {
                disconnect();
            }
        };
    }, [taskId, autoConnect, connect, disconnect]);

    return {
        // Connection control
        connect,
        disconnect,
        sendMessage,
        forcePoll,
        
        // Connection state
        connectionStatus,
        isConnected: isConnected(),
        lastMessage,
        error,
        
        // Connection reference (for advanced usage)
        connection: connectionRef.current
    };
};

export default useWebSocketConnection;
