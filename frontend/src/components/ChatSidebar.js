import React, { useEffect, useState } from 'react';
import { api } from '../services/api';

const ChatSidebar = ({ onSelectSession, currentSessionId }) => {
    const [sessions, setSessions] = useState([]);

    useEffect(() => {
        loadSessions();
    }, []);

    const loadSessions = async () => {
        try {
            const response = await api.getSessions();
            setSessions(response.sessions || []);
        } catch (error) {
            console.error('Error loading sessions:', error);
        }
    };

    const handleSessionClick = (sessionId) => {
        if (sessionId === currentSessionId) {
            // If clicking the same session, force a refresh
            onSelectSession(null);  // Clear first
            setTimeout(() => onSelectSession(sessionId), 0);  // Then reload
        } else {
            // Different session, normal switch
            onSelectSession(sessionId);
        }
    };

    return (
        <div className="w-64 bg-white border-r h-full overflow-y-auto">
            <div className="p-4">
                <h2 className="text-lg font-semibold text-gray-700 mb-4">Chats</h2>
                <div className="space-y-1">
                    {sessions.map((session) => (
                        <div
                            key={session.id}
                            onClick={() => handleSessionClick(session.id)}
                            className={`w-full px-3 py-2 text-left rounded-lg hover:bg-gray-100 ${
                                currentSessionId === session.id ? 'bg-indigo-50 text-indigo-600' : 'text-gray-700'
                            }`}
                        >
                            <span className="text-sm font-medium">
                                {session.topic || 'New Chat'}
                            </span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default ChatSidebar;
