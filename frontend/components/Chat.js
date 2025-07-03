import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import axios from 'axios';
import '../styles/Chat.css';

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const messagesEndRef = useRef(null);
  const { logout } = useAuth();
  const navigate = useNavigate();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    fetchSessions();
    scrollToBottom();
  }, [messages]);

  const fetchSessions = async () => {
    try {
      const response = await axios.get('/api/chat/sessions', {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
      });
      setSessions(response.data.sessions);
    } catch (error) {
      console.error('Failed to fetch sessions:', error);
    }
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    try {
      const response = await axios.post('/api/chat/send', 
        { content: input, session_id: currentSession },
        { headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }}
      );

      setMessages(prev => [...prev, 
        { content: input, isUser: true },
        { content: response.data.response, isUser: false }
      ]);
      setInput('');
      
      if (!currentSession) {
        setCurrentSession(response.data.session_id);
        fetchSessions();
      }
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post('/api/chat/upload', formData, {
        headers: { 
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'multipart/form-data'
        }
      });
      setMessages(prev => [...prev, {
        content: `File ${file.name} uploaded successfully`,
        isUser: false
      }]);
    } catch (error) {
      console.error('Failed to upload file:', error);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <div className="logo-container">
          <img src="/echat_logo.svg" alt="echat logo" className="logo" />
          <h1 className="app-title">echat</h1>
        </div>
        <button onClick={handleLogout} className="logout-btn">Logout</button>
      </header>

      <aside className="chat-sidebar">
        <button className="new-chat" onClick={() => setCurrentSession(null)}>
          + New Chat
        </button>
        
        <div className="file-upload">
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileUpload}
            id="file-upload"
          />
          <label htmlFor="file-upload">Upload PDF</label>
        </div>

        <div className="sessions-list">
          {sessions.map(session => (
            <button
              key={session.id}
              className={`session-btn ${currentSession === session.id ? 'active' : ''}`}
              onClick={() => setCurrentSession(session.id)}
            >
              📝 {session.topic}
            </button>
          ))}
        </div>
      </aside>

      <main className="chat-main">
        <div className="messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.isUser ? 'user' : 'assistant'}`}>
              {msg.content}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSend} className="input-container">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a message..."
            className="message-input"
          />
          <button type="submit" className="send-btn">Send</button>
        </form>
      </main>
    </div>
  );
};

export default Chat;
