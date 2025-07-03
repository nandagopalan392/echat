import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../styles/Login.css';

const Login = () => {
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    try {
      if (isRegister) {
        if (password !== confirmPassword) {
          setError('Passwords do not match');
          return;
        }
        await axios.post('/api/auth/register', { username, password });
        setIsRegister(false);
        return;
      }

      const response = await axios.post('/api/auth/login', { username, password });
      localStorage.setItem('token', response.data.access_token);
      navigate('/');
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred');
    }
  };

  return (
    <div className="login-container">
      <div className="logo-container">
        <img src="/echat_logo.svg" alt="echat logo" className="logo" />
        <h1 className="app-title">echat</h1>
      </div>
      
      <form onSubmit={handleSubmit} className="login-form">
        <h2>{isRegister ? 'Create Account' : 'Login'}</h2>
        
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        
        {isRegister && (
          <input
            type="password"
            placeholder="Confirm Password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
          />
        )}
        
        {error && <div className="error">{error}</div>}
        
        <button type="submit">
          {isRegister ? 'Register' : 'Login'}
        </button>
        
        <button
          type="button"
          className="switch-btn"
          onClick={() => setIsRegister(!isRegister)}
        >
          {isRegister ? 'Back to Login' : 'Create New Account'}
        </button>
      </form>
    </div>
  );
};

export default Login;
