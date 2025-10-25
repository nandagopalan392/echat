import React, { createContext, useContext, useState, useEffect } from 'react';
import { authService } from '../src/services/authService';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userInfo, setUserInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Initialize auth service
    authService.initialize();
    
    // Check authentication status
    const authenticated = authService.isAuthenticated();
    setIsAuthenticated(authenticated);
    
    if (authenticated) {
      setUserInfo(authService.getUserInfo());
    }
    
    setLoading(false);
  }, []);

  const login = async (username, password) => {
    try {
      const data = await authService.login(username, password);
      setIsAuthenticated(true);
      setUserInfo({
        username: data.username,
        is_admin: data.is_admin,
        role: data.role
      });
      return data;
    } catch (error) {
      throw error;
    }
  };

  const logout = async () => {
    await authService.logout();
    setIsAuthenticated(false);
    setUserInfo(null);
  };

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <AuthContext.Provider value={{ 
      isAuthenticated, 
      userInfo,
      login, 
      logout 
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);

