// Only the handleSubmit function needs updating

const handleSubmit = async (e) => {
  e.preventDefault();
  setLoginError('');
  
  // Basic validation
  if (!username.trim()) {
    setLoginError('Username is required');
    return;
  }
  
  if (!password) {
    setLoginError('Password is required');
    return;
  }
  
  // Use updated login API
  try {
    setLoading(true);
    console.log("Submitting login with:", { username });
    
    // Using fetch directly to avoid any middleware issues
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      mode: 'cors',
      credentials: 'omit', // Avoid CORS preflight complexities
      body: JSON.stringify({ 
        username, 
        password 
      })
    });
    
    console.log("Login response status:", response.status);
    
    if (!response.ok) {
      let errorMessage = 'Login failed';
      try {
        const errorData = await response.json();
        errorMessage = errorData.detail || errorMessage;
      } catch (e) {
        // Ignore JSON parsing error
      }
      
      setLoginError(errorMessage);
      setLoading(false);
      return;
    }
    
    const data = await response.json();
    console.log("Login successful, got token");
    
    // Store token and user info in local storage
    localStorage.setItem('token', data.access_token);
    localStorage.setItem('username', data.username);
    
    // Redirect to home page
    navigate('/');
  } catch (err) {
    console.error("Login error:", err);
    setLoginError(err.message || 'Failed to connect to server. Please try again.');
    setLoading(false);
  }
};
