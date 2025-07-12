// Create or update the API client

const API_URL = ''; // Use relative URLs to avoid CORS issues

// Function to handle login with detailed error reporting
export const login = async (credentials) => {
  try {
    // Use direct fetch for login instead of axios
    const response = await fetch(`${API_URL}/api/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      // Important: Set mode to 'cors' and referrer policy
      mode: 'cors',
      referrerPolicy: 'no-referrer',
      body: JSON.stringify({
        username: credentials.username,
        password: credentials.password
      })
    });
    
    // Handle errors
    if (!response.ok) {
      // Try to extract error message
      try {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Login failed');
      } catch (e) {
        throw new Error(`Login failed with status: ${response.status}`);
      }
    }
    
    // Parse successful response
    const data = await response.json();
    
    // Store token in localStorage
    localStorage.setItem('token', data.access_token);
    localStorage.setItem('username', data.username);
    
    return data;
  } catch (error) {
    console.error('Login error:', error);
    throw error;
  }
};

// Function to logout
export const logout = () => {
  localStorage.removeItem('token');
  localStorage.removeItem('username');
  window.location.href = '/login';
};

// Function to check if user is authenticated
export const isAuthenticated = () => {
  const token = localStorage.getItem('token');
  return !!token; // Return true if token exists
};

// Function to get current user
export const getCurrentUser = () => {
  return localStorage.getItem('username');
};

const sendMessage = async (message, token) => {
    const response = await fetch(`${API_URL}/chat/send`, {
        // ...existing code...
    });
    // ...existing code...
};

const uploadFile = async (file, token, onProgress) => {
    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_URL}/chat/upload`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            },
            body: formData
        });

        const data = await response.json();
        
        if (response.ok) {
            // Start progress monitoring
            const eventSource = new EventSource(`${API_URL}/upload-progress/${data.file_id}`);
            
            eventSource.onmessage = (event) => {
                const progress = JSON.parse(event.data);
                if (onProgress) {
                    onProgress(progress.progress);
                }
                if (progress.progress >= 100) {
                    eventSource.close();
                }
            };

            eventSource.onerror = () => {
                eventSource.close();
            };

            return data;
        } else {
            throw new Error(data.detail || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        throw error;
    }
};

// ...existing code...
