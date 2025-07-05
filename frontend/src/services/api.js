const API_BASE_URL = 'http://localhost:8000';

const getAuthHeader = () => {
    const token = localStorage.getItem('token');
    return token ? { 'Authorization': `Bearer ${token}` } : {};
};

export const api = {
    // RLHF feedback
    submitRLHFFeedback: async (sessionId, chosenIndex) => {
        console.log('=== RLHF FEEDBACK SUBMISSION ===');
        console.log('Session ID:', sessionId);
        console.log('Chosen Index:', chosenIndex);
        console.log('API Base URL:', API_BASE_URL);
        
        try {
            const payload = {
                session_id: sessionId,
                chosen_index: chosenIndex
            };
            
            console.log('Payload:', payload);
            console.log('Auth Header:', getAuthHeader());
            
            const response = await fetch(`${API_BASE_URL}/api/chat/rlhf-feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify(payload)
            });
            
            console.log('Response status:', response.status);
            console.log('Response ok:', response.ok);
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Response error text:', errorText);
                throw new Error(`Failed to submit RLHF feedback: ${response.status} - ${errorText}`);
            }
            
            const result = await response.json();
            console.log('RLHF feedback success:', result);
            return result;
        } catch (error) {
            console.error('RLHF feedback error:', error);
            throw error;
        }
    },

    // Update message content (for RLHF response selection)
    updateMessage: async (sessionId, content) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/chat/message/update`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    content: content
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to update message');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Update message error:', error);
            throw error;
        }
    },

    // Auth endpoints
    login: async (username, password) => {
        try {
            console.log('Login attempt:', { username });
            const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, password }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Login failed');
            }

            const data = await response.json();
            console.log('Login response:', data);

            if (data.access_token) {
                localStorage.setItem('token', data.access_token);
                localStorage.setItem('username', data.username);
            }
            return data;
        } catch (error) {
            console.error('Login error:', error);
            throw error;
        }
    },

    // Add register endpoint
    register: async (username, password) => {
        try {
            console.log('Register attempt:', { username });
            const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                mode: 'cors',
                body: JSON.stringify({ username, password }),
            });

            const data = await response.json();
            console.log('Register response:', data);

            if (!response.ok) {
                throw new Error(data.detail || 'Registration failed');
            }

            return data;
        } catch (error) {
            console.error('Registration error:', error);
            throw error;
        }
    },

    // Upload PDF document
    uploadPDF: async (file) => {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${API_BASE_URL}/api/chat/upload`, {
                method: 'POST',
                headers: {
                    ...getAuthHeader(),
                },
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Upload failed');
            }

            return data;
        } catch (error) {
            console.error('Upload error:', error);
            throw error;
        }
    },

    // Upload file
    uploadFile: async (file) => {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${API_BASE_URL}/api/admin/upload`, {
                method: 'POST',
                headers: {
                    ...getAuthHeader(),
                },
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Upload failed');
            }

            return data;
        } catch (error) {
            console.error('Upload error:', error);
            throw error;
        }
    },

    // Upload file with progress tracking
    uploadFileWithProgress: async (file, isFolder = false, folderPath = "", onProgress) => {
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('is_folder', isFolder ? 'true' : 'false');
            formData.append('folder_path', folderPath || '');

            const xhr = new XMLHttpRequest();
            
            const promise = new Promise((resolve, reject) => {
                let completed = false;

                xhr.upload.addEventListener('progress', (event) => {
                    if (event.lengthComputable && onProgress && !completed) {
                        const percentCompleted = Math.round((event.loaded * 100) / event.total);
                        onProgress(Math.min(percentCompleted, 90)); // Cap at 90% for upload
                    }
                });

                xhr.addEventListener('load', () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        const response = JSON.parse(xhr.response);
                        onProgress(100); // Set to 100% immediately when done
                        completed = true;
                        resolve(response);
                    } else {
                        reject(new Error(xhr.response || 'Upload failed'));
                    }
                });

                xhr.addEventListener('error', () => {
                    reject(new Error('Network error occurred'));
                });
            });

            xhr.open('POST', `${API_BASE_URL}/api/admin/upload`);
            const token = localStorage.getItem('token');
            if (token) {
                xhr.setRequestHeader('Authorization', `Bearer ${token}`);
            }
            xhr.send(formData);

            return promise;
        } catch (error) {
            console.error('Upload error:', error);
            throw error;
        }
    },

    // Send chat message
    sendMessage: async (content, sessionId = null) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/chat/send`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify({ content, session_id: sessionId })
            });

            if (!response.ok) {
                throw new Error('Failed to send message');
            }

            const data = await response.json();
            
            // Handle the new response structure for RLHF
            return {
                status: data.status || 'success',
                response: data.content || data.response,
                session_id: data.session_id,
                processing_time: data.processing_time,
                message_received: data.message_received,
                // New RLHF fields
                response_options: data.response_options,
                rlhf_enabled: data.rlhf_enabled,
                is_final: data.is_final,
                thinking_included: data.thinking_included,
                full_response: data.full_response
            };
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },

    // Get chat sessions
    getSessions: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/chat/sessions`, {
                headers: getAuthHeader(),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Get sessions failed');
            }

            return data;
        } catch (error) {
            console.error('Get sessions error:', error);
            throw error;
        }
    },

    // Create new chat session
    createSession: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/sessions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader(),
                },
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Create session failed');
            }

            return data;
        } catch (error) {
            console.error('Create session error:', error);
            throw error;
        }
    },

    // Delete chat session
    deleteSession: async (sessionId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}`, {
                method: 'DELETE',
                headers: getAuthHeader(),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Delete session failed');
            }

            return data;
        } catch (error) {
            console.error('Delete session error:', error);
            throw error;
        }
    },

    // Get chat history for a session
    getChatHistory: async (sessionId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/history`, {
                headers: getAuthHeader(),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Get chat history failed');
            }

            return data;
        } catch (error) {
            console.error('Get chat history error:', error);
            throw error;
        }
    },

    // Get chat messages for a session
    getChatMessages: async (sessionId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/chat/sessions/${sessionId}/messages`, {
                headers: {
                    ...getAuthHeader(),
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch chat messages');
            }
            
            const data = await response.json();
            return {
                messages: data.messages.map((msg, index) => ({
                    id: `${sessionId}-${index}`,
                    content: msg.content,
                    isUser: msg.isUser,
                    timestamp: msg.timestamp
                })),
                session_id: sessionId
            };
        } catch (error) {
            console.error('Error fetching chat messages:', error);
            throw error;
        }
    },

    // Error handler wrapper
    handleError: async (promise) => {
        try {
            return await promise;
        } catch (error) {
            console.error('API Error:', error);
            throw new Error('Failed to fetch data from the server');
        }
    },

    // Admin endpoints
    get: async (url) => {
        try {
            const response = await fetch(`${API_BASE_URL}${url}`, {
                headers: {
                    ...getAuthHeader(),
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            return response.json();
        } catch (error) {
            console.error('API Get error:', error);
            throw error;
        }
    },
    
    post: async (url, data) => {
        try {
            const response = await fetch(`${API_BASE_URL}${url}`, {
                method: 'POST',
                headers: {
                    ...getAuthHeader(),
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            return response.json();
        } catch (error) {
            console.error('API Post error:', error);
            throw error;
        }
    },

    getUserStats: async (username) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/admin/user-stats/${username}`, {
                headers: {
                    ...getAuthHeader(),
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch user stats');
            }
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching user stats:', error);
            throw error;
        }
    }
};

export default api;

export const sendMessage = async (content, sessionId = null) => {
    const token = localStorage.getItem('token');
    if (!token) throw new Error('No authentication token');

    const response = await fetch('/api/chat/send', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ content, session_id: sessionId })
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to send message');
    }

    return await response.json();
};
