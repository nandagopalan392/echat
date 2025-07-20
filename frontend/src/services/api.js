// ========================================
// DEBUG CONFIGURATION
// ========================================
// Set DEBUG_MODE to true to enable console logs for debugging
// Set to false in production to reduce console noise
const DEBUG_MODE = false;

// ========================================

// Debug helper function
const debugLog = (message, ...args) => {
  if (DEBUG_MODE) {
    console.log(message, ...args);
  }
};

const API_BASE_URL = ''; // Use relative URLs to avoid CORS issues

const getAuthHeader = () => {
    const token = localStorage.getItem('token');
    return token ? { 'Authorization': `Bearer ${token}` } : {};
};

export const api = {
    // Generic API call method
    call: async (endpoint, options = {}) => {
        try {
            const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader(),
                    ...options.headers
                },
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`API call failed: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API call error:', error);
            throw error;
        }
    },

    // RLHF feedback
    submitRLHFFeedback: async (sessionId, chosenIndex) => {
        debugLog('=== RLHF FEEDBACK SUBMISSION ===');
        debugLog('Session ID:', sessionId);
        debugLog('Chosen Index:', chosenIndex);
        debugLog('API Base URL:', API_BASE_URL);
        
        try {
            const payload = {
                session_id: sessionId,
                chosen_index: chosenIndex
            };
            
            debugLog('Payload:', payload);
            debugLog('Auth Header:', getAuthHeader());
            
            const response = await fetch(`${API_BASE_URL}/api/chat/rlhf-feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify(payload)
            });
            
            debugLog('Response status:', response.status);
            debugLog('Response ok:', response.ok);
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Response error text:', errorText);
                throw new Error(`Failed to submit RLHF feedback: ${response.status} - ${errorText}`);
            }
            
            const result = await response.json();
            debugLog('RLHF feedback success:', result);
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
            debugLog('Login attempt:', { username });
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
            debugLog('Login response:', data);

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
            debugLog('Register attempt:', { username });
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
            debugLog('Register response:', data);

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

    // Upload file with chunking configuration
    uploadFileWithChunking: async (file, chunkingConfig, isFolder = false, folderPath = "", onProgress) => {
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('is_folder', isFolder ? 'true' : 'false');
            formData.append('folder_path', folderPath || '');
            
            // Add chunking configuration parameters
            formData.append('chunking_method', chunkingConfig.method || 'auto');
            formData.append('chunk_token_num', chunkingConfig.chunk_token_num || 1000);
            formData.append('chunk_overlap', chunkingConfig.chunk_overlap || 200);
            formData.append('delimiter', chunkingConfig.delimiter || "\\n\\n|\\n|\\.|\\!|\\?");
            formData.append('max_token', chunkingConfig.max_token || 4096);
            formData.append('layout_recognize', chunkingConfig.layout_recognize || 'auto');
            formData.append('preserve_formatting', chunkingConfig.preserve_formatting || true);
            formData.append('extract_tables', chunkingConfig.extract_tables || true);
            formData.append('extract_images', chunkingConfig.extract_images || false);

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
    },

    // New function to get general user statistics
    getUserStatsGeneral: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/users/stats`, {
                headers: {
                    ...getAuthHeader(),
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch general stats');
            }
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching general stats:', error);
            throw error;
        }
    },

    // File management
    listFiles: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/documents`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader()
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch files');
            }
            
            const data = await response.json();
            return { files: data.documents || [] };
        } catch (error) {
            console.error('List files error:', error);
            throw error;
        }
    },

    deleteFile: async (filename) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/files/${encodeURIComponent(filename)}`, {
                method: 'DELETE',
                headers: {
                    ...getAuthHeader()
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to delete file');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Delete file error:', error);
            throw error;
        }
    },

    getDocuments: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/documents`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader()
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to get documents');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Get documents error:', error);
            throw error;
        }
    },

    getDocumentChunks: async (filename) => {
        try {
            // Extract just the filename from the path (e.g., "IMX662/IMX662_AppNote/file.pdf" → "file.pdf")
            const cleanFilename = filename.split('/').pop() || filename;
            const response = await fetch(`${API_BASE_URL}/api/files/${encodeURIComponent(cleanFilename)}/chunks`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader()
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to get document chunks');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Get document chunks error:', error);
            throw error;
        }
    },

    getDocumentPreview: async (documentId) => {
        try {
            // Add cache-busting parameter
            const cacheBuster = new Date().getTime();
            const response = await fetch(`${API_BASE_URL}/api/documents/${documentId}/preview?t=${cacheBuster}`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader()
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to get document preview');
            }
            
            const result = await response.json();
            console.log('DEBUG: API getDocumentPreview result:', result);
            console.log('DEBUG: API result type:', result.type);
            console.log('DEBUG: API result has_html:', result.has_html);
            if (result.slides && result.slides.length > 0) {
                console.log('DEBUG: API first slide structure:', Object.keys(result.slides[0]));
                console.log('DEBUG: API first slide data:', result.slides[0]);
                console.log('DEBUG: API first slide format:', result.slides[0].format);
                console.log('DEBUG: API first slide html_content length:', result.slides[0].html_content ? result.slides[0].html_content.length : 'none');
            }
            return result;
        } catch (error) {
            console.error('Get document preview error:', error);
            throw error;
        }
    },

    getCollectionDebugInfo: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/debug/collection-info`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader()
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to get collection debug info');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Get collection debug info error:', error);
            throw error;
        }
    },

    // Model settings
    getModelSettings: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/settings`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader()
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch model settings');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Get model settings error:', error);
            throw error;
        }
    },

    updateModelSettings: async (settings) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/settings`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify(settings)
            });
            
            if (!response.ok) {
                throw new Error('Failed to update model settings');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Update model settings error:', error);
            throw error;
        }
    },

    getAvailableModels: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/available`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader()
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch available models');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Get available models error:', error);
            throw error;
        }
    },

    // User management
    getUserProfile: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/users/profile`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader()
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch user profile');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Get user profile error:', error);
            throw error;
        }
    },

    getUsers: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/admin/users`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader()
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch users');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Get users error:', error);
            throw error;
        }
    },

    deleteUser: async (userId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/admin/users/${userId}`, {
                method: 'DELETE',
                headers: {
                    ...getAuthHeader()
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to delete user');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Delete user error:', error);
            throw error;
        }
    },

    updateUserRole: async (userId, newRole) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/admin/users/${userId}/role`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify({ role: newRole })
            });
            
            if (!response.ok) {
                throw new Error('Failed to update user role');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Update user role error:', error);
            throw error;
        }
    },

    getUserActivities: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/users/activities`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader()
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch user activities');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Get user activities error:', error);
            throw error;
        }
    },

    // Vector store and embedding management
    getVectorStoreStats: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/vector-store/stats`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader()
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch vector store stats');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Get vector store stats error:', error);
            throw error;
        }
    },

    reingestDocuments: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/documents/reingest`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to reingest documents');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Reingest documents error:', error);
            throw error;
        }
    },

    clearVectorStore: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/vectorstore/clear`, {
                method: 'DELETE',
                headers: {
                    ...getAuthHeader()
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to clear vector store');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Clear vector store error:', error);
            throw error;
        }
    },

    // User management functions
    getUserProfile: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/users/profile`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader(),
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch user profile');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Get user profile error:', error);
            throw error;
        }
    },

    getUsers: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/admin/users`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader(),
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch users');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Get users error:', error);
            throw error;
        }
    },

    createUser: async (userData) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/admin/add-user`, {
                method: 'POST',
                headers: {
                    ...getAuthHeader(),
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(userData)
            });
            
            if (!response.ok) {
                throw new Error('Failed to create user');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Create user error:', error);
            throw error;
        }
    },

    deleteUser: async (userId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/users/${userId}`, {
                method: 'DELETE',
                headers: {
                    ...getAuthHeader(),
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to delete user');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Delete user error:', error);
            throw error;
        }
    },

    updateUserRole: async (userId, role) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/users/${userId}/role`, {
                method: 'PUT',
                headers: {
                    ...getAuthHeader(),
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ role })
            });
            
            if (!response.ok) {
                throw new Error('Failed to update user role');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Update user role error:', error);
            throw error;
        }
    },

    getUserActivities: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/users/activities`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader(),
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch user activities');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Get user activities error:', error);
            throw error;
        }
    },

    // Note: getUserStats already exists above

    // Embedding model management (deprecated - use get('/api/models/available') instead)
    getEmbeddingModels: async () => {
        console.warn('getEmbeddingModels is deprecated. Use get("/api/models/available") instead.');
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/available`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader()
                }
            });
            
            if (!response.ok) {
                throw new Error('Failed to fetch models');
            }
            
            const data = await response.json();
            return {
                models: data.embedding_models || []
            };
        } catch (error) {
            console.error('Get embedding models error:', error);
            throw error;
        }
    },

    switchEmbeddingModel: async (modelName) => {
        console.warn('switchEmbeddingModel is deprecated. Use post("/api/models/settings") instead.');
        try {
            // Get current models first
            const currentResponse = await fetch(`${API_BASE_URL}/api/models/current`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader()
                }
            });
            
            if (!currentResponse.ok) {
                throw new Error('Failed to get current models');
            }
            
            const currentModels = await currentResponse.json();
            
            // Update with new embedding model
            const response = await fetch(`${API_BASE_URL}/api/models/settings`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify({
                    llm: currentModels.llm,
                    embedding: modelName
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to switch embedding model');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Switch embedding model error:', error);
            throw error;
        }
    },

    // Get chunking methods and configurations
    getChunkingMethods: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/chunking/methods`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader(),
                },
            });

            if (!response.ok) {
                throw new Error('Failed to get chunking methods');
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting chunking methods:', error);
            throw error;
        }
    },

    // Get chunking configuration for a method
    getChunkingConfig: async (method) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/chunking/config/${method}`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader(),
                },
            });

            if (!response.ok) {
                throw new Error('Failed to get chunking config');
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting chunking config:', error);
            throw error;
        }
    },

    // Update chunking configuration
    updateChunkingConfig: async (method, config) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/chunking/config/${method}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader(),
                },
                body: JSON.stringify(config),
            });

            if (!response.ok) {
                throw new Error('Failed to update chunking config');
            }

            return await response.json();
        } catch (error) {
            console.error('Error updating chunking config:', error);
            throw error;
        }
    },

    // Get optimal chunking method for file extension
    getOptimalChunkingMethod: async (fileExtension) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/chunking/optimal/${fileExtension}`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader(),
                },
            });

            if (!response.ok) {
                throw new Error('Failed to get optimal chunking method');
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting optimal chunking method:', error);
            throw error;
        }
    },
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

// Check if a file exists by comparing filename and hash
export const checkFileExists = async (filename, hash) => {
    try {
        const response = await fetch(`/api/files/check-duplicate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...getAuthHeader()
            },
            body: JSON.stringify({ filename, hash })
        });

        if (!response.ok) {
            if (response.status === 404) {
                return null; // File doesn't exist
            }
            throw new Error(`Failed to check file existence: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error checking file existence:', error);
        throw error;
    }
};
