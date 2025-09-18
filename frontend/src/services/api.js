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
    // Get chat history for a session
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

    deleteFile: async (fileId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/documents/${encodeURIComponent(fileId)}`, {
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

    reingestSpecificDocuments: async (reingestionData) => {
        try {
            // reingestionData is an array of {document_id, chunking_method, chunking_config}
            const requestBody = {
                documents: reingestionData
            };
            
            const response = await fetch(`${API_BASE_URL}/api/documents/reingest-specific`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify(requestBody)
            });
            
            if (!response.ok) {
                throw new Error('Failed to reingest specific documents');
            }
            
            return await response.json();
        } catch (error) {
            console.error('Reingest specific documents error:', error);
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

    // Retrieval Configuration API methods
    getRetrievalConfig: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/retrieval/config`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader(),
                },
            });

            if (!response.ok) {
                throw new Error('Failed to get retrieval config');
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting retrieval config:', error);
            throw error;
        }
    },

    // Update retrieval configuration
    updateRetrievalConfig: async (config) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/retrieval/config`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader(),
                },
                body: JSON.stringify(config),
            });

            if (!response.ok) {
                throw new Error('Failed to update retrieval config');
            }

            return await response.json();
        } catch (error) {
            console.error('Error updating retrieval config:', error);
            throw error;
        }
    },

    // Get available reranker models
    getRerankerModels: async (provider = null) => {
        try {
            let url = `${API_BASE_URL}/api/retrieval/reranker-models`;
            if (provider) {
                url += `?provider=${encodeURIComponent(provider)}`;
            }
            
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    ...getAuthHeader(),
                },
            });

            if (!response.ok) {
                throw new Error('Failed to get reranker models');
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting reranker models:', error);
            throw error;
        }
    },

    // Get reranker model download status
    getRerankerDownloadStatus: async (modelName) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/retrieval/reranker-download-status?model_name=${encodeURIComponent(modelName)}`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader(),
                },
            });

            if (!response.ok) {
                throw new Error('Failed to get download status');
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting download status:', error);
            throw error;
        }
    },

    // ========================================
    // FINETUNING API ENDPOINTS
    // ========================================
    
    // Create a new finetuning experiment
    createFineTuningExperiment: async (formData) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/finetuning/experiments`, {
                method: 'POST',
                headers: {
                    ...getAuthHeader()
                },
                body: formData  // FormData for file upload
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(errorData.detail || 'Failed to create experiment');
            }

            return await response.json();
        } catch (error) {
            console.error('Error creating finetuning experiment:', error);
            throw error;
        }
    },

    // Get all finetuning experiments
    getFineTuningExperiments: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/finetuning/experiments`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error('Failed to fetch experiments');
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching finetuning experiments:', error);
            throw error;
        }
    },

    // Get a specific finetuning experiment
    getFineTuningExperiment: async (experimentId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/finetuning/experiments/${experimentId}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error('Failed to fetch experiment');
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching finetuning experiment:', error);
            throw error;
        }
    },

    // Update finetuning experiment
    updateFineTuningExperiment: async (experimentId, updates) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/finetuning/experiments/${experimentId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify(updates)
            });

            if (!response.ok) {
                throw new Error('Failed to update experiment');
            }

            return await response.json();
        } catch (error) {
            console.error('Error updating finetuning experiment:', error);
            throw error;
        }
    },

    // Delete finetuning experiment
    deleteFineTuningExperiment: async (experimentId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/finetuning/experiments/${experimentId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error('Failed to delete experiment');
            }

            return await response.json();
        } catch (error) {
            console.error('Error deleting finetuning experiment:', error);
            throw error;
        }
    },

    // Start training
    startFineTuningTraining: async (experimentId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/finetuning/experiments/${experimentId}/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error('Failed to start training');
            }

            return await response.json();
        } catch (error) {
            console.error('Error starting finetuning training:', error);
            throw error;
        }
    },

    // Stop training
    stopFineTuningTraining: async (experimentId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/finetuning/experiments/${experimentId}/stop`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error('Failed to stop training');
            }

            return await response.json();
        } catch (error) {
            console.error('Error stopping finetuning training:', error);
            throw error;
        }
    },

    // Get training logs
    getFineTuningLogs: async (experimentId, offset = 0, limit = 100) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/finetuning/experiments/${experimentId}/logs?offset=${offset}&limit=${limit}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error('Failed to fetch training logs');
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching finetuning logs:', error);
            throw error;
        }
    },

    // Get training metrics
    getFineTuningMetrics: async (experimentId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/finetuning/experiments/${experimentId}/metrics`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error('Failed to fetch training metrics');
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching finetuning metrics:', error);
            throw error;
        }
    },

    // Get available HuggingFace models
    getAvailableHFModels: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/finetuning/models`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error('Failed to fetch available models');
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching available HF models:', error);
            throw error;
        }
    },

    // Validate dataset
    validateFineTuningDataset: async (file) => {
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${API_BASE_URL}/api/finetuning/validate-dataset`, {
                method: 'POST',
                headers: {
                    ...getAuthHeader()
                },
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Validation failed' }));
                throw new Error(errorData.detail || 'Dataset validation failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Error validating dataset:', error);
            throw error;
        }
    },

    // Get finetuning datasets
    getFineTuningDatasets: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/finetuning/datasets`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error('Failed to fetch datasets');
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching finetuning datasets:', error);
            throw error;
        }
    },

    // Upload finetuning dataset
    uploadFineTuningDataset: async (file, name, description = '') => {
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('name', name);
            formData.append('description', description);

            const response = await fetch(`${API_BASE_URL}/api/finetuning/upload-dataset`, {
                method: 'POST',
                headers: {
                    ...getAuthHeader()
                },
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Upload failed' }));
                throw new Error(errorData.detail || 'Dataset upload failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Error uploading dataset:', error);
            throw error;
        }
    },

    // Create finetuning dataset from documents
    createFinetuningDataset: async (datasetConfig) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/finetuning/datasets/create`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify(datasetConfig)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error creating dataset:', error);
            throw error;
        }
    },

    // Convert evaluation dataset to finetuning format
    convertEvaluationDataset: async (evaluationDatasetId, name, description = '') => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/finetuning/datasets/convert`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify({
                    evaluation_dataset_id: evaluationDatasetId,
                    name: name,
                    description: description
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Conversion failed' }));
                throw new Error(errorData.detail || 'Dataset conversion failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Error converting evaluation dataset:', error);
            throw error;
        }
    },

    // Get dataset details
    getFineTuningDatasetDetails: async (datasetId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/finetuning/datasets/${datasetId}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error('Failed to fetch dataset details');
            }

            return await response.json();
        } catch (error) {
            console.error('Error fetching dataset details:', error);
            throw error;
        }
    },

    // Download dataset
    downloadFineTuningDataset: async (datasetId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/finetuning/datasets/${datasetId}/download`, {
                method: 'GET',
                headers: {
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error('Failed to download dataset');
            }

            // Return blob for download
            const blob = await response.blob();
            return blob;
        } catch (error) {
            console.error('Error downloading dataset:', error);
            throw error;
        }
    },

    // Delete finetuning dataset
    deleteFineTuningDataset: async (datasetId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/finetuning/datasets/${datasetId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error('Failed to delete dataset');
            }

            return await response.json();
        } catch (error) {
            console.error('Error deleting finetuning dataset:', error);
            throw error;
        }
    },

    // Create WebSocket connection for training progress
    createFineTuningWebSocket: (experimentId, onMessage, onError, onClose) => {
        // Build ws/wss URL and correct backend path: /api/ws/finetuning/{experiment_id}
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
            const host = window.location.host; // includes hostname:port
            const wsUrl = `${protocol}://${host}/api/ws/finetuning/${experimentId}`;
            const ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                // Optional: can log or send a ping if needed
                // console.debug('WebSocket connected to', wsUrl);
            };
        
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    onMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                    onError(error);
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                onError(error);
            };
            
            ws.onclose = () => {
                console.log('WebSocket connection closed');
                onClose();
            };
            
            return ws;
        } catch (e) {
            console.error('Failed to create WebSocket:', e);
            // Surface error to caller
            onError(e);
            return null;
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

        const result = await response.json();
        
        // Return the existing file info only if it actually exists
        if (result.exists === true) {
            return result.existing_file;
        } else {
            return null; // No duplicate found
        }
    } catch (error) {
        console.error('Error checking file existence:', error);
        throw error;
    }
};

// ========================================
// EVALUATION API ENDPOINTS
// ========================================

export const evaluationApi = {
    // ========================================
    // BACKGROUND EVALUATION API ENDPOINTS
    // ========================================
    
    // Start async evaluation
    startAsyncEvaluation: async (evaluationData) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/evaluation/evaluate/async`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify(evaluationData)
            });

            if (!response.ok) {
                throw new Error(`Failed to start async evaluation: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error starting async evaluation:', error);
            throw error;
        }
    },

    // Start dataset evaluation (new method)
    startDatasetEvaluation: async (datasetEvaluationData) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/evaluation/evaluate/dataset`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify(datasetEvaluationData)
            });

            if (!response.ok) {
                throw new Error(`Failed to start dataset evaluation: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error starting dataset evaluation:', error);
            throw error;
        }
    },

    // Start batch evaluation
    startBatchEvaluation: async (batchData) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/evaluation/evaluate/batch`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify(batchData)
            });

            if (!response.ok) {
                throw new Error(`Failed to start batch evaluation: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error starting batch evaluation:', error);
            throw error;
        }
    },

    // Get task status
    getTaskStatus: async (taskId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/evaluation/evaluate/status/${taskId}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to get task status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting task status:', error);
            throw error;
        }
    },

    // Cancel evaluation task
    cancelTask: async (taskId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/evaluation/evaluate/task/${taskId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to cancel task: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error canceling task:', error);
            throw error;
        }
    },

    // Get queue status
    getQueueStatus: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/evaluation/evaluate/queue/status`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to get queue status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting queue status:', error);
            throw error;
        }
    },

    // Get recent evaluation results
    getRecentResults: async (limit = 10, taskType = null) => {
        console.log('🌐 [DEBUG] evaluationApi.getRecentResults() called with limit:', limit, 'taskType:', taskType);
        try {
            let url = `${API_BASE_URL}/api/evaluation/evaluate/results/recent?limit=${limit}`;
            if (taskType) {
                url += `&task_type=${taskType}`;
            }
            
            console.log('🌐 [DEBUG] Making fetch request to:', url);
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            console.log('🌐 [DEBUG] Response status:', response.status, response.ok);
            if (!response.ok) {
                throw new Error(`Failed to get recent results: ${response.status}`);
            }

            const data = await response.json();
            console.log('✅ [DEBUG] evaluationApi.getRecentResults() successful, data:', data);
            return data;
        } catch (error) {
            console.error('❌ [DEBUG] Error in evaluationApi.getRecentResults():', error);
            throw error;
        }
    },

    // Get detailed task status (for polling fallback)
    getDetailedTaskStatus: async (taskId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/evaluation/status/${taskId}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to get detailed task status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting detailed task status:', error);
            throw error;
        }
    },

    // Create WebSocket connection for real-time updates (Legacy - use WebSocketService instead)
    createWebSocketConnection: (taskId, onMessage, onError, onClose) => {
        const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/api/evaluation/ws/evaluation/${taskId}`;
        const ws = new WebSocket(wsUrl);
        
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                onMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
                onError(error);
            }
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            onError(error);
        };
        
        ws.onclose = () => {
            console.log('WebSocket connection closed');
            onClose();
        };
        
        return ws;
    },

    // ========================================
    // LEGACY EVALUATION API ENDPOINTS
    // ========================================

    // Get current evaluation metrics
    getMetrics: async (timeframe = '7d') => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/evaluation/metrics?timeframe=${timeframe}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to get evaluation metrics: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting evaluation metrics:', error);
            throw error;
        }
    },

    // Get historical evaluation data
    getHistoricalMetrics: async (days = 30) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/evaluation/historical?days=${days}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to get historical metrics: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting historical metrics:', error);
            throw error;
        }
    },

    // Get latency distribution
    getLatencyDistribution: async (timeframe = '7d') => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/evaluation/latency-distribution?timeframe=${timeframe}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to get latency distribution: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting latency distribution:', error);
            throw error;
        }
    },

    // Get quality breakdown
    getQualityBreakdown: async (metric = 'answer_quality') => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/evaluation/quality-breakdown?metric=${metric}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to get quality breakdown: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting quality breakdown:', error);
            throw error;
        }
    },

    // Get evaluation datasets
    getDatasets: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/evaluation/datasets`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to get datasets: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting datasets:', error);
            throw error;
        }
    },

    // Create evaluation dataset using background task
    createDataset: async (datasetConfig) => {
        try {
            console.log('Creating dataset with config:', datasetConfig);
            
            const requestBody = {
                name: datasetConfig.name,
                description: datasetConfig.description,
                document_ids: datasetConfig.document_ids,
                num_questions_per_doc: datasetConfig.num_questions_per_doc || 3,
                model_name: datasetConfig.model_name || 'llama3',
                difficulty_levels: datasetConfig.difficulty_levels || ['easy', 'medium', 'hard'],
                user_id: datasetConfig.user_id || 'admin'
            };
            
            console.log('Request body:', requestBody);
            console.log('Request body types:', {
                name: typeof requestBody.name,
                description: typeof requestBody.description,
                document_ids: Array.isArray(requestBody.document_ids) ? 'array' : typeof requestBody.document_ids,
                document_ids_content: requestBody.document_ids,
                num_questions_per_doc: typeof requestBody.num_questions_per_doc,
                model_name: typeof requestBody.model_name,
                difficulty_levels: Array.isArray(requestBody.difficulty_levels) ? 'array' : typeof requestBody.difficulty_levels,
                user_id: typeof requestBody.user_id
            });
            
            const response = await fetch(`${API_BASE_URL}/api/evaluation/create/dataset`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                // Get detailed error information for 422 errors
                let errorDetail = `Failed to create dataset: ${response.status}`;
                try {
                    const errorData = await response.json();
                    console.error('API error response:', errorData);
                    if (errorData.detail) {
                        errorDetail = Array.isArray(errorData.detail) 
                            ? errorData.detail.map(e => e.msg || e.message || e).join(', ')
                            : errorData.detail;
                    }
                } catch (e) {
                    console.error('Error parsing error response:', e);
                }
                throw new Error(errorDetail);
            }

            return await response.json();
        } catch (error) {
            console.error('Error creating dataset:', error);
            throw error;
        }
    },

    // Get dataset generation progress
    getDatasetProgress: async (datasetId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/evaluation/datasets/${datasetId}/progress`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to get progress: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting dataset progress:', error);
            throw error;
        }
    },

    // Preview dataset creation (estimate what will be generated)
    previewDataset: async (previewConfig) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/evaluation/datasets/preview`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify(previewConfig)
            });

            if (!response.ok) {
                throw new Error(`Failed to preview dataset: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error previewing dataset:', error);
            throw error;
        }
    },

    // Delete evaluation dataset
    deleteDataset: async (datasetId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/evaluation/datasets/${datasetId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to delete dataset: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error deleting dataset:', error);
            throw error;
        }
    },

    // Get detailed information about a specific dataset
    getDatasetDetails: async (datasetId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/evaluation/datasets/${datasetId}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to get dataset details: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting dataset details:', error);
            throw error;
        }
    },

    // Run evaluation test case
    runTestCase: async (datasetId, models) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/evaluation/test-cases/run`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                },
                body: JSON.stringify({
                    dataset_id: datasetId,
                    models: models
                })
            });

            if (!response.ok) {
                throw new Error(`Failed to run test case: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error running test case:', error);
            throw error;
        }
    },

    // Get evaluation results
    getResults: async (modelFilter = null, datasetFilter = null) => {
        console.log('🌐 [DEBUG] evaluationApi.getResults() called with filters:', { modelFilter, datasetFilter });
        try {
            let url = `${API_BASE_URL}/api/evaluation/results`;
            const params = new URLSearchParams();
            
            if (modelFilter) params.append('model_filter', modelFilter);
            if (datasetFilter) params.append('dataset_filter', datasetFilter);
            
            if (params.toString()) {
                url += `?${params.toString()}`;
            }

            console.log('🌐 [DEBUG] Making fetch request to:', url);
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...getAuthHeader()
                }
            });

            console.log('🌐 [DEBUG] Response status:', response.status, response.ok);
            if (!response.ok) {
                throw new Error(`Failed to get evaluation results: ${response.status}`);
            }

            const data = await response.json();
            console.log('✅ [DEBUG] evaluationApi.getResults() successful, data:', data);
            return data;
        } catch (error) {
            console.error('❌ [DEBUG] Error in evaluationApi.getResults():', error);
            throw error;
        }
    }
};
