import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import FolderUploadReview from '../components/FolderUploadReview';
import FileUploadReview from '../components/FileUploadReview';
import DocumentReingestionModal from '../components/DocumentReingestionModal';
import {
    Box,
    Drawer,
    List,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Typography,
    IconButton,
    Divider,
    ThemeProvider,
    Card,
    CardContent,
    Avatar,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Chip,
} from '@mui/material';
import {
    ArrowBack,
    Description,
    Settings,
    Search,
    Storage as StorageIcon,
    CloudUpload as CloudUploadIcon,
    Folder as FolderIcon,
    Article as ArticleIcon,
    CheckCircle as CheckCircleIcon,
} from '@mui/icons-material';
import ollamaIcon from '../assets/ollama.svg';
import huggingfaceIcon from '../assets/huggingface.svg';
import { theme } from '../theme';

// Add custom styles for resizable table
const tableStyles = `
.resizable-table {
    table-layout: fixed;
    width: 100%;
}

.resizable-table th {
    position: relative;
    border-right: 2px solid #e5e7eb;
    min-width: 80px;
    user-select: none;
}

.resizable-table th:hover {
    border-right-color: #6366f1;
}

.resizable-table th .resize-handle {
    position: absolute;
    right: -3px;
    top: 0;
    width: 6px;
    height: 100%;
    cursor: col-resize;
    background: transparent;
    z-index: 10;
}

.resizable-table th .resize-handle:hover {
    background: rgba(99, 102, 241, 0.3);
}

.resizable-table td {
    overflow: hidden;
    word-wrap: break-word;
    word-break: break-word;
}

.filename-cell {
    min-width: 0;
    max-width: 100%;
}

.filename-text {
    word-break: break-all;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    line-height: 1.4;
}

.actions-cell {
    white-space: nowrap;
    min-width: 120px;
}

.actions-button {
    white-space: nowrap;
    display: inline-flex;
    align-items: center;
    min-width: 80px;
}

.table-container {
    max-height: 70vh;
    overflow: auto;
    width: 100%;
}

.text-truncate-multiline {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
    word-break: break-word;
}

.resizing {
    cursor: col-resize;
    user-select: none;
}

@media (max-width: 1024px) {
    .table-container {
        max-height: 60vh;
    }
    
    .resizable-table th, .resizable-table td {
        min-width: 60px;
        font-size: 0.875rem;
    }
}
`;

const KnowledgeHubPage = () => {
    const navigate = useNavigate();
    const [files, setFiles] = useState([]);
    const [uploadProgress, setUploadProgress] = useState({});
    const [loading, setLoading] = useState(true);
    const [isReingesting, setIsReingesting] = useState(false);
    const [activeTab, setActiveTab] = useState('documents');
    const [chunkingMethods, setChunkingMethods] = useState([]);
    const [selectedMethod, setSelectedMethod] = useState('general');
    const [methodConfigs, setMethodConfigs] = useState({});
    const [defaultConfigs, setDefaultConfigs] = useState({});
    const [activeConfig, setActiveConfig] = useState(null);
    const [savingConfig, setSavingConfig] = useState(false);
    const [saveMessage, setSaveMessage] = useState(null);
    const [loadingChunking, setLoadingChunking] = useState(false);
    const [chunkingError, setChunkingError] = useState(null);
    const [methodsData, setMethodsData] = useState({});
    const [warningDialog, setWarningDialog] = useState(null);
    const [fileValidationToast, setFileValidationToast] = useState(null);
    
    const [openDropdown, setOpenDropdown] = useState(null);
    const [dropdownPosition, setDropdownPosition] = useState({ top: 0, left: 0 });
    
    // Folder upload states
    const [showFolderUploadReview, setShowFolderUploadReview] = useState(false);
    const [showFileUploadReview, setShowFileUploadReview] = useState(false);
    const [selectedFilesForReview, setSelectedFilesForReview] = useState([]);
    const [processingDocuments, setProcessingDocuments] = useState(new Set());

    // Document management states
    const [selectedDocuments, setSelectedDocuments] = useState(new Set());
    const [searchTerm, setSearchTerm] = useState('');
    const [showReingestionDialog, setShowReingestionDialog] = useState(false);
    const [reingestionConfig, setReingestionConfig] = useState({
        chunkingMethod: 'general',
        chunkSize: 1000,
        chunkOverlap: 200
    });

    // Retrieval settings states
    const [retrievalConfig, setRetrievalConfig] = useState({
        similarity_threshold: 0.2,
        keyword_similarity_weight: 0.7,
        reranker_enabled: false,
        reranker_model: '',
        reranker_provider: 'ollama', // Add provider for reranker
        max_chunks: 5,
        search_type: 'similarity',
        auto_merging_enabled: false,
        auto_merging_similarity_threshold: 0.8
    });
    const [rerankerModels, setRerankerModels] = useState([]);
    const [rerankerModelsCache, setRerankerModelsCache] = useState({});
    const [loadingRerankerModels, setLoadingRerankerModels] = useState(false);
    const [selectedRerankerProvider, setSelectedRerankerProvider] = useState('ollama');
    const [loadingRetrievalConfig, setLoadingRetrievalConfig] = useState(false);
    const [savingRetrievalConfig, setSavingRetrievalConfig] = useState(false);
    const [retrievalConfigMessage, setRetrievalConfigMessage] = useState(null);
    const [downloadingRerankerModel, setDownloadingRerankerModel] = useState(false);
    const [downloadProgress, setDownloadProgress] = useState('');
    const [showAdvancedReingestionConfig, setShowAdvancedReingestionConfig] = useState(false);
    
    // WebSocket connection for download progress
    const downloadWebSocketRef = useRef(null);

    // Status polling states
    const [statusPolling, setStatusPolling] = useState(false);
    const [pendingDocuments, setPendingDocuments] = useState(new Set());
    const [pollingInterval, setPollingInterval] = useState(null);

    // Utility function to extract just the filename from a path
    const extractFilename = (filepath) => {
        if (!filepath) return '';
        return filepath.split('/').pop();
    };

    // Inject table styles
    useEffect(() => {
        const styleId = 'resizable-table-styles';
        if (!document.getElementById(styleId)) {
            const style = document.createElement('style');
            style.id = styleId;
            style.textContent = tableStyles;
            document.head.appendChild(style);
        }
        return () => {
            const existingStyle = document.getElementById(styleId);
            if (existingStyle) {
                existingStyle.remove();
            }
        };
    }, []);

    // Add column resizing functionality
    useEffect(() => {
        const addResizeHandles = () => {
            const table = document.querySelector('.resizable-table');
            if (!table) return;

            const headers = table.querySelectorAll('th');
            
            headers.forEach((header, index) => {
                // Skip the last column (Actions) - don't make it resizable
                if (index === headers.length - 1) return;

                // Remove existing handle
                const existingHandle = header.querySelector('.resize-handle');
                if (existingHandle) {
                    existingHandle.remove();
                }

                // Create resize handle
                const resizeHandle = document.createElement('div');
                resizeHandle.className = 'resize-handle';
                header.appendChild(resizeHandle);

                let isResizing = false;
                let startX = 0;
                let startWidth = 0;

                const onMouseDown = (e) => {
                    isResizing = true;
                    startX = e.clientX;
                    startWidth = header.offsetWidth;
                    document.body.classList.add('resizing');
                    e.preventDefault();
                };

                const onMouseMove = (e) => {
                    if (!isResizing) return;
                    
                    const diff = e.clientX - startX;
                    const newWidth = startWidth + diff;
                    const minWidth = 80;
                    
                    if (newWidth >= minWidth) {
                        header.style.width = newWidth + 'px';
                    }
                };

                const onMouseUp = () => {
                    isResizing = false;
                    document.body.classList.remove('resizing');
                };

                resizeHandle.addEventListener('mousedown', onMouseDown);
                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);

                // Cleanup function
                const cleanup = () => {
                    resizeHandle.removeEventListener('mousedown', onMouseDown);
                    document.removeEventListener('mousemove', onMouseMove);
                    document.removeEventListener('mouseup', onMouseUp);
                };

                // Store cleanup function on the handle
                resizeHandle._cleanup = cleanup;
            });
        };

        // Add handles after component mounts and when files change
        const timer = setTimeout(addResizeHandles, 100);

        return () => {
            clearTimeout(timer);
            // Cleanup all resize handles
            const handles = document.querySelectorAll('.resize-handle');
            handles.forEach(handle => {
                if (handle._cleanup) {
                    handle._cleanup();
                }
            });
        };
    }, [files]);

    useEffect(() => {
        loadFiles();
        loadChunkingMethods();
        loadRetrievalConfig();
        preloadRerankerModels(); // Use preload instead of loadRerankerModels
    }, []);

    // Load retrieval config when switching to retrieval tab
    useEffect(() => {
        if (activeTab === 'retrieval') {
            loadRetrievalConfig();
            // Don't reload models here since they're preloaded
        }
    }, [activeTab]);

    // Reload reranker models when provider changes (now instant from cache)
    useEffect(() => {
        loadRerankerModels();
    }, [selectedRerankerProvider]);

    // Status polling effect - monitors documents with pending status
    useEffect(() => {
        const checkPendingDocuments = () => {
            const pending = new Set();
            files.forEach(file => {
                const status = getDocumentStatus(file);
                // Only poll for truly pending documents (not failed, completed, or partial)
                if (status.status === 'pending') {
                    pending.add(file.id);
                }
                // Also include documents that are manually marked as processing
                if (processingDocuments.has(file.id) && status.status === 'pending') {
                    pending.add(file.id);
                }
            });
            setPendingDocuments(pending);
            return pending.size > 0;
        };

        // Start polling if there are pending documents
        if (files.length > 0) {
            const hasPending = checkPendingDocuments();
            
            if (hasPending && !statusPolling) {
                setStatusPolling(true);
                const interval = setInterval(async () => {
                    try {
                        await loadFiles(true); // Silent refresh during polling
                        const stillHasPending = checkPendingDocuments();
                        if (!stillHasPending) {
                            setStatusPolling(false);
                            clearInterval(interval);
                            setPollingInterval(null);
                        }
                    } catch (error) {
                        console.error('Error during status polling:', error);
                    }
                }, 3000); // Poll every 3 seconds
                
                setPollingInterval(interval);
            } else if (!hasPending && statusPolling) {
                setStatusPolling(false);
                if (pollingInterval) {
                    clearInterval(pollingInterval);
                    setPollingInterval(null);
                }
            }
        }

        return () => {
            if (pollingInterval) {
                clearInterval(pollingInterval);
                setPollingInterval(null);
            }
        };
    }, [files, processingDocuments, statusPolling]);

    // Cleanup polling on component unmount
    useEffect(() => {
        return () => {
            if (pollingInterval) {
                clearInterval(pollingInterval);
            }
        };
    }, []);

    const loadFiles = async (silent = false) => {
        try {
            if (!silent) {
                setLoading(true);
            }
            const response = await api.listFiles();
            const newFiles = response.files || [];
            
            // Track status changes for notifications and cleanup processing state
            if (files.length > 0) {
                const oldFileMap = new Map(files.map(f => [f.id, f]));
                const completedOrFailedDocuments = new Set();
                
                newFiles.forEach(newFile => {
                    const oldFile = oldFileMap.get(newFile.id);
                    if (oldFile) {
                        const oldStatus = getDocumentStatus(oldFile);
                        const newStatus = getDocumentStatus(newFile);
                        
                        // Check if document is no longer pending
                        if (newStatus.status !== 'pending') {
                            completedOrFailedDocuments.add(newFile.id);
                        }
                        
                        // Notify on status change from pending to completed
                        if (oldStatus.status === 'pending' && newStatus.status === 'completed') {
                            if (!silent) {
                                showValidationToast(
                                    `✅ Document "${extractFilename(newFile.filename)}" has been successfully indexed!`,
                                    'success',
                                    4000
                                );
                            }
                        } else if (oldStatus.status === 'pending' && newStatus.status === 'failed') {
                            if (!silent) {
                                showValidationToast(
                                    `❌ Document "${extractFilename(newFile.filename)}" indexing failed.`,
                                    'error',
                                    5000
                                );
                            }
                        } else if (oldStatus.status === 'pending' && newStatus.status === 'partial') {
                            if (!silent) {
                                showValidationToast(
                                    `⚠️ Document "${extractFilename(newFile.filename)}" partially indexed (some models failed).`,
                                    'warning',
                                    5000
                                );
                            }
                        }
                    }
                });
                
                // Clear completed/failed documents from processing state
                if (completedOrFailedDocuments.size > 0) {
                    setProcessingDocuments(prev => {
                        const newSet = new Set(prev);
                        completedOrFailedDocuments.forEach(id => newSet.delete(id));
                        return newSet;
                    });
                }
            }
            
            setFiles(newFiles);
        } catch (error) {
            console.error('Error loading files:', error);
            if (!silent) {
                showValidationToast('❌ Error refreshing document list', 'error');
            }
        } finally {
            if (!silent) {
                setLoading(false);
            }
        }
    };

    const loadChunkingMethods = async () => {
        setLoadingChunking(true);
        setChunkingError(null);
        try {
            const response = await api.getChunkingMethods();
            // Backend returns {methods: {method1: {...}, method2: {...}}}
            const methodsData = response.methods || {};
            const methodsList = Object.keys(methodsData);
            setChunkingMethods(methodsList);
            setMethodsData(methodsData); // Store full methods data for validation
            
            // Load configurations for all methods and store defaults
            const configs = {};
            const defaults = {};
            for (const method of methodsList) {
                try {
                    const response = await api.getChunkingConfig(method);
                    // Backend returns {config: {...}}
                    const config = response.config || {};
                    configs[method] = config;
                    // Store a deep copy as default config for reset functionality
                    defaults[method] = JSON.parse(JSON.stringify(config));
                } catch (error) {
                    console.error(`Error loading config for ${method}:`, error);
                }
            }
            setMethodConfigs(configs);
            setDefaultConfigs(defaults);
            
            // Set active config to default method
            if (methodsList.length > 0) {
                const defaultMethod = methodsList.includes('general') ? 'general' : methodsList[0];
                setSelectedMethod(defaultMethod);
                setActiveConfig(configs[defaultMethod] || null);
            }
        } catch (error) {
            console.error('Error loading chunking methods:', error);
            setChunkingError(error.message || 'Failed to load chunking methods');
        } finally {
            setLoadingChunking(false);
        }
    };

    const handleMethodChange = async (method) => {
        // Save current config changes to methodConfigs before switching
        if (selectedMethod && activeConfig) {
            setMethodConfigs(prev => ({
                ...prev,
                [selectedMethod]: activeConfig
            }));
        }
        
        setSelectedMethod(method);
        if (methodConfigs[method]) {
            setActiveConfig(methodConfigs[method]);
        } else {
            // Load config if not already loaded
            try {
                const response = await api.getChunkingConfig(method);
                // Backend returns {config: {...}}
                const config = response.config || {};
                setMethodConfigs(prev => ({ ...prev, [method]: config }));
                setActiveConfig(config);
            } catch (error) {
                console.error(`Error loading config for ${method}:`, error);
            }
        }
    };

    const handleConfigChange = (field, value) => {
        setActiveConfig(prev => ({
            ...prev,
            [field]: value
        }));
    };

    const saveChunkingConfig = async () => {
        if (!selectedMethod || !activeConfig) return;
        
        setSavingConfig(true);
        setSaveMessage(null);
        try {
            const response = await api.updateChunkingConfig(selectedMethod, activeConfig);
            setMethodConfigs(prev => ({
                ...prev,
                [selectedMethod]: activeConfig
            }));
            
            // Show success message
            setSaveMessage({ type: 'success', text: 'Configuration saved successfully!' });
            console.log('Chunking configuration saved successfully:', response);
            
            // Clear message after 3 seconds
            setTimeout(() => setSaveMessage(null), 3000);
        } catch (error) {
            console.error('Error saving chunking configuration:', error);
            setSaveMessage({ 
                type: 'error', 
                text: error.message || 'Failed to save configuration. Please try again.' 
            });
            
            // Clear error message after 5 seconds
            setTimeout(() => setSaveMessage(null), 5000);
        } finally {
            setSavingConfig(false);
        }
    };

    const resetChunkingConfig = () => {
        if (!selectedMethod || !defaultConfigs[selectedMethod]) return;
        
        // Show confirmation
        if (!window.confirm('Are you sure you want to reset all settings to default values? Any unsaved changes will be lost.')) {
            return;
        }
        
        // Reset to default configuration
        const defaultConfig = JSON.parse(JSON.stringify(defaultConfigs[selectedMethod]));
        setActiveConfig(defaultConfig);
        
        // Update methodConfigs to reflect the reset
        setMethodConfigs(prev => ({
            ...prev,
            [selectedMethod]: defaultConfig
        }));
        
        // Show reset message
        setSaveMessage({ type: 'success', text: 'Settings reset to default values. Remember to save if you want to keep these changes.' });
        
        // Clear message after 4 seconds
        setTimeout(() => setSaveMessage(null), 4000);
    };

    // Retrieval Configuration Functions
    const loadRetrievalConfig = async () => {
        setLoadingRetrievalConfig(true);
        try {
            const response = await api.getRetrievalConfig();
            const config = response.config || {
                similarity_threshold: 0.2,
                keyword_similarity_weight: 0.7,
                reranker_enabled: false,
                reranker_model: '',
                reranker_provider: 'ollama',
                max_chunks: 5,
                search_type: 'similarity',
                auto_merging_enabled: false,
                auto_merging_similarity_threshold: 0.8
            };
            
            setRetrievalConfig(config);
            
            // Set the reranker provider if it exists in the config
            if (config.reranker_provider) {
                setSelectedRerankerProvider(config.reranker_provider);
            }
        } catch (error) {
            console.error('Error loading retrieval config:', error);
            setRetrievalConfigMessage({ type: 'error', text: 'Failed to load retrieval configuration' });
        } finally {
            setLoadingRetrievalConfig(false);
        }
    };

    const loadRerankerModels = async (provider = selectedRerankerProvider) => {
        // Check cache first
        if (rerankerModelsCache[provider]) {
            console.log(`Using cached models for ${provider}:`, rerankerModelsCache[provider].length);
            setRerankerModels(rerankerModelsCache[provider]);
            return;
        }
        
        console.log(`Loading models for provider: ${provider}`);
        setLoadingRerankerModels(true);
        try {
            // Pass the selected provider to get filtered models from backend
            const response = await api.getRerankerModels(provider);
            const allModels = response.models || [];
            
            console.log(`Loaded ${allModels.length} models for ${provider}`);
            
            // Cache the result
            setRerankerModelsCache(prev => ({
                ...prev,
                [provider]: allModels
            }));
            
            // Set current models
            setRerankerModels(allModels);
        } catch (error) {
            console.error('Error loading reranker models:', error);
        } finally {
            setLoadingRerankerModels(false);
        }
    };

    // Pre-load both providers' models
    const preloadRerankerModels = async () => {
        console.log('Preloading reranker models for both providers...');
        try {
            // Load both providers in parallel
            const [ollamaResponse, huggingfaceResponse] = await Promise.all([
                api.getRerankerModels('ollama'),
                api.getRerankerModels('huggingface')
            ]);
            
            const cache = {
                ollama: ollamaResponse.models || [],
                huggingface: huggingfaceResponse.models || []
            };
            
            console.log('Preloaded models:', cache);
            setRerankerModelsCache(cache);
            
            // Set current models based on selected provider
            setRerankerModels(cache[selectedRerankerProvider] || []);
        } catch (error) {
            console.error('Error preloading reranker models:', error);
        }
    };

    // WebSocket connection management for download progress
    const connectToDownloadWebSocket = () => {
        try {
            // ✅ UPDATED: No longer using localStorage token
            // Authentication now uses httpOnly cookies automatically
            
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host;
            const wsUrl = `${protocol}//${host}/api/ws/download-progress`;
            
            console.log('🔌 Connecting to download progress WebSocket (cookie-based auth)');
            downloadWebSocketRef.current = new WebSocket(wsUrl);
            
            downloadWebSocketRef.current.onopen = () => {
                console.log('✅ Connected to download progress WebSocket');
            };
            
            downloadWebSocketRef.current.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'download_progress') {
                        setDownloadProgress(data.message);
                        
                        if (data.status === 'completed') {
                            setDownloadingRerankerModel(false);
                            setDownloadProgress('');
                            setRetrievalConfigMessage({ 
                                type: 'success', 
                                text: `Retrieval configuration saved and model ${data.model_name} downloaded successfully!` 
                            });
                            setTimeout(() => setRetrievalConfigMessage(null), 5000);
                            disconnectDownloadWebSocket();
                        } else if (data.status === 'failed') {
                            setDownloadingRerankerModel(false);
                            setDownloadProgress('');
                            setRetrievalConfigMessage({ 
                                type: 'error', 
                                text: `Model download failed: ${data.message}` 
                            });
                            disconnectDownloadWebSocket();
                        }
                    }
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            downloadWebSocketRef.current.onclose = (event) => {
                // Handle authentication failures (code 1008)
                if (event.code === 1008) {
                    console.error('❌ Download WebSocket authentication failed:', event.reason);
                    setDownloadingRerankerModel(false);
                    setDownloadProgress('');
                    setRetrievalConfigMessage({ 
                        type: 'error', 
                        text: 'Authentication failed. Please log in again.' 
                    });
                } else {
                    console.log('Download progress WebSocket disconnected:', event.code, event.reason);
                }
            };
            
            downloadWebSocketRef.current.onerror = (error) => {
                console.error('Download progress WebSocket error:', error);
                setDownloadingRerankerModel(false);
                setDownloadProgress('');
                setRetrievalConfigMessage({ 
                    type: 'error', 
                    text: 'WebSocket connection error. Please try again.' 
                });
            };
        } catch (error) {
            console.error('Error connecting to download WebSocket:', error);
            setDownloadingRerankerModel(false);
            setDownloadProgress('');
        }
    };
    
    const disconnectDownloadWebSocket = () => {
        if (downloadWebSocketRef.current) {
            downloadWebSocketRef.current.close();
            downloadWebSocketRef.current = null;
        }
    };

    const pollDownloadStatus = async (modelName) => {
        try {
            const status = await api.getRerankerDownloadStatus(modelName);
            
            if (status.downloading) {
                setDownloadProgress(status.message);
                // Continue polling
                setTimeout(() => pollDownloadStatus(modelName), 2000);
            } else if (status.completed) {
                setDownloadingRerankerModel(false);
                setDownloadProgress('');
                setRetrievalConfigMessage({ type: 'success', text: `Model ${modelName} downloaded successfully!` });
                setTimeout(() => setRetrievalConfigMessage(null), 3000);
            } else if (status.message.includes('failed') || status.message.includes('error')) {
                setDownloadingRerankerModel(false);
                setDownloadProgress('');
                setRetrievalConfigMessage({ type: 'error', text: `Model download failed: ${status.message}` });
            }
        } catch (error) {
            console.error('Error polling download status:', error);
            setDownloadingRerankerModel(false);
            setDownloadProgress('');
        }
    };

    const saveRetrievalConfig = async () => {
        setSavingRetrievalConfig(true);
        setRetrievalConfigMessage(null);
        try {
            // Include the selected reranker provider in the configuration
            const configToSave = {
                ...retrievalConfig,
                reranker_provider: selectedRerankerProvider
            };
            
            const response = await api.updateRetrievalConfig(configToSave);
            
            // Check if we need to start download progress tracking for HuggingFace models
            if (configToSave.reranker_enabled && 
                configToSave.reranker_model && 
                configToSave.reranker_model.toLowerCase() !== "none" &&
                selectedRerankerProvider === 'huggingface') {
                
                setDownloadingRerankerModel(true);
                setDownloadProgress('Starting download...');
                // Connect to WebSocket for real-time progress updates
                connectToDownloadWebSocket();
                
                // Don't show success message yet, wait for download completion via WebSocket
            } else {
                // Only show success message if no download is in progress
                if (response.warnings && response.warnings.length > 0) {
                    setRetrievalConfigMessage({ 
                        type: 'warning', 
                        text: `Configuration saved with warnings: ${response.warnings.join(', ')}` 
                    });
                } else {
                    setRetrievalConfigMessage({ type: 'success', text: 'Retrieval configuration saved successfully!' });
                }
            }
            
            // Clear message after 3 seconds (unless downloading)
            if (!downloadingRerankerModel) {
                setTimeout(() => setRetrievalConfigMessage(null), 3000);
            }
        } catch (error) {
            console.error('Error saving retrieval config:', error);
            setRetrievalConfigMessage({ type: 'error', text: 'Failed to save retrieval configuration' });
        } finally {
            setSavingRetrievalConfig(false);
        }
    };

    const handleRetrievalConfigChange = (field, value) => {
        setRetrievalConfig(prev => ({
            ...prev,
            [field]: value
        }));
    };

    const resetRetrievalConfig = () => {
        if (!window.confirm('Are you sure you want to reset all retrieval settings to default values?')) {
            return;
        }
        
        setRetrievalConfig({
            similarity_threshold: 0.2,
            keyword_similarity_weight: 0.7,
            reranker_enabled: false,
            reranker_model: '',
            reranker_provider: 'ollama',
            max_chunks: 5,
            search_type: 'similarity',
            auto_merging_enabled: false,
            auto_merging_similarity_threshold: 0.8
        });
        
        setSelectedRerankerProvider('ollama');
        
        setRetrievalConfigMessage({ type: 'success', text: 'Settings reset to default values. Remember to save if you want to keep these changes.' });
        setTimeout(() => setRetrievalConfigMessage(null), 4000);
    };

    // File extension validation for chunking methods
    const validateFileForChunkingMethod = (fileName, fileExtension, method) => {
        if (!methodsData[method] || !methodsData[method].supported_formats) {
            return {
                isValid: false,
                message: `Unknown chunking method: ${method}`,
                supportedFormats: []
            };
        }
        
        const supportedFormats = methodsData[method].supported_formats || [];
        const isSupported = supportedFormats.includes(fileExtension);
        
        return {
            isValid: isSupported,
            message: isSupported 
                ? `File format '${fileExtension}' is supported by ${method} method` 
                : `File format '${fileExtension}' is not supported by the '${method}' chunking method`,
            supportedFormats: supportedFormats,
            method: method
        };
    };

    // File type detection
    const getFileTypeInfo = (fileName, fileExtension) => {
        const fileTypeMap = {
            'pdf': { type: 'PDF', icon: '📄', category: 'document' },
            'docx': { type: 'Word Document', icon: '📝', category: 'document' },
            'doc': { type: 'Word Document', icon: '📝', category: 'document' },
            'pptx': { type: 'PowerPoint', icon: '📊', category: 'presentation' },
            'ppt': { type: 'PowerPoint', icon: '📊', category: 'presentation' },
            'xlsx': { type: 'Excel', icon: '📈', category: 'spreadsheet' },
            'xls': { type: 'Excel', icon: '📈', category: 'spreadsheet' },
            'csv': { type: 'CSV', icon: '📊', category: 'data' },
            'txt': { type: 'Plain Text', icon: '📄', category: 'text' },
            'md': { type: 'Markdown', icon: '📝', category: 'text' },
            'jpg': { type: 'Image', icon: '🖼️', category: 'image' },
            'jpeg': { type: 'Image', icon: '🖼️', category: 'image' },
            'png': { type: 'Image', icon: '🖼️', category: 'image' },
            'gif': { type: 'Image', icon: '🖼️', category: 'image' },
            'tif': { type: 'Image', icon: '🖼️', category: 'image' },
            'tiff': { type: 'Image', icon: '🖼️', category: 'image' },
            'json': { type: 'JSON', icon: '📋', category: 'data' },
            'html': { type: 'HTML', icon: '🌐', category: 'web' },
            'htm': { type: 'HTML', icon: '🌐', category: 'web' },
            'eml': { type: 'Email', icon: '📧', category: 'email' }
        };

        return fileTypeMap[fileExtension] || { type: 'Unknown', icon: '📄', category: 'unknown' };
    };

    // Method recommendation based on file type
    const getRecommendedMethod = (fileExtension, fileType) => {
        const recommendations = {
            'pdf': ['general', 'qa'],
            'docx': ['general', 'resume', 'qa'],
            'doc': ['general', 'resume', 'qa'],
            'pptx': ['presentation'],
            'ppt': ['presentation'],
            'xlsx': ['table'],
            'xls': ['table'],
            'csv': ['table'],
            'txt': ['general', 'qa'],
            'md': ['general', 'qa'],
            'jpg': ['picture'],
            'jpeg': ['picture'],
            'png': ['picture'],
            'gif': ['picture'],
            'tif': ['picture'],
            'tiff': ['picture'],
            'json': ['qa'],
            'html': ['general'],
            'htm': ['general'],
            'eml': ['email']
        };

        return recommendations[fileExtension] || ['general'];
    };

    // Check if current method is good for file type
    const validateMethodForFileType = (fileExtension, selectedMethod) => {
        const goodMethods = getRecommendedMethod(fileExtension);
        const isGoodMatch = goodMethods.includes(selectedMethod);
        
        const warnings = {
            'pdf': {
                bad: ['resume', 'picture', 'presentation'],
                message: 'PDF files work best with General or Q&A chunking methods'
            },
            'docx': {
                bad: ['picture', 'presentation', 'table'],
                message: 'Word documents work best with General, Resume, or Q&A chunking methods'
            },
            'pptx': {
                bad: ['resume', 'qa', 'general', 'picture'],
                message: 'PowerPoint files should use Presentation chunking method'
            },
            'jpg': {
                bad: ['resume', 'general', 'qa', 'table'],
                message: 'Image files should use Picture chunking method'
            },
            'xlsx': {
                bad: ['resume', 'picture', 'presentation'],
                message: 'Spreadsheet files should use Table chunking method'
            }
        };

        const fileWarning = warnings[fileExtension];
        const shouldWarn = fileWarning && fileWarning.bad.includes(selectedMethod);

        return {
            isGoodMatch,
            shouldWarn,
            recommendedMethods: goodMethods,
            warningMessage: shouldWarn ? fileWarning.message : null
        };
    };

    // Auto-suggest better method
    const suggestBetterMethod = (fileExtension, currentMethod) => {
        const recommended = getRecommendedMethod(fileExtension);
        const bestMethod = recommended[0]; // First one is usually the best
        
        if (currentMethod !== bestMethod && chunkingMethods.includes(bestMethod)) {
            return bestMethod;
        }
        
        return null;
    };

    // Show validation toast
    const showValidationToast = (message, type = 'warning', duration = 5000) => {
        setFileValidationToast({ message, type });
        setTimeout(() => setFileValidationToast(null), duration);
    };

    // Helper function to get document status
    const getDocumentStatus = (file) => {
        // If no model_status field, check if file is indexed (backward compatibility)
        if (!file.model_status) {
            // Check if file has indexed field or if it was successfully processed
            if (file.indexed === true) {
                return { status: 'completed', label: 'Indexed', color: 'bg-green-100 text-green-800' };
            }
            return { status: 'pending', label: 'Pending', color: 'bg-yellow-100 text-yellow-800' };
        }
        
        // Handle empty model_status object
        if (typeof file.model_status === 'object' && Object.keys(file.model_status).length === 0) {
            // If model_status is empty but file.indexed is true, consider it completed
            if (file.indexed === true) {
                return { status: 'completed', label: 'Indexed', color: 'bg-green-100 text-green-800' };
            }
            return { status: 'pending', label: 'Pending', color: 'bg-yellow-100 text-yellow-800' };
        }
        
        // Check statuses across all embedding models
        const statuses = Object.values(file.model_status);
        const hasFailed = statuses.some(status => status === 'failed');
        const hasCompleted = statuses.some(status => status === 'completed');
        const hasPending = statuses.some(status => status === 'pending');
        
        // Priority order: failed > completed > pending > unknown
        if (hasFailed) {
            // If any model failed and none completed, show failed
            if (!hasCompleted) {
                return { status: 'failed', label: 'Failed', color: 'bg-red-100 text-red-800' };
            }
            // If some failed but some completed, show partial success
            return { status: 'partial', label: 'Partial', color: 'bg-orange-100 text-orange-800' };
        } else if (hasCompleted) {
            return { status: 'completed', label: 'Indexed', color: 'bg-green-100 text-green-800' };
        } else if (hasPending) {
            return { status: 'pending', label: 'Pending', color: 'bg-yellow-100 text-yellow-800' };
        } else {
            // Final fallback - check if file.indexed is true
            if (file.indexed === true) {
                return { status: 'completed', label: 'Indexed', color: 'bg-green-100 text-green-800' };
            }
            return { status: 'unknown', label: 'Unknown', color: 'bg-gray-100 text-gray-800' };
        }
    };

    // Helper function to get error message for failed documents
    const getErrorMessage = (file) => {
        // Check if we have error details in the model_status or other fields
        if (file.error_message) {
            return file.error_message;
        }
        
        // Check for specific error patterns based on file type
        const fileExtension = extractFilename(file.filename || '').split('.').pop()?.toLowerCase();
        
        switch (fileExtension) {
            case 'pdf':
                return 'PDF processing failed. The file may be corrupted, password-protected, or contain unsupported content.';
            case 'docx':
            case 'doc':
                return 'Word document processing failed. Please ensure the file is not corrupted or password-protected.';
            case 'xlsx':
            case 'xls':
            case 'csv':
                return 'Spreadsheet processing failed. Check for table formatting issues or data corruption.';
            case 'pptx':
            case 'ppt':
                return 'PowerPoint processing failed. The file may contain unsupported content or be corrupted.';
            case 'jpg':
            case 'jpeg':
            case 'png':
            case 'gif':
            case 'tif':
            case 'tiff':
                return 'Image processing failed. OCR extraction may have encountered issues with image quality.';
            default:
                return 'Document processing failed. Please check the file format, ensure it\'s not corrupted, and try uploading again.';
        }
    };

    const handleFileUpload = async (event) => {
        const uploadedFiles = Array.from(event.target.files);
        
        if (uploadedFiles.length === 0) return;
        
        // Show the file upload review dialog
        setSelectedFilesForReview(uploadedFiles);
        setShowFileUploadReview(true);
        
        // Clear the input value so the same files can be selected again if needed
        event.target.value = '';
    };

    // Handle folder upload review
    const handleFolderUpload = async (files, options = {}) => {
        const { folderName, onProgress } = options;
        
        try {
            for (const fileData of files) {
                if (onProgress) {
                    onProgress(fileData.id, 0);
                }

                // Create chunking config for this file
                const chunkingConfig = {
                    method: fileData.method,
                    ...fileData.config
                };

                await api.uploadFileWithChunking(
                    fileData.file,
                    chunkingConfig,
                    true, // isFolder
                    fileData.path, // folder path
                    (progress) => {
                        if (onProgress) {
                            onProgress(fileData.id, progress);
                        }
                    }
                );
            }

            // Close the folder review modal
            setShowFolderUploadReview(false);
            
            // Refresh the files list
            await loadFiles();
            
            // Start status polling for newly uploaded files
            setStatusPolling(true);
            
            // Show success message
            showValidationToast(
                `✅ Folder "${folderName}" uploaded successfully with ${files.length} files! Processing will begin shortly.`,
                'success',
                5000
            );
            
        } catch (error) {
            console.error('Folder upload error:', error);
            showValidationToast(
                `❌ Error uploading folder: ${error.message}`,
                'error',
                5000
            );
        }
    };

    // Show folder upload review modal
    const handleShowFolderUpload = () => {
        setShowFolderUploadReview(true);
    };

    // Cancel folder upload review
    const handleCancelFolderUpload = () => {
        setShowFolderUploadReview(false);
    };

    // Handle file upload review (for individual files)
    const handleFileUploadFromReview = async (files, options = {}) => {
        try {
            for (const fileData of files) {
                // Create chunking config for this file
                const chunkingConfig = {
                    method: fileData.method,
                    ...fileData.config
                };
                await api.uploadFileWithChunking(
                    fileData.file,
                    chunkingConfig,
                    false, // isFolder
                    '', // no folder path
                    (progress) => {
                        setUploadProgress(prev => ({
                            ...prev,
                            [fileData.id]: progress
                        }));
                    }
                );
            }
            setUploadProgress({}); // Clear upload progress after completion
            setShowFileUploadReview(false);
            await loadFiles();
            
            // Start status polling for newly uploaded files
            setStatusPolling(true);
            
            showValidationToast(
                `✅ Uploaded ${files.length} file${files.length !== 1 ? 's' : ''} successfully! Processing will begin shortly.`,
                'success',
                5000
            );
        } catch (error) {
            console.error('File upload review error:', error);
            setUploadProgress({}); // Clear upload progress on error too
            showValidationToast(
                `❌ Error uploading file(s): ${error.message}`,
                'error',
                5000
            );
        }
    };
    // Retry failed document processing
    const handleRetryDocument = async (file) => {
        if (!file || !file.id) {
            showValidationToast('Invalid document selected for retry', 'error');
            return;
        }

        try {
            setProcessingDocuments(prev => new Set([...prev, file.id]));
            
            // Get current chunking configuration
            const configResponse = await fetch('/api/get-chunking-config', {
                method: 'GET',
                headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
            });
            
            if (!configResponse.ok) {
                throw new Error('Failed to get chunking configuration');
            }
            
            const config = await configResponse.json();
            
            // Create form data for retry request
            const formData = new FormData();
            formData.append('method', config.method || 'auto');
            formData.append('chunk_token_num', config.chunk_token_num || 1000);
            formData.append('chunk_overlap', config.chunk_overlap || 200);
            formData.append('delimiter', config.delimiter || '\\n\\n|\\n|\\.|\\!|\\?');
            formData.append('max_token', config.max_token || 4096);
            formData.append('layout_recognize', config.layout_recognize || 'auto');
            formData.append('preserve_formatting', config.preserve_formatting || true);
            formData.append('extract_tables', config.extract_tables || true);
            formData.append('extract_images', config.extract_images || false);
            
            // Retry processing with current configuration
            const retryResponse = await fetch(`/api/documents/${file.id}/retry`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: formData
            });

            if (!retryResponse.ok) {
                const errorData = await retryResponse.json();
                throw new Error(errorData.detail || 'Failed to retry document processing');
            }

            showValidationToast('Document processing restarted successfully', 'success');
            
            // Refresh documents after a short delay
            setTimeout(() => {
                loadFiles();
            }, 1000);
            
        } catch (error) {
            console.error('Error retrying document:', error);
            showValidationToast(`Error retrying document: ${error.message}`, 'error');
        } finally {
            setProcessingDocuments(prev => {
                const newSet = new Set(prev);
                newSet.delete(file.id);
                return newSet;
            });
        }
    };

    const handleDeleteFile = async (file) => {
        const displayName = file.filename || file.name || 'Unknown file';
        if (window.confirm(`Are you sure you want to delete "${displayName}"?`)) {
            try {
                // Ensure we have a valid file ID
                if (!file.id) {
                    throw new Error('File ID is required for deletion');
                }
                await api.deleteFile(file.id);
                await loadFiles();
                showValidationToast('✅ File deleted successfully!', 'success');
            } catch (error) {
                console.error('Error deleting file:', error);
                showValidationToast(`❌ Error deleting file: ${error.message}`, 'error');
            }
        }
    };

    const handleReingestion = async () => {
        if (!window.confirm('Are you sure you want to re-ingest all documents? This will recreate the vector store and may take some time.')) {
            return;
        }

        try {
            setIsReingesting(true);
            await api.reingestDocuments();
            
            // Wait a bit for processing to complete, then refresh data
            setTimeout(async () => {
                await loadFiles();
            }, 2000);
            
            alert('Documents re-ingested successfully! Data will refresh shortly.');
        } catch (error) {
            console.error('Error re-ingesting documents:', error);
            alert('Error re-ingesting documents. Please try again.');
        } finally {
            setIsReingesting(false);
        }
    };

    // Document selection handlers
    const handleDocumentSelection = (filename, isSelected) => {
        setSelectedDocuments(prev => {
            const newSet = new Set(prev);
            if (isSelected) {
                newSet.add(filename);
            } else {
                newSet.delete(filename);
            }
            return newSet;
        });
    };

    // Handle individual document selection
    const handleDocumentSelect = (documentId) => {
        setSelectedDocuments(prev => {
            const newSet = new Set(prev);
            if (newSet.has(documentId)) {
                newSet.delete(documentId);
            } else {
                newSet.add(documentId);
            }
            return newSet;
        });
    };

    const handleSelectAllDocuments = () => {
        const filteredFileList = getFilteredFiles();
        if (selectedDocuments.size === filteredFileList.length && filteredFileList.length > 0) {
            setSelectedDocuments(new Set());
        } else {
            setSelectedDocuments(new Set(filteredFileList.map(f => f.id)));
        }
    };

    // Bulk delete handler
    const handleBulkDelete = async () => {
        if (selectedDocuments.size === 0) return;
        
        const count = selectedDocuments.size;
        if (!window.confirm(`Are you sure you want to delete ${count} selected document${count !== 1 ? 's' : ''}?`)) {
            return;
        }

        try {
            // Get the file objects for selected documents by ID
            const selectedFiles = files.filter(f => selectedDocuments.has(f.id));
            
            for (const file of selectedFiles) {
                // Ensure we have a valid file ID
                if (!file.id) {
                    throw new Error(`File ID is required for deletion: ${file.filename}`);
                }
                await api.deleteFile(file.id);
            }
            await loadFiles();
            setSelectedDocuments(new Set());
            showValidationToast(`✅ Successfully deleted ${count} document${count !== 1 ? 's' : ''}!`, 'success');
        } catch (error) {
            console.error('Error deleting files:', error);
            showValidationToast(`❌ Error deleting files: ${error.message}`, 'error');
        }
    };

    // Bulk reingestion handler
    const handleBulkReingestion = () => {
        if (selectedDocuments.size === 0) return;
        
        setShowReingestionDialog(true);
    };

    // Handle confirm reingestion with per-document configuration
    const handleConfirmReingestion = async (reingestionData) => {
        if (!reingestionData || reingestionData.length === 0) return;
        
        setIsReingesting(true);
        try {
            const result = await api.reingestSpecificDocuments(reingestionData);
            
            if (result.results.successful > 0) {
                showValidationToast(`✅ Successfully reingested ${result.results.successful}/${result.results.total} documents`, 'success');
                
                // Refresh the documents list
                await loadFiles();
                
                // Start status polling for reingested documents
                setStatusPolling(true);
            } else {
                showValidationToast(`⚠️ No documents were successfully reingested. Check the logs for details.`, 'warning');
            }
            
            // Close dialog and clear selection
            setShowReingestionDialog(false);
            setSelectedDocuments(new Set());
            
        } catch (error) {
            console.error('Error reingesting documents:', error);
            showValidationToast(`❌ Error reingesting documents: ${error.message}`, 'error');
        } finally {
            setIsReingesting(false);
        }
    };

    // Filter files based on search term
    const getFilteredFiles = () => {
        if (!searchTerm) return files;
        return files.filter(file => 
            file.filename.toLowerCase().includes(searchTerm.toLowerCase())
        );
    };

    const formatFileSize = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    const formatDate = (dateString) => {
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    const formatChunkingMethod = (method) => {
        const methodMap = {
            'general': 'General',
            'qa': 'Q&A',
            'resume': 'Resume',
            'table': 'Table',
            'presentation': 'Presentation',
            'picture': 'Image',
            'email': 'Email'
        };
        return methodMap[method] || method || 'General';
    };

    const getChunkingMethodStyle = (method) => {
        const styleMap = {
            'qa': 'bg-green-100 text-green-700',
            'resume': 'bg-purple-100 text-purple-700',
            'general': 'bg-orange-100 text-orange-700',
            'table': 'bg-yellow-100 text-yellow-700',
            'presentation': 'bg-pink-100 text-pink-700',
            'picture': 'bg-indigo-100 text-indigo-700',
            'email': 'bg-teal-100 text-teal-700'
        };
        return styleMap[method] || 'bg-blue-100 text-blue-700';
    };

    const refreshData = async () => {
        try {
            setLoading(true);
            await loadFiles();
            showValidationToast('✅ Document list refreshed', 'success', 2000);
        } catch (error) {
            showValidationToast('❌ Error refreshing document list', 'error');
        } finally {
            setLoading(false);
        }
    };

    const toggleDropdown = (filename, event) => {
        console.log('Toggle called:', filename, 'Current:', openDropdown, 'Event:', event?.type);
        
        if (event) {
            event.preventDefault();
            event.stopPropagation();
            
            // Calculate position
            const rect = event.target.getBoundingClientRect();
            setDropdownPosition({
                top: rect.bottom + window.scrollY + 5,
                left: rect.left + window.scrollX
            });
        }
        
        setOpenDropdown(prev => {
            const newValue = prev === filename ? null : filename;
            console.log('Setting openDropdown from', prev, 'to', newValue);
            return newValue;
        });
    };

    // Close dropdown when clicking outside - simplified approach
    useEffect(() => {
        const handleDocumentClick = (event) => {
            if (openDropdown) {
                // Check if the clicked element is part of any dropdown
                const isDropdownClick = event.target.closest('.dropdown-container');
                if (!isDropdownClick) {
                    console.log('Document click outside dropdown, closing...');
                    setOpenDropdown(null);
                }
            }
        };
        
        if (openDropdown) {
            document.addEventListener('click', handleDocumentClick);
        }
        
        return () => {
            document.removeEventListener('click', handleDocumentClick);
        };
    }, [openDropdown]);

    // Cleanup WebSocket connection on unmount
    useEffect(() => {
        return () => {
            disconnectDownloadWebSocket();
        };
    }, []);

    return (
        <div className="min-h-screen bg-gray-50 flex">
            {/* File Validation Toast */}
            {fileValidationToast && (
                <div className={`fixed top-4 right-4 z-50 max-w-md p-4 rounded-lg shadow-lg border ${
                    fileValidationToast.type === 'success' 
                        ? 'bg-green-50 border-green-200 text-green-800' 
                        : fileValidationToast.type === 'info'
                        ? 'bg-blue-50 border-blue-200 text-blue-800'
                        : 'bg-yellow-50 border-yellow-200 text-yellow-800'
                }`}>
                    <div className="flex items-start">
                        <div className="flex-shrink-0">
                            {fileValidationToast.type === 'success' && (
                                <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                </svg>
                            )}
                            {fileValidationToast.type === 'info' && (
                                <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                            )}
                            {fileValidationToast.type === 'warning' && (
                                <svg className="w-5 h-5 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                            )}
                        </div>
                        <div className="ml-3 flex-1">
                            <p className="text-sm font-medium">{fileValidationToast.message}</p>
                        </div>
                        <div className="ml-4 flex-shrink-0">
                            <button
                                onClick={() => setFileValidationToast(null)}
                                className="inline-flex text-gray-400 hover:text-gray-600"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Sidebar */}
            <Drawer
                variant="permanent"
                sx={{
                    width: 280,
                    flexShrink: 0,
                    '& .MuiDrawer-paper': {
                        width: 280,
                        boxSizing: 'border-box',
                        bgcolor: '#ffffff',
                        borderRight: '1px solid #f1f5f9',
                        boxShadow: '0px 4px 6px rgba(0, 0, 0, 0.07), 0px 2px 4px rgba(0, 0, 0, 0.06)',
                        background: 'linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)',
                    },
                }}
            >
                <Box sx={{ p: 3 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                        <IconButton
                            edge="start"
                            onClick={() => navigate('/chat')}
                            sx={{ 
                                mr: 2,
                                color: '#64748b',
                                borderRadius: '10px',
                                transition: 'all 0.2s ease-in-out',
                                '&:hover': {
                                    backgroundColor: 'rgba(37, 99, 235, 0.08)',
                                    color: '#2563eb',
                                    transform: 'scale(1.05)',
                                },
                            }}
                        >
                            <ArrowBack />
                        </IconButton>
                        <StorageIcon sx={{ 
                            mr: 1, 
                            color: '#2563eb',
                            fontSize: '1.5rem',
                        }} />
                        <Typography variant="h6" sx={{ 
                            fontWeight: 700,
                            color: '#0f172a',
                            fontSize: '1.125rem',
                        }}>
                            Knowledge Hub
                        </Typography>
                    </Box>
                    <Divider sx={{ 
                        mb: 3,
                        borderColor: 'rgba(148, 163, 184, 0.2)',
                    }} />
                    
                    {/* Navigation Items */}
                    <List sx={{ p: 0 }}>
                        <ListItemButton
                            selected={activeTab === 'documents'}
                            onClick={() => setActiveTab('documents')}
                            sx={{ 
                                borderRadius: '12px', 
                                mb: 1,
                                margin: '4px 0',
                                padding: '12px 16px',
                                transition: 'all 0.2s ease-in-out',
                                '&:hover': {
                                    backgroundColor: 'rgba(37, 99, 235, 0.08)',
                                    transform: 'translateX(4px)',
                                },
                                '&.Mui-selected': {
                                    backgroundColor: 'rgba(37, 99, 235, 0.12)',
                                    color: '#2563eb',
                                    '&:hover': {
                                        backgroundColor: 'rgba(37, 99, 235, 0.16)',
                                    },
                                    '& .MuiListItemIcon-root': {
                                        color: '#2563eb',
                                    },
                                    '& .MuiListItemText-primary': {
                                        color: '#2563eb',
                                        fontWeight: 600,
                                    },
                                },
                            }}
                        >
                            <ListItemIcon sx={{ 
                                color: activeTab === 'documents' ? '#2563eb' : '#64748b',
                                minWidth: '40px',
                            }}>
                                <Description />
                            </ListItemIcon>
                            <ListItemText 
                                primary="Documents" 
                                primaryTypographyProps={{
                                    fontSize: '0.875rem',
                                    fontWeight: activeTab === 'documents' ? 600 : 500,
                                    color: activeTab === 'documents' ? '#2563eb' : '#475569',
                                }}
                            />
                        </ListItemButton>

                        <ListItemButton
                            selected={activeTab === 'chunking'}
                            onClick={() => setActiveTab('chunking')}
                            sx={{ 
                                borderRadius: '12px', 
                                mb: 1,
                                margin: '4px 0',
                                padding: '12px 16px',
                                transition: 'all 0.2s ease-in-out',
                                '&:hover': {
                                    backgroundColor: 'rgba(37, 99, 235, 0.08)',
                                    transform: 'translateX(4px)',
                                },
                                '&.Mui-selected': {
                                    backgroundColor: 'rgba(37, 99, 235, 0.12)',
                                    color: '#2563eb',
                                    '&:hover': {
                                        backgroundColor: 'rgba(37, 99, 235, 0.16)',
                                    },
                                    '& .MuiListItemIcon-root': {
                                        color: '#2563eb',
                                    },
                                    '& .MuiListItemText-primary': {
                                        color: '#2563eb',
                                        fontWeight: 600,
                                    },
                                },
                            }}
                        >
                            <ListItemIcon sx={{ 
                                color: activeTab === 'chunking' ? '#2563eb' : '#64748b',
                                minWidth: '40px',
                            }}>
                                <Settings />
                            </ListItemIcon>
                            <ListItemText 
                                primary="Chunking Settings" 
                                primaryTypographyProps={{
                                    fontSize: '0.875rem',
                                    fontWeight: activeTab === 'chunking' ? 600 : 500,
                                    color: activeTab === 'chunking' ? '#2563eb' : '#475569',
                                }}
                            />
                        </ListItemButton>

                        <ListItemButton
                            selected={activeTab === 'retrieval'}
                            onClick={() => setActiveTab('retrieval')}
                            sx={{ 
                                borderRadius: '12px', 
                                mb: 1,
                                margin: '4px 0',
                                padding: '12px 16px',
                                transition: 'all 0.2s ease-in-out',
                                '&:hover': {
                                    backgroundColor: 'rgba(37, 99, 235, 0.08)',
                                    transform: 'translateX(4px)',
                                },
                                '&.Mui-selected': {
                                    backgroundColor: 'rgba(37, 99, 235, 0.12)',
                                    color: '#2563eb',
                                    '&:hover': {
                                        backgroundColor: 'rgba(37, 99, 235, 0.16)',
                                    },
                                    '& .MuiListItemIcon-root': {
                                        color: '#2563eb',
                                    },
                                    '& .MuiListItemText-primary': {
                                        color: '#2563eb',
                                        fontWeight: 600,
                                    },
                                },
                            }}
                        >
                            <ListItemIcon sx={{ 
                                color: activeTab === 'retrieval' ? '#2563eb' : '#64748b',
                                minWidth: '40px',
                            }}>
                                <Search />
                            </ListItemIcon>
                            <ListItemText 
                                primary="Retrieval Settings" 
                                primaryTypographyProps={{
                                    fontSize: '0.875rem',
                                    fontWeight: activeTab === 'retrieval' ? 600 : 500,
                                    color: activeTab === 'retrieval' ? '#2563eb' : '#475569',
                                }}
                            />
                        </ListItemButton>
                        
                        {activeTab === 'documents' && (
                            <Box sx={{ mt: 2, pl: 2, borderLeft: '2px solid #f1f5f9' }}>
                                <label 
                                    style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        padding: '12px 16px',
                                        color: '#475569',
                                        cursor: 'pointer',
                                        borderRadius: '12px',
                                        transition: 'all 0.2s ease-in-out',
                                        fontSize: '0.875rem',
                                        fontWeight: 500,
                                    }}
                                    onMouseEnter={(e) => {
                                        e.target.style.backgroundColor = 'rgba(37, 99, 235, 0.08)';
                                        e.target.style.transform = 'translateX(4px)';
                                    }}
                                    onMouseLeave={(e) => {
                                        e.target.style.backgroundColor = 'transparent';
                                        e.target.style.transform = 'translateX(0)';
                                    }}
                                >
                                    <CloudUploadIcon sx={{ width: 16, height: 16, mr: 1.5 }} />
                                    Upload Files
                                    <input
                                        type="file"
                                        multiple
                                        onChange={handleFileUpload}
                                        style={{ display: 'none' }}
                                        accept=".pdf,.doc,.docx,.txt,.md,.xlsx,.csv"
                                    />
                                </label>
                                
                                {/* Chunking Method Status */}
                                {selectedMethod && (
                                    <Box sx={{ 
                                        px: 2, 
                                        py: 1.5, 
                                        fontSize: '0.75rem', 
                                        bgcolor: 'rgba(37, 99, 235, 0.08)', 
                                        color: '#2563eb', 
                                        borderRadius: '8px',
                                        mt: 1,
                                    }}>
                                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                                            <ArticleIcon sx={{ width: 12, height: 12, mr: 1 }} />
                                            <Typography sx={{ fontWeight: 600, fontSize: '0.75rem' }}>
                                                Chunking Method: {selectedMethod}
                                            </Typography>
                                        </Box>
                                        {methodsData[selectedMethod]?.supported_formats && (
                                            <Typography sx={{ 
                                                fontSize: '0.7rem', 
                                                color: '#2563eb',
                                                opacity: 0.8,
                                                mt: 0.5
                                            }}>
                                                Supports: {methodsData[selectedMethod].supported_formats.join(', ')}
                                            </Typography>
                                        )}
                                    </Box>
                                )}

                                <Box
                                    onClick={handleShowFolderUpload}
                                    sx={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        padding: '12px 16px',
                                        color: '#475569',
                                        cursor: 'pointer',
                                        borderRadius: '12px',
                                        transition: 'all 0.2s ease-in-out',
                                        fontSize: '0.875rem',
                                        fontWeight: 500,
                                        mt: 1,
                                        '&:hover': {
                                            backgroundColor: 'rgba(37, 99, 235, 0.08)',
                                            transform: 'translateX(4px)',
                                        },
                                    }}
                                >
                                    <FolderIcon sx={{ width: 16, height: 16, mr: 1.5 }} />
                                    Upload Folder
                                </Box>
                            </Box>
                        )}
                    </List>
                    
                    {/* Stats */}
                    <Box sx={{ 
                        mt: 3, 
                        p: 2.5, 
                        bgcolor: 'rgba(148, 163, 184, 0.05)', 
                        borderRadius: '12px',
                        border: '1px solid rgba(148, 163, 184, 0.1)',
                    }}>
                        <Typography variant="subtitle2" sx={{ 
                            fontWeight: 600,
                            color: '#0f172a',
                            mb: 1.5,
                            fontSize: '0.875rem',
                        }}>
                            Quick Stats
                        </Typography>
                        <Box sx={{ fontSize: '0.875rem', color: '#64748b' }}>
                            {activeTab === 'documents' && (
                                <>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                        <span>Files:</span>
                                        <Typography sx={{ fontWeight: 600, color: '#2563eb', fontSize: '0.875rem' }}>
                                            {files.length}
                                        </Typography>
                                    </Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <span>Total Size:</span>
                                        <Typography sx={{ fontWeight: 600, color: '#2563eb', fontSize: '0.875rem' }}>
                                            {formatFileSize(files.reduce((total, file) => total + (file.size || 0), 0))}
                                        </Typography>
                                    </Box>
                                </>
                            )}
                        </Box>
                    </Box>
                </Box>
            </Drawer>

            {/* Main Content */}
            <div className="flex-1 flex flex-col">
                {/* Header */}
                <div className="bg-white shadow-sm border-b">
                    <div className="px-6 py-4">
                        <div className="flex justify-between items-center">
                            <div>
                                <h2 className="text-2xl font-bold text-gray-900">
                                    {activeTab === 'documents' && 'Your Documents'}
                                    {activeTab === 'chunking' && 'Chunking Settings'}
                                    {activeTab === 'retrieval' && 'Retrieval Settings'}
                                </h2>
                                <p className="mt-1 text-sm text-gray-500">
                                    {activeTab === 'documents' && 'Manage your uploaded documents for AI conversations'}
                                    {activeTab === 'chunking' && 'Configure how documents are split into chunks for processing'}
                                    {activeTab === 'retrieval' && 'Configure document retrieval and search parameters'}
                                </p>
                            </div>
                            <div className="flex items-center space-x-4">
                                {/* Status Polling Indicator */}
                                {statusPolling && pendingDocuments.size > 0 && (
                                    <div className="flex items-center px-3 py-2 text-sm bg-blue-50 text-blue-700 rounded-lg">
                                        <svg className="w-4 h-4 mr-2 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                        </svg>
                                        Processing {pendingDocuments.size} document{pendingDocuments.size !== 1 ? 's' : ''}
                                    </div>
                                )}
                                
                                <button
                                    onClick={refreshData}
                                    disabled={loading}
                                    className={`flex items-center px-3 py-2 text-sm rounded-lg transition-colors ${
                                        loading 
                                            ? 'bg-gray-50 text-gray-400 cursor-not-allowed' 
                                            : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                                    }`}
                                >
                                    <svg className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                    </svg>
                                    Refresh
                                </button>
                                <div className="text-sm text-gray-500">
                                    {activeTab === 'documents' && `${files.length} document${files.length !== 1 ? 's' : ''}`}
                                    {activeTab === 'chunking' && `${(chunkingMethods || []).length} method${(chunkingMethods || []).length !== 1 ? 's' : ''} available`}
                                    {activeTab === 'retrieval' && `${rerankerModels.length} reranker model${rerankerModels.length !== 1 ? 's' : ''} available`}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 p-6">
                    
                    {/* Upload Progress */}
                    {Object.keys(uploadProgress).length > 0 && (
                        <div className="mb-6 bg-white rounded-lg shadow p-6">
                            <h3 className="text-lg font-medium text-gray-900 mb-4">Uploading Files</h3>
                            {Object.values(uploadProgress).map((file, index) => (
                                <div key={index} className="mb-3">
                                    <div className="flex justify-between items-center mb-1">
                                        <span className="text-sm text-gray-600">{file.name}</span>
                                        <span className="text-sm text-gray-500">{file.progress}%</span>
                                    </div>
                                    <div className="w-full bg-gray-200 rounded-full h-2">
                                        <div
                                            className="bg-indigo-600 h-2 rounded-full transition-all"
                                            style={{ width: `${file.progress}%` }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Documents Tab */}
                    {activeTab === 'documents' && (
                        <div className="bg-white rounded-lg shadow">
                            {loading ? (
                                <div className="p-8 text-center">
                                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
                                    <p className="mt-2 text-gray-500">Loading files...</p>
                                </div>
                            ) : files.length === 0 ? (
                                <div className="p-8 text-center">
                                    <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                    </svg>
                                    <h3 className="mt-2 text-sm font-medium text-gray-900">No documents</h3>
                                    <p className="mt-1 text-sm text-gray-500">
                                        Get started by uploading your first document using the sidebar.
                                    </p>
                                </div>
                            ) : (
                                <>
                                    {/* Search and Bulk Actions */}
                                    <div className="mb-4 space-y-4">
                                        {/* Search and Select All Row */}
                                        <div className="flex flex-wrap items-center justify-between gap-4">
                                            <div className="flex items-center space-x-4">
                                                {/* Search Input */}
                                                <div className="relative">
                                                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                                        <svg className="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                                                        </svg>
                                                    </div>
                                                    <input
                                                        type="text"
                                                        placeholder="Search documents..."
                                                        value={searchTerm}
                                                        onChange={(e) => setSearchTerm(e.target.value)}
                                                        className="pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500 text-sm"
                                                    />
                                                </div>
                                                
                                                {/* Select All Checkbox */}
                                                <label className="flex items-center space-x-2 text-sm text-gray-600 cursor-pointer">
                                                    <input
                                                        type="checkbox"
                                                        checked={selectedDocuments.size === getFilteredFiles().length && getFilteredFiles().length > 0}
                                                        onChange={handleSelectAllDocuments}
                                                        className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                                                    />
                                                    <span>Select All ({getFilteredFiles().length})</span>
                                                </label>
                                            </div>
                                            
                                            {/* Bulk Actions */}
                                            {selectedDocuments.size > 0 && (
                                                <div className="flex items-center space-x-2">
                                                    <span className="text-sm text-gray-600">
                                                        {selectedDocuments.size} selected
                                                    </span>
                                                    <button
                                                        onClick={handleBulkDelete}
                                                        className="px-3 py-2 bg-red-600 text-white text-sm rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500"
                                                    >
                                                        Delete Selected
                                                    </button>
                                                    <button
                                                        onClick={handleBulkReingestion}
                                                        className="px-3 py-2 bg-indigo-600 text-white text-sm rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                                    >
                                                        Reingest Selected
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                        
                                        {/* Table Controls */}
                                        <div className="flex flex-wrap items-center justify-between gap-4">
                                            <div className="text-sm text-gray-600">
                                                💡 <span className="font-medium">Tip:</span> Drag the handles at column borders to resize widths. Filenames wrap automatically.
                                            </div>
                                            <div className="flex items-center space-x-2">
                                                <button 
                                                    onClick={() => {
                                                        // Reset to default column widths
                                                        const table = document.querySelector('.resizable-table');
                                                        if (table) {
                                                            const ths = table.querySelectorAll('th');
                                                            const defaultWidths = ['5%', '25%', '10%', '8%', '12%', '10%', '15%', '15%'];
                                                            ths.forEach((th, index) => {
                                                                if (defaultWidths[index]) {
                                                                    th.style.width = defaultWidths[index];
                                                                }
                                                            });
                                                        }
                                                    }}
                                                    className="text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-1 rounded-md transition-colors"
                                                >
                                                    Reset Columns
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div className="table-container">
                                    <table className="min-w-full divide-y divide-gray-200 resizable-table">
                                        <thead className="bg-gray-50">
                                            <tr>
                                                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" 
                                                    style={{width: '5%', minWidth: '50px'}}>
                                                    <input
                                                        type="checkbox"
                                                        checked={selectedDocuments.size === getFilteredFiles().length && getFilteredFiles().length > 0}
                                                        onChange={handleSelectAllDocuments}
                                                        className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                                                    />
                                                </th>
                                                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" 
                                                    style={{width: '25%', minWidth: '200px'}}>
                                                    File Name
                                                </th>
                                                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider actions-cell" 
                                                    style={{width: '10%', minWidth: '120px'}}>
                                                    Actions
                                                </th>
                                                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" 
                                                    style={{width: '8%', minWidth: '80px'}}>
                                                    Size
                                                </th>
                                                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" 
                                                    style={{width: '12%', minWidth: '100px'}}>
                                                    Uploaded
                                                </th>
                                                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" 
                                                    style={{width: '10%', minWidth: '80px'}}>
                                                    Status
                                                </th>
                                                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" 
                                                    style={{width: '15%', minWidth: '120px'}}>
                                                    Embedding Model
                                                </th>
                                                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider" 
                                                    style={{width: '15%', minWidth: '120px'}}>
                                                    Chunking Method
                                                </th>
                                            </tr>
                                        </thead>
                                        <tbody className="bg-white divide-y divide-gray-200">
                                            {getFilteredFiles().map((file) => (
                                                <tr key={file.filename} className="hover:bg-gray-50">
                                                    <td className="px-3 py-4">
                                                        <input
                                                            type="checkbox"
                                                            checked={selectedDocuments.has(file.id)}
                                                            onChange={() => handleDocumentSelect(file.id)}
                                                            className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                                                        />
                                                    </td>
                                                    <td className="px-4 py-4 filename-cell">
                                                        <div className="flex items-start">
                                                            <div className="flex-shrink-0 h-8 w-8 mt-1">
                                                                <div className="h-8 w-8 rounded-lg bg-indigo-100 flex items-center justify-center">
                                                                    <svg className="h-5 w-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                                                    </svg>
                                                                </div>
                                                            </div>
                                                            <div className="ml-3 min-w-0 flex-1">
                                                                <div className="text-sm font-medium text-gray-900 filename-text" title={file.filename}>
                                                                    {extractFilename(file.filename)}
                                                                </div>
                                                                {file.filename.includes('/') && (
                                                                    <div className="text-xs text-gray-500 truncate" title={file.filename}>
                                                                        {file.filename}
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </div>
                                                    </td>
                                                    <td className="px-3 py-4 text-sm font-medium actions-cell">
                                                        <div className="relative inline-block text-left dropdown-container">
                                                            <button
                                                                type="button"
                                                                onClick={(e) => {
                                                                    console.log('Button clicked for file:', file.filename);
                                                                    toggleDropdown(file.filename, e);
                                                                }}
                                                                className="actions-button px-2 py-1 border border-gray-300 shadow-sm text-xs leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                                                            >
                                                                Actions
                                                                <svg className="ml-1 -mr-0.5 h-3 w-3 flex-shrink-0" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                                                    <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                                                                </svg>
                                                            </button>
                                                        </div>
                                                        {openDropdown && openDropdown === file.filename && (
                                                            <div className="fixed bg-white ring-1 ring-black ring-opacity-5 focus:outline-none shadow-xl rounded-md w-48 py-1"
                                                                 style={{
                                                                     position: 'fixed',
                                                                     top: `${dropdownPosition.top}px`,
                                                                     left: `${dropdownPosition.left}px`,
                                                                     zIndex: 9999
                                                                 }}
                                                                 role="menu">
                                                                <button
                                                                    onClick={(e) => {
                                                                        e.stopPropagation();
                                                                        setOpenDropdown(null);
                                                                        navigate(`/documents/${file.id}/chunks`);
                                                                    }}
                                                                    className="text-blue-600 block px-4 py-2 text-sm hover:bg-gray-100 w-full text-left flex items-center"
                                                                    role="menuitem"
                                                                >
                                                                    📄 View Chunks
                                                                </button>
                                                                {getDocumentStatus(file).status === 'failed' && (
                                                                    <button
                                                                        onClick={(e) => {
                                                                            e.stopPropagation();
                                                                            setOpenDropdown(null);
                                                                            handleRetryDocument(file);
                                                                        }}
                                                                        className="text-orange-600 block px-4 py-2 text-sm hover:bg-gray-100 w-full text-left flex items-center"
                                                                        role="menuitem"
                                                                    >
                                                                        🔄 Retry Processing
                                                                    </button>
                                                                )}
                                                                <button
                                                                    onClick={(e) => {
                                                                        e.stopPropagation();
                                                                        setOpenDropdown(null);
                                                                        handleDeleteFile(file);
                                                                    }}
                                                                    className="text-red-600 block px-4 py-2 text-sm hover:bg-gray-100 w-full text-left"
                                                                    role="menuitem"
                                                                >
                                                                    Delete
                                                                </button>
                                                            </div>
                                                        )}
                                                    </td>
                                                    <td className="px-3 py-4 text-sm text-gray-500">
                                                        <div className="text-truncate-multiline" title={formatFileSize(file.size)}>
                                                            {formatFileSize(file.size)}
                                                        </div>
                                                    </td>
                                                    <td className="px-3 py-4 text-sm text-gray-500">
                                                        <div className="text-truncate-multiline" title={formatDate(file.upload_date)}>
                                                            {formatDate(file.upload_date)}
                                                        </div>
                                                    </td>
                                                    <td className="px-3 py-4">
                                                        {(() => {
                                                            const docStatus = getDocumentStatus(file);
                                                            return (
                                                                <div className="flex items-center">
                                                                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${docStatus.color}`}>
                                                                        {docStatus.label}
                                                                    </span>
                                                                    {docStatus.status === 'failed' && (
                                                                        <button
                                                                            onClick={() => {
                                                                                alert(getErrorMessage(file));
                                                                            }}
                                                                            className="ml-2 text-red-600 hover:text-red-800"
                                                                            title="View error details"
                                                                        >
                                                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                                            </svg>
                                                                        </button>
                                                                    )}
                                                                </div>
                                                            );
                                                        })()}
                                                    </td>
                                                    <td className="px-3 py-4">
                                                        <span className="inline-flex px-2 py-1 text-xs font-medium rounded-md bg-gray-100 text-gray-700 text-truncate-multiline" title={file.embedding_model || 'Unknown'}>
                                                            {file.embedding_model || 'Unknown'}
                                                        </span>
                                                    </td>
                                                    <td className="px-3 py-4">
                                                        <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-md text-truncate-multiline ${getChunkingMethodStyle(file.chunking_method)}`} title={formatChunkingMethod(file.chunking_method)}>
                                                            {formatChunkingMethod(file.chunking_method)}
                                                        </span>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                                </>
                            )}
                            
                            {/* Document Reingestion Modal */}
                            <DocumentReingestionModal
                                selectedDocuments={selectedDocuments}
                                onReingestion={handleConfirmReingestion}
                                onCancel={() => setShowReingestionDialog(false)}
                                isVisible={showReingestionDialog}
                                isProcessing={isReingesting}
                                documents={files}
                                defaultConfigs={defaultConfigs}
                            />
                        </div>
                    )}

                    {/* Chunking Settings Tab */}
                    {activeTab === 'chunking' && (
                        <div className="space-y-6">
                            {/* Loading State */}
                            {loadingChunking && (
                                <div className="bg-white rounded-lg shadow p-6">
                                    <div className="flex items-center justify-center">
                                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-purple-600"></div>
                                        <span className="ml-2 text-gray-600">Loading chunking methods...</span>
                                    </div>
                                </div>
                            )}

                            {/* Error State */}
                            {chunkingError && (
                                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                                    <div className="flex items-center">
                                        <svg className="w-5 h-5 text-red-400 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                        <span className="text-red-800">{chunkingError}</span>
                                    </div>
                                    <button
                                        onClick={loadChunkingMethods}
                                        className="mt-2 text-sm bg-red-100 hover:bg-red-200 text-red-800 px-3 py-1 rounded transition-colors"
                                    >
                                        Retry
                                    </button>
                                </div>
                            )}

                            {/* Content when loaded successfully */}
                            {!loadingChunking && !chunkingError && (chunkingMethods || []).length > 0 && (
                                <>
                            {/* Method Selection */}
                            <div className="bg-white rounded-lg shadow">
                                <div className="px-6 py-4 border-b border-gray-200">
                                    <h3 className="text-lg font-medium text-gray-900">Chunking Method</h3>
                                    <p className="mt-1 text-sm text-gray-500">Select and configure your document chunking strategy</p>
                                </div>
                                <div className="p-6">
                                    <div className="space-y-4">
                                        <div>
                                            <label htmlFor="chunking-method" className="block text-sm font-medium text-gray-700 mb-2">
                                                Chunking Method
                                            </label>
                                            <select
                                                id="chunking-method"
                                                value={selectedMethod}
                                                onChange={(e) => handleMethodChange(e.target.value)}
                                                className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-purple-500 focus:border-purple-500"
                                            >
                                                {(chunkingMethods || []).map(method => (
                                                    <option key={method} value={method}>
                                                        {method.charAt(0).toUpperCase() + method.slice(1)}
                                                    </option>
                                                ))}
                                            </select>
                                            
                                            {/* Method recommendations */}
                                            <div className="mt-2 text-xs text-gray-500">
                                                <div className="grid grid-cols-2 gap-2">
                                                    <div>
                                                        <span className="font-medium">📄 PDF:</span> General, Q&A
                                                    </div>
                                                    <div>
                                                        <span className="font-medium">📝 Word:</span> General, Resume, Q&A
                                                    </div>
                                                    <div>
                                                        <span className="font-medium">📊 PowerPoint:</span> Presentation
                                                    </div>
                                                    <div>
                                                        <span className="font-medium">🖼️ Images:</span> Picture
                                                    </div>
                                                    <div>
                                                        <span className="font-medium">📈 Excel:</span> Table
                                                    </div>
                                                    <div>
                                                        <span className="font-medium">📧 Email:</span> Email
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Configuration Settings */}
                            {activeConfig && (
                                <div className="bg-white rounded-lg shadow">
                                    <div className="px-6 py-4 border-b border-gray-200">
                                        <h3 className="text-lg font-medium text-gray-900">Configuration Settings</h3>
                                        <p className="mt-1 text-sm text-gray-500">Configure parameters for the {selectedMethod} chunking method</p>
                                    </div>
                                    <div className="p-6">
                                        <div className="space-y-6">
                                            {/* Chunk Token Number */}
                                            {activeConfig.chunk_token_num !== undefined && (
                                                <div>
                                                    <label htmlFor="chunk-token-num" className="block text-sm font-medium text-gray-700 mb-2">
                                                        Chunk Token Number
                                                    </label>
                                                    <input
                                                        type="number"
                                                        id="chunk-token-num"
                                                        value={activeConfig.chunk_token_num}
                                                        onChange={(e) => handleConfigChange('chunk_token_num', parseInt(e.target.value))}
                                                        className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-purple-500 focus:border-purple-500"
                                                        min="100"
                                                        max="8000"
                                                    />
                                                    <p className="mt-1 text-xs text-gray-500">Number of tokens per chunk (100-8000)</p>
                                                </div>
                                            )}

                                            {/* Max Token */}
                                            {activeConfig.max_token !== undefined && (
                                                <div>
                                                    <label htmlFor="max-token" className="block text-sm font-medium text-gray-700 mb-2">
                                                        Maximum Tokens
                                                    </label>
                                                    <input
                                                        type="number"
                                                        id="max-token"
                                                        value={activeConfig.max_token}
                                                        onChange={(e) => handleConfigChange('max_token', parseInt(e.target.value))}
                                                        className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-purple-500 focus:border-purple-500"
                                                        min="512"
                                                        max="32768"
                                                    />
                                                    <p className="mt-1 text-xs text-gray-500">Maximum tokens allowed per chunk (512-32768)</p>
                                                </div>
                                            )}

                                            {/* Chunk Size */}
                                            {activeConfig.chunk_size !== undefined && (
                                                <div>
                                                    <label htmlFor="chunk-size" className="block text-sm font-medium text-gray-700 mb-2">
                                                        Chunk Size
                                                    </label>
                                                    <input
                                                        type="number"
                                                        id="chunk-size"
                                                        value={activeConfig.chunk_size}
                                                        onChange={(e) => handleConfigChange('chunk_size', parseInt(e.target.value))}
                                                        className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-purple-500 focus:border-purple-500"
                                                        min="100"
                                                        max="8000"
                                                    />
                                                    <p className="mt-1 text-xs text-gray-500">Number of characters per chunk (100-8000)</p>
                                                </div>
                                            )}

                                            {/* Chunk Overlap */}
                                            {activeConfig.chunk_overlap !== undefined && (
                                                <div>
                                                    <label htmlFor="chunk-overlap" className="block text-sm font-medium text-gray-700 mb-2">
                                                        Chunk Overlap
                                                    </label>
                                                    <input
                                                        type="number"
                                                        id="chunk-overlap"
                                                        value={activeConfig.chunk_overlap}
                                                        onChange={(e) => handleConfigChange('chunk_overlap', parseInt(e.target.value))}
                                                        className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-purple-500 focus:border-purple-500"
                                                        min="0"
                                                        max="1000"
                                                    />
                                                    <p className="mt-1 text-xs text-gray-500">Number of characters to overlap between chunks (0-1000)</p>
                                                </div>
                                            )}

                                            {/* Text separators for chunking */}
                                            {activeConfig.separators !== undefined && (
                                                <div>
                                                    <label htmlFor="separators" className="block text-sm font-medium text-gray-700 mb-2">
                                                        Separators
                                                    </label>
                                                    <textarea
                                                        id="separators"
                                                        value={Array.isArray(activeConfig.separators) ? activeConfig.separators.join('\n') : activeConfig.separators}
                                                        onChange={(e) => handleConfigChange('separators', e.target.value.split('\n').filter(s => s.trim()))}
                                                        className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-purple-500 focus:border-purple-500"
                                                        rows="3"
                                                        placeholder="Enter separators (one per line)"
                                                    />
                                                    <p className="mt-1 text-xs text-gray-500">Text separators used to split documents (one per line)</p>
                                                </div>
                                            )}

                                            {/* Additional config fields can be added based on method */}
                                            {Object.entries(activeConfig).map(([key, value]) => {
                                                if (['chunk_size', 'chunk_overlap', 'chunk_token_num', 'max_token', 'separators', 'method'].includes(key)) return null;
                                                if (typeof value === 'boolean') {
                                                    return (
                                                        <div key={key} className="flex items-center">
                                                            <input
                                                                type="checkbox"
                                                                id={key}
                                                                checked={value}
                                                                onChange={(e) => handleConfigChange(key, e.target.checked)}
                                                                className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                                                            />
                                                            <label htmlFor={key} className="ml-2 block text-sm text-gray-700">
                                                                {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                                            </label>
                                                        </div>
                                                    );
                                                } else if (typeof value === 'number') {
                                                    return (
                                                        <div key={key}>
                                                            <label htmlFor={key} className="block text-sm font-medium text-gray-700 mb-2">
                                                                {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                                            </label>
                                                            <input
                                                                type="number"
                                                                id={key}
                                                                value={value}
                                                                onChange={(e) => handleConfigChange(key, parseFloat(e.target.value))}
                                                                className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-purple-500 focus:border-purple-500"
                                                            />
                                                        </div>
                                                    );
                                                } else if (typeof value === 'string') {
                                                    return (
                                                        <div key={key}>
                                                            <label htmlFor={key} className="block text-sm font-medium text-gray-700 mb-2">
                                                                {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                                            </label>
                                                            <input
                                                                type="text"
                                                                id={key}
                                                                value={value}
                                                                onChange={(e) => handleConfigChange(key, e.target.value)}
                                                                className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-purple-500 focus:border-purple-500"
                                                            />
                                                        </div>
                                                    );
                                                }
                                                return null;
                                            })}
                                        </div>

                                        {/* Save Message */}
                                        {saveMessage && (
                                            <div className={`mb-4 p-3 rounded-lg ${
                                                saveMessage.type === 'success' 
                                                    ? 'bg-green-50 border border-green-200 text-green-800' 
                                                    : 'bg-red-50 border border-red-200 text-red-800'
                                            }`}>
                                                <div className="flex items-center">
                                                    {saveMessage.type === 'success' ? (
                                                        <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                                        </svg>
                                                    ) : (
                                                        <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                        </svg>
                                                    )}
                                                    <span className="text-sm font-medium">{saveMessage.text}</span>
                                                </div>
                                            </div>
                                        )}

                                        {/* Save and Reset Buttons */}
                                        <div className="mt-6 pt-6 border-t border-gray-200">
                                            <div className="flex space-x-3">
                                                <button
                                                    onClick={saveChunkingConfig}
                                                    disabled={savingConfig}
                                                    className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 text-white px-4 py-2 rounded-lg transition-colors"
                                                >
                                                    {savingConfig ? 'Saving...' : 'Save Configuration'}
                                                </button>
                                                <button
                                                    onClick={resetChunkingConfig}
                                                    disabled={savingConfig}
                                                    className="px-4 py-2 bg-gray-200 hover:bg-gray-300 disabled:opacity-50 text-gray-700 rounded-lg transition-colors"
                                                    title="Reset to default values"
                                                >
                                                    Reset
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Method Information */}
                            <div className="bg-white rounded-lg shadow">
                                <div className="px-6 py-4 border-b border-gray-200">
                                    <h3 className="text-lg font-medium text-gray-900">Method Information</h3>
                                </div>
                                <div className="p-6">
                                    <div className="text-sm text-gray-600">
                                        {selectedMethod === 'general' && (
                                            <div>
                                                <p><strong>General Chunking:</strong> Smart text splitting based on document structure and content.</p>
                                                <p className="mt-2">Best for: General documents, articles, and plain text files.</p>
                                            </div>
                                        )}
                                        {selectedMethod === 'qa' && (
                                            <div>
                                                <p><strong>Q&A Chunking:</strong> Optimized for question-answer format documents.</p>
                                                <p className="mt-2">Best for: FAQ documents, interview transcripts, and Q&A datasets.</p>
                                            </div>
                                        )}
                                        {selectedMethod === 'resume' && (
                                            <div>
                                                <p><strong>Resume Chunking:</strong> Specialized for resume and CV documents.</p>
                                                <p className="mt-2">Best for: Resume databases, CV collections, and professional profiles.</p>
                                            </div>
                                        )}
                                        {!['general', 'qa', 'resume'].includes(selectedMethod) && (
                                            <div>
                                                <p><strong>{selectedMethod.charAt(0).toUpperCase() + selectedMethod.slice(1)} Chunking:</strong> Specialized chunking method.</p>
                                                <p className="mt-2">Configure the parameters above to optimize for your specific use case.</p>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                                </>
                            )}

                            {/* Empty state when no methods available */}
                            {!loadingChunking && !chunkingError && (chunkingMethods || []).length === 0 && (
                                <div className="bg-white rounded-lg shadow p-6">
                                    <div className="text-center text-gray-500">
                                        <svg className="w-12 h-12 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                        </svg>
                                        <p>No chunking methods available</p>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Retrieval Settings Tab */}
                    {activeTab === 'retrieval' && (
                        <div className="space-y-6">
                            {/* Loading State */}
                            {loadingRetrievalConfig && (
                                <div className="bg-white rounded-lg shadow p-6">
                                    <div className="flex items-center justify-center">
                                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-green-600"></div>
                                        <span className="ml-2 text-gray-600">Loading retrieval configuration...</span>
                                    </div>
                                </div>
                            )}

                            {/* Retrieval Settings Form */}
                            {!loadingRetrievalConfig && (
                                <div className="bg-white rounded-lg shadow">
                                    <div className="p-6">
                                        <div className="flex justify-between items-center mb-6">
                                            <h3 className="text-lg font-medium text-gray-900">Retrieval Configuration</h3>
                                            <div className="flex space-x-3">
                                                <button
                                                    onClick={resetRetrievalConfig}
                                                    className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 border border-gray-300 rounded-lg hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
                                                >
                                                    Reset to Defaults
                                                </button>
                                                <button
                                                    onClick={saveRetrievalConfig}
                                                    disabled={savingRetrievalConfig || downloadingRerankerModel}
                                                    className={`px-4 py-2 text-sm font-medium text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                                                        savingRetrievalConfig || downloadingRerankerModel
                                                            ? 'bg-gray-400 cursor-not-allowed'
                                                            : 'bg-green-600 hover:bg-green-700 focus:ring-green-500'
                                                    }`}
                                                >
                                                    {savingRetrievalConfig 
                                                        ? 'Saving...' 
                                                        : downloadingRerankerModel 
                                                            ? `Downloading... ${downloadProgress}`
                                                            : 'Save Configuration'
                                                    }
                                                </button>
                                            </div>
                                        </div>

                                        {/* Save Message */}
                                        {retrievalConfigMessage && (
                                            <div className={`mb-4 p-3 rounded-lg ${
                                                retrievalConfigMessage.type === 'success' 
                                                    ? 'bg-green-50 text-green-700 border border-green-200' 
                                                    : retrievalConfigMessage.type === 'warning'
                                                    ? 'bg-yellow-50 text-yellow-700 border border-yellow-200'
                                                    : 'bg-red-50 text-red-700 border border-red-200'
                                            }`}>
                                                {retrievalConfigMessage.text}
                                            </div>
                                        )}

                                        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                                            {/* Similarity Threshold */}
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                                    Similarity Threshold
                                                    <span className="text-xs text-gray-500 ml-1">(0.0 - 1.0)</span>
                                                </label>
                                                <input
                                                    type="number"
                                                    min="0"
                                                    max="1"
                                                    step="0.1"
                                                    value={retrievalConfig.similarity_threshold}
                                                    onChange={(e) => handleRetrievalConfigChange('similarity_threshold', parseFloat(e.target.value) || 0)}
                                                    className="mt-1 block w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500"
                                                />
                                                <p className="mt-1 text-xs text-gray-500">
                                                    Minimum similarity score for retrieving chunks. Higher values are more selective.
                                                </p>
                                            </div>

                                            {/* Keyword Similarity Weight */}
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                                    Keyword Similarity Weight
                                                    <span className="text-xs text-gray-500 ml-1">(0.0 - 1.0)</span>
                                                </label>
                                                <input
                                                    type="number"
                                                    min="0"
                                                    max="1"
                                                    step="0.1"
                                                    value={retrievalConfig.keyword_similarity_weight}
                                                    onChange={(e) => handleRetrievalConfigChange('keyword_similarity_weight', parseFloat(e.target.value) || 0)}
                                                    className="mt-1 block w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500"
                                                />
                                                <p className="mt-1 text-xs text-gray-500">
                                                    Weight of keyword similarity vs semantic similarity (only used with Hybrid search type). 1.0 = pure keyword search, 0.0 = pure semantic search.
                                                </p>
                                            </div>

                                            {/* Max Chunks */}
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                                    Maximum Chunks
                                                    <span className="text-xs text-gray-500 ml-1">(1 - 20)</span>
                                                </label>
                                                <input
                                                    type="number"
                                                    min="1"
                                                    max="20"
                                                    value={retrievalConfig.max_chunks}
                                                    onChange={(e) => handleRetrievalConfigChange('max_chunks', parseInt(e.target.value) || 1)}
                                                    className="mt-1 block w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500"
                                                />
                                                <p className="mt-1 text-xs text-gray-500">
                                                    Maximum number of document chunks to retrieve for answering questions.
                                                </p>
                                            </div>

                                            {/* Search Type */}
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                                    Search Type
                                                </label>
                                                <select
                                                    value={retrievalConfig.search_type}
                                                    onChange={(e) => handleRetrievalConfigChange('search_type', e.target.value)}
                                                    className="mt-1 block w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500"
                                                >
                                                    <option value="similarity">Similarity Search</option>
                                                    <option value="mmr">Maximum Marginal Relevance (MMR)</option>
                                                    <option value="similarity_score_threshold">Similarity with Score Threshold</option>
                                                    <option value="hybrid">Hybrid (Semantic + Keyword)</option>
                                                </select>
                                                <p className="mt-1 text-xs text-gray-500">
                                                    Search algorithm to use for document retrieval. Hybrid combines semantic similarity with keyword matching.
                                                </p>
                                            </div>
                                        </div>

                                        {/* Reranker Settings */}
                                        <div className="mt-8 pt-6 border-t border-gray-200">
                                            <h4 className="text-md font-medium text-gray-900 mb-6">Reranker Configuration</h4>
                                            
                                            <div className="space-y-6">
                                                {/* Enable Reranker */}
                                                <div className="flex items-center">
                                                    <input
                                                        id="reranker-enabled"
                                                        type="checkbox"
                                                        checked={retrievalConfig.reranker_enabled}
                                                        onChange={(e) => handleRetrievalConfigChange('reranker_enabled', e.target.checked)}
                                                        className="h-4 w-4 text-green-600 focus:ring-green-500 border-gray-300 rounded"
                                                    />
                                                    <label htmlFor="reranker-enabled" className="ml-2 block text-sm text-gray-900">
                                                        Enable Reranker Model
                                                    </label>
                                                </div>

                                                {/* Reranker Provider and Model Selection */}
                                                {retrievalConfig.reranker_enabled && (
                                                    <ThemeProvider theme={theme}>
                                                        <div className="space-y-6">
                                                            {/* Provider Selection */}
                                                            <div>
                                                                <label className="block text-sm font-medium text-gray-700 mb-3">
                                                                    Model Provider
                                                                </label>
                                                                <div className="grid grid-cols-2 gap-4">
                                                                    {/* Ollama Provider */}
                                                                    <Card 
                                                                        className={`cursor-pointer transition-all duration-200 ${
                                                                            selectedRerankerProvider === 'ollama' 
                                                                                ? 'ring-2 ring-indigo-500 bg-indigo-50' 
                                                                                : 'hover:shadow-md'
                                                                        }`}
                                                                        onClick={() => setSelectedRerankerProvider('ollama')}
                                                                    >
                                                                        <CardContent className="p-4">
                                                                            <div className="flex items-center space-x-3">
                                                                                <Avatar className="h-8 w-8">
                                                                                    <img 
                                                                                        src={ollamaIcon} 
                                                                                        alt="Ollama" 
                                                                                        className="h-8 w-8"
                                                                                    />
                                                                                </Avatar>
                                                                                <div className="flex-1">
                                                                                    <h3 className="text-sm font-medium text-gray-900">Ollama</h3>
                                                                                    <p className="text-xs text-gray-500">Local models</p>
                                                                                </div>
                                                                                {selectedRerankerProvider === 'ollama' && (
                                                                                    <CheckCircleIcon className="h-5 w-5 text-indigo-600" />
                                                                                )}
                                                                            </div>
                                                                        </CardContent>
                                                                    </Card>

                                                                    {/* HuggingFace Provider */}
                                                                    <Card 
                                                                        className={`cursor-pointer transition-all duration-200 ${
                                                                            selectedRerankerProvider === 'huggingface' 
                                                                                ? 'ring-2 ring-indigo-500 bg-indigo-50' 
                                                                                : 'hover:shadow-md'
                                                                        }`}
                                                                        onClick={() => setSelectedRerankerProvider('huggingface')}
                                                                    >
                                                                        <CardContent className="p-4">
                                                                            <div className="flex items-center space-x-3">
                                                                                <Avatar className="h-8 w-8">
                                                                                    <img 
                                                                                        src={huggingfaceIcon} 
                                                                                        alt="HuggingFace" 
                                                                                        className="h-8 w-8"
                                                                                    />
                                                                                </Avatar>
                                                                                <div className="flex-1">
                                                                                    <h3 className="text-sm font-medium text-gray-900">HuggingFace</h3>
                                                                                    <p className="text-xs text-gray-500">Cloud models</p>
                                                                                </div>
                                                                                {selectedRerankerProvider === 'huggingface' && (
                                                                                    <CheckCircleIcon className="h-5 w-5 text-indigo-600" />
                                                                                )}
                                                                            </div>
                                                                        </CardContent>
                                                                    </Card>
                                                                </div>
                                                            </div>

                                                            {/* Model Selection */}
                                                            <div>
                                                                <Box sx={{ minWidth: 120 }}>
                                                                    <FormControl fullWidth variant="outlined">
                                                                        <InputLabel id="reranker-model-label">Reranker Model</InputLabel>
                                                                        <Select
                                                                            labelId="reranker-model-label"
                                                                            value={retrievalConfig.reranker_model || ''}
                                                                            label="Reranker Model"
                                                                            onChange={(e) => handleRetrievalConfigChange('reranker_model', e.target.value)}
                                                                            size="small"
                                                                            disabled={loadingRerankerModels}
                                                                        >
                                                                            {loadingRerankerModels ? (
                                                                                <MenuItem disabled>
                                                                                    <em>Loading models...</em>
                                                                                </MenuItem>
                                                                            ) : rerankerModels.length === 0 ? (
                                                                                <MenuItem disabled>
                                                                                    <em>No models available for {selectedRerankerProvider}</em>
                                                                                </MenuItem>
                                                                            ) : (
                                                                                rerankerModels.map((model) => (
                                                                                    <MenuItem key={model.name} value={model.name}>
                                                                                        <div className="flex items-center justify-between w-full">
                                                                                            <span>{model.display_name || model.name}</span>
                                                                                            {model.size && (
                                                                                                <Chip 
                                                                                                    label={model.size} 
                                                                                                    size="small" 
                                                                                                    variant="outlined"
                                                                                                    sx={{ ml: 1, fontSize: '0.7rem' }}
                                                                                                />
                                                                                            )}
                                                                                        </div>
                                                                                    </MenuItem>
                                                                                ))
                                                                            )}
                                                                        </Select>
                                                                    </FormControl>
                                                                </Box>
                                                                <p className="mt-2 text-xs text-gray-500">
                                                                    {rerankerModels.find(m => m.name === retrievalConfig.reranker_model)?.description || 
                                                                     'Select a reranker model to improve retrieval relevance by re-scoring and re-ordering retrieved documents'}
                                                                </p>
                                                            </div>
                                                        </div>
                                                    </ThemeProvider>
                                                )}
                                            </div>
                                        </div>

                                        {/* Auto Merging Settings */}
                                        <div className="mt-8 pt-6 border-t border-gray-200">
                                            <h4 className="text-md font-medium text-gray-900 mb-4">Auto Merging Retrieval</h4>
                                            
                                            <div className="space-y-4">
                                                {/* Enable Auto Merging */}
                                                <div className="flex items-center">
                                                    <input
                                                        id="auto-merging-enabled"
                                                        type="checkbox"
                                                        checked={retrievalConfig.auto_merging_enabled}
                                                        onChange={(e) => handleRetrievalConfigChange('auto_merging_enabled', e.target.checked)}
                                                        className="h-4 w-4 text-green-600 focus:ring-green-500 border-gray-300 rounded"
                                                    />
                                                    <label htmlFor="auto-merging-enabled" className="ml-2 block text-sm text-gray-900">
                                                        Enable Auto Merging
                                                    </label>
                                                </div>
                                                
                                                <p className="text-xs text-gray-500">
                                                    Auto merging combines similar document chunks to provide more comprehensive context. 
                                                    Applied after reranking when both are enabled.
                                                </p>

                                                {/* Auto Merging Similarity Threshold */}
                                                {retrievalConfig.auto_merging_enabled && (
                                                    <div>
                                                        <label className="block text-sm font-medium text-gray-700 mb-2">
                                                            Merging Similarity Threshold: {retrievalConfig.auto_merging_similarity_threshold}
                                                        </label>
                                                        <input
                                                            type="range"
                                                            min="0.5"
                                                            max="1.0"
                                                            step="0.05"
                                                            value={retrievalConfig.auto_merging_similarity_threshold}
                                                            onChange={(e) => handleRetrievalConfigChange('auto_merging_similarity_threshold', parseFloat(e.target.value))}
                                                            className="mt-1 block w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                                                        />
                                                        <p className="mt-1 text-xs text-gray-500">
                                                            Higher values (0.8-1.0) merge only very similar chunks. Lower values (0.5-0.7) merge more broadly related content.
                                                        </p>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                </div>
            </div>

            {/* Warning Dialog Modal */}
            {warningDialog && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-white rounded-lg p-6 m-4 max-w-lg w-full">
                        <div className="flex items-center mb-4">
                            <div className="flex-shrink-0">
                                <svg className="w-6 h-6 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                                </svg>
                            </div>
                            <div className="ml-3">
                                <h3 className="text-lg font-medium text-gray-900">File Format Warning</h3>
                            </div>
                        </div>
                        
                        <div className="mb-6">
                            <p className="text-sm text-gray-600 mb-3">
                                {warningDialog.message}
                            </p>
                            
                            {/* File Information */}
                            <div className="bg-gray-50 p-4 rounded-md mb-4">
                                <div className="flex items-center mb-2">
                                    <span className="text-lg mr-2">{warningDialog.fileTypeInfo?.icon || '📄'}</span>
                                    <div>
                                        <p className="text-sm font-medium text-gray-900">{warningDialog.fileName}</p>
                                        <p className="text-xs text-gray-600">
                                            {warningDialog.fileTypeInfo?.type || 'Unknown'} file (.{warningDialog.fileExtension})
                                        </p>
                                    </div>
                                </div>
                                <div className="text-xs text-gray-600 space-y-1">
                                    <p><strong>Selected method:</strong> {warningDialog.method.charAt(0).toUpperCase() + warningDialog.method.slice(1)}</p>
                                    <p><strong>Supported formats:</strong> {warningDialog.supportedFormats.join(', ')}</p>
                                    {warningDialog.recommendedMethods && (
                                        <p><strong>Recommended methods:</strong> {warningDialog.recommendedMethods.map(m => m.charAt(0).toUpperCase() + m.slice(1)).join(', ')}</p>
                                    )}
                                </div>
                            </div>
                            
                            {/* Recommendations */}
                            {warningDialog.recommendedMethods && warningDialog.recommendedMethods.length > 0 && (
                                <div className="mb-4">
                                    <p className="text-sm font-medium text-gray-900 mb-2">💡 Recommended actions:</p>
                                    <div className="space-y-2">
                                        {warningDialog.recommendedMethods.filter(method => chunkingMethods.includes(method)).map(method => (
                                            <button
                                                key={method}
                                                onClick={() => warningDialog.onSwitchMethod(method)}
                                                className="w-full text-left px-3 py-2 text-sm bg-blue-50 hover:bg-blue-100 border border-blue-200 rounded-md transition-colors"
                                            >
                                                <span className="font-medium">Switch to '{method.charAt(0).toUpperCase() + method.slice(1)}' method</span>
                                                <span className="text-blue-600 block text-xs">Best for {warningDialog.fileTypeInfo?.type || 'this file type'}</span>
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}
                            
                            <p className="text-sm text-gray-600">
                                Or continue uploading with the 'General' chunking method (fallback option).
                            </p>
                        </div>
                        
                        <div className="flex justify-end space-x-3">
                            <button
                                onClick={warningDialog.onCancel}
                                className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={warningDialog.onConfirm}
                                className="px-4 py-2 border border-transparent rounded-md text-sm font-medium text-white bg-yellow-600 hover:bg-yellow-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500"
                            >
                                Use General Method
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Folder Upload Review Modal */}
            <FolderUploadReview
                chunkingMethods={chunkingMethods}
                onUpload={handleFolderUpload}
                onCancel={handleCancelFolderUpload}
                defaultConfigs={defaultConfigs}
                isVisible={showFolderUploadReview}
            />

            {/* File Upload Review Modal */}
            <FileUploadReview
                chunkingMethods={chunkingMethods}
                onUpload={handleFileUploadFromReview}
                onCancel={() => setShowFileUploadReview(false)}
                defaultConfigs={defaultConfigs}
                isVisible={showFileUploadReview}
                selectedFiles={selectedFilesForReview}
            />

        </div>
    );
};

export default KnowledgeHubPage;
