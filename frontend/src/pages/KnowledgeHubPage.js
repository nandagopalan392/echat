import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import FolderUploadReview from '../components/FolderUploadReview';
import FileUploadReview from '../components/FileUploadReview';
import DocumentReingestionModal from '../components/DocumentReingestionModal';

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
    const [showAdvancedReingestionConfig, setShowAdvancedReingestionConfig] = useState(false);

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
    }, []);

    const loadFiles = async () => {
        try {
            setLoading(true);
            const response = await api.listFiles();
            setFiles(response.files || []);
        } catch (error) {
            console.error('Error loading files:', error);
        } finally {
            setLoading(false);
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
        if (!file.model_status) {
            return { status: 'pending', label: 'Pending', color: 'bg-yellow-100 text-yellow-800' };
        }
        
        // Check if any model has failed status
        const statuses = Object.values(file.model_status);
        const hasFailed = statuses.includes('failed');
        const hasCompleted = statuses.includes('completed');
        const hasPending = statuses.includes('pending');
        
        if (hasFailed && !hasCompleted) {
            return { status: 'failed', label: 'Failed', color: 'bg-red-100 text-red-800' };
        } else if (hasCompleted) {
            return { status: 'completed', label: 'Indexed', color: 'bg-green-100 text-green-800' };
        } else if (hasPending) {
            return { status: 'pending', label: 'Pending', color: 'bg-yellow-100 text-yellow-800' };
        } else {
            return { status: 'unknown', label: 'Unknown', color: 'bg-gray-100 text-gray-800' };
        }
    };

    // Helper function to get error message for failed documents
    const getErrorMessage = (file) => {
        // This would need to be added to the backend response - for now return a generic message
        return 'Document processing failed. Please check the file format and try re-uploading.';
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
            
            // Show success message
            showValidationToast(
                `✅ Folder "${folderName}" uploaded successfully with ${files.length} files!`,
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
            showValidationToast(
                `✅ Uploaded ${files.length} file${files.length !== 1 ? 's' : ''} successfully!`,
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
            'naive': 'bg-blue-100 text-blue-700',
            'qa': 'bg-green-100 text-green-700',
            'resume': 'bg-purple-100 text-purple-700',
            'manual': 'bg-orange-100 text-orange-700',
            'table': 'bg-yellow-100 text-yellow-700',
            'laws': 'bg-red-100 text-red-700',
            'presentation': 'bg-pink-100 text-pink-700',
            'picture': 'bg-indigo-100 text-indigo-700',
            'one': 'bg-gray-100 text-gray-700',
            'email': 'bg-teal-100 text-teal-700'
        };
        return styleMap[method] || 'bg-blue-100 text-blue-700';
    };

    const refreshData = async () => {
        setLoading(true);
        try {
            await loadFiles();
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
            <div className="w-64 bg-white shadow-lg">
                <div className="p-6 border-b border-gray-200">
                    <div className="flex items-center">
                        <button
                            onClick={() => navigate('/chat')}
                            className="mr-3 p-2 text-gray-400 hover:text-gray-600"
                        >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                            </svg>
                        </button>
                        <h1 className="text-xl font-bold text-gray-900">Knowledge Hub</h1>
                    </div>
                </div>
                
                <div className="p-4">
                    <nav className="space-y-2">
                        <button
                            onClick={() => setActiveTab('documents')}
                            className={`w-full flex items-center px-3 py-2 rounded-lg transition-colors ${
                                activeTab === 'documents' 
                                    ? 'bg-indigo-50 text-indigo-700' 
                                    : 'text-gray-700 hover:bg-gray-100'
                            }`}
                        >
                            <svg className="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            Documents
                        </button>

                        <button
                            onClick={() => setActiveTab('chunking')}
                            className={`w-full flex items-center px-3 py-2 rounded-lg transition-colors ${
                                activeTab === 'chunking' 
                                    ? 'bg-purple-50 text-purple-700' 
                                    : 'text-gray-700 hover:bg-gray-100'
                            }`}
                        >
                            <svg className="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            Chunking Settings
                        </button>
                        
                        {activeTab === 'documents' && (
                            <div className="mt-4 space-y-2 pl-2 border-l border-gray-200">
                                <label className="flex items-center px-3 py-2 text-gray-700 hover:bg-gray-100 rounded-lg cursor-pointer transition-colors">
                                    <svg className="w-4 h-4 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                                    </svg>
                                    Upload Files
                                    <input
                                        type="file"
                                        multiple
                                        onChange={handleFileUpload}
                                        className="hidden"
                                        accept=".pdf,.doc,.docx,.txt,.md,.xlsx,.csv"
                                    />
                                </label>
                                
                                {/* Chunking Method Status */}
                                {selectedMethod && (
                                    <div className="px-3 py-2 text-xs bg-blue-50 text-blue-700 rounded-lg">
                                        <div className="flex items-center">
                                            <svg className="w-3 h-3 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                            </svg>
                                            <span className="font-medium">Chunking Method: {selectedMethod}</span>
                                        </div>
                                        {methodsData[selectedMethod]?.supported_formats && (
                                            <div className="mt-1 text-blue-600">
                                                Supports: {methodsData[selectedMethod].supported_formats.join(', ')}
                                            </div>
                                        )}
                                    </div>
                                )}
                                
                                <button 
                                    onClick={handleShowFolderUpload}
                                    className="flex items-center px-3 py-2 text-gray-700 hover:bg-gray-100 rounded-lg cursor-pointer transition-colors w-full text-left"
                                >
                                    <svg className="w-4 h-4 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                    </svg>
                                    Upload Folder
                                </button>
                            </div>
                        )}
                    </nav>
                    
                    {/* Stats */}
                    <div className="mt-8 p-4 bg-gray-50 rounded-lg">
                        <h3 className="text-sm font-medium text-gray-900 mb-2">Quick Stats</h3>
                        <div className="text-sm text-gray-600 space-y-1">
                            {activeTab === 'documents' && (
                                <>
                                    <div className="flex justify-between">
                                        <span>Files:</span>
                                        <span className="font-medium">{files.length}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Total Size:</span>
                                        <span className="font-medium">
                                            {formatFileSize(files.reduce((total, file) => total + (file.size || 0), 0))}
                                        </span>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                </div>
            </div>

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
                                </h2>
                                <p className="mt-1 text-sm text-gray-500">
                                    {activeTab === 'documents' && 'Manage your uploaded documents for AI conversations'}
                                    {activeTab === 'chunking' && 'Configure how documents are split into chunks for processing'}
                                </p>
                            </div>
                            <div className="flex items-center space-x-4">
                                <button
                                    onClick={refreshData}
                                    className="flex items-center px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
                                >
                                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                    </svg>
                                    Refresh
                                </button>
                                <div className="text-sm text-gray-500">
                                    {activeTab === 'documents' && `${files.length} document${files.length !== 1 ? 's' : ''}`}
                                    {activeTab === 'chunking' && `${(chunkingMethods || []).length} method${(chunkingMethods || []).length !== 1 ? 's' : ''} available`}
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
                                                        <span className="font-medium">📄 PDF:</span> Manual, Naive, Q&A
                                                    </div>
                                                    <div>
                                                        <span className="font-medium">📝 Word:</span> Naive, Resume, Q&A
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

                                            {/* Separators (for naive method) */}
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
                                        {selectedMethod === 'naive' && (
                                            <div>
                                                <p><strong>Naive Chunking:</strong> Simple text splitting based on separators and character count.</p>
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
                                        {!['naive', 'qa', 'resume'].includes(selectedMethod) && (
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
                                Or continue uploading with the 'Naive' chunking method (fallback option).
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
                                Use Naive Method
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
