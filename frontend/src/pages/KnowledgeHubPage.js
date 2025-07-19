import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';

const KnowledgeHubPage = () => {
    const navigate = useNavigate();
    const [files, setFiles] = useState([]);
    const [uploadProgress, setUploadProgress] = useState({});
    const [loading, setLoading] = useState(true);
    const [isReingesting, setIsReingesting] = useState(false);
    const [activeTab, setActiveTab] = useState('documents');
    const [chunkingMethods, setChunkingMethods] = useState([]);
    const [selectedMethod, setSelectedMethod] = useState('naive');
    const [methodConfigs, setMethodConfigs] = useState({});
    const [activeConfig, setActiveConfig] = useState(null);
    const [savingConfig, setSavingConfig] = useState(false);
    const [loadingChunking, setLoadingChunking] = useState(false);
    const [chunkingError, setChunkingError] = useState(null);
    const [methodsData, setMethodsData] = useState({});
    const [warningDialog, setWarningDialog] = useState(null);
    
    const [openDropdown, setOpenDropdown] = useState(null);

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
            
            // Load configurations for all methods
            const configs = {};
            for (const method of methodsList) {
                try {
                    const response = await api.getChunkingConfig(method);
                    // Backend returns {config: {...}}
                    configs[method] = response.config || {};
                } catch (error) {
                    console.error(`Error loading config for ${method}:`, error);
                }
            }
            setMethodConfigs(configs);
            
            // Set active config to default method
            if (methodsList.length > 0) {
                const defaultMethod = methodsList.includes('naive') ? 'naive' : methodsList[0];
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
        try {
            await api.updateChunkingConfig(selectedMethod, activeConfig);
            setMethodConfigs(prev => ({
                ...prev,
                [selectedMethod]: activeConfig
            }));
            // Show success message
            console.log('Chunking configuration saved successfully');
        } catch (error) {
            console.error('Error saving chunking configuration:', error);
        } finally {
            setSavingConfig(false);
        }
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

    const handleFileUpload = async (event) => {
        const uploadedFiles = Array.from(event.target.files);
        
        for (const file of uploadedFiles) {
            const fileId = Date.now() + Math.random();
            const fileName = file.name;
            const fileExtension = fileName.split('.').pop()?.toLowerCase() || '';
            
            // Use selected method or default to 'naive' if none selected
            const currentMethod = selectedMethod || 'naive';
            
            // Validate file against selected chunking method
            const validationResult = validateFileForChunkingMethod(fileName, fileExtension, currentMethod);
            
            if (!validationResult.isValid) {
                // Show warning dialog and ask user to confirm
                setWarningDialog({
                    fileName: fileName,
                    fileExtension: fileExtension,
                    method: currentMethod,
                    message: validationResult.message,
                    supportedFormats: validationResult.supportedFormats,
                    file: file,
                    fileId: fileId,
                    onConfirm: () => continueFileUpload(file, fileId, fileName, false), // false = use naive method
                    onCancel: () => {
                        setWarningDialog(null);
                        // Remove from progress tracking
                        setUploadProgress(prev => {
                            const newProgress = { ...prev };
                            delete newProgress[fileId];
                            return newProgress;
                        });
                    }
                });
                continue; // Skip processing this file until user decides
            }
            
            // File is valid, process immediately
            await continueFileUpload(file, fileId, fileName, true); // true = use selected method
        }
        
        // Reload all data after all uploads complete
        await loadFiles();
    };

    const continueFileUpload = async (file, fileId, fileName, useSelectedMethod) => {
        // Initialize progress tracking
        setUploadProgress(prev => ({
            ...prev,
            [fileId]: { name: fileName, progress: 0 }
        }));
        
        try {
            // Use selected chunking method and config, or fallback to naive
            const chunkingConfig = useSelectedMethod && activeConfig 
                ? {
                    method: selectedMethod || 'naive',
                    ...activeConfig
                }
                : {
                    method: 'naive',
                    chunk_token_num: 1000,
                    chunk_overlap: 200,
                    max_token: 8192
                };
            
            await api.uploadFileWithChunking(
                file,
                chunkingConfig,
                false,
                "",
                (progress) => {
                    setUploadProgress(prev => ({
                        ...prev,
                        [fileId]: { name: fileName, progress }
                    }));
                }
            );

            // Remove from progress tracking
            setUploadProgress(prev => {
                const newProgress = { ...prev };
                delete newProgress[fileId];
                return newProgress;
            });

            // Close warning dialog if it was open for this file
            setWarningDialog(null);

            // Show completion message
            console.log(`File ${fileName} uploaded successfully with chunking method: ${chunkingConfig.method}`);
        } catch (error) {
            console.error('Error uploading file:', error);
            setUploadProgress(prev => {
                const newProgress = { ...prev };
                delete newProgress[fileId];
                return newProgress;
            });
            setWarningDialog(null);
        }
    };

    const handleDeleteFile = async (filename) => {
        if (window.confirm(`Are you sure you want to delete "${filename}"?`)) {
            try {
                await api.deleteFile(filename);
                await loadFiles();
                alert('File deleted successfully!');
            } catch (error) {
                console.error('Error deleting file:', error);
                alert('Error deleting file. Please try again.');
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
            'naive': 'General',
            'qa': 'Q&A',
            'resume': 'Resume',
            'manual': 'Manual',
            'table': 'Table',
            'laws': 'Legal',
            'presentation': 'Presentation',
            'picture': 'Image',
            'one': 'Single Chunk',
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

    const toggleDropdown = (filename) => {
        setOpenDropdown(openDropdown === filename ? null : filename);
    };

    // Close dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = () => {
            setOpenDropdown(null);
        };
        document.addEventListener('click', handleClickOutside);
        return () => document.removeEventListener('click', handleClickOutside);
    }, []);

    return (
        <div className="min-h-screen bg-gray-50 flex">
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
                                
                                <label className="flex items-center px-3 py-2 text-gray-700 hover:bg-gray-100 rounded-lg cursor-pointer transition-colors">
                                    <svg className="w-4 h-4 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                                    </svg>
                                    Upload Folder
                                    <input
                                        type="file"
                                        webkitdirectory="true"
                                        directory="true"
                                        multiple
                                        onChange={handleFileUpload}
                                        className="hidden"
                                    />
                                </label>
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
                                <div className="overflow-x-auto">
                                    <table className="min-w-full divide-y divide-gray-200">
                                        <thead className="bg-gray-50">
                                            <tr>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                    File Name
                                                </th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                    Size
                                                </th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                    Uploaded
                                                </th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                    Status
                                                </th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                    Embedding Model
                                                </th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                    Chunking Method
                                                </th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                    Actions
                                                </th>
                                            </tr>
                                        </thead>
                                        <tbody className="bg-white divide-y divide-gray-200">
                                            {files.map((file) => (
                                                <tr key={file.filename} className="hover:bg-gray-50">
                                                    <td className="px-6 py-4 whitespace-nowrap">
                                                        <div className="flex items-center">
                                                            <div className="flex-shrink-0 h-10 w-10">
                                                                <div className="h-10 w-10 rounded-lg bg-indigo-100 flex items-center justify-center">
                                                                    <svg className="h-6 w-6 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                                                    </svg>
                                                                </div>
                                                            </div>
                                                            <div className="ml-4">
                                                                <div className="text-sm font-medium text-gray-900">
                                                                    {file.filename}
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </td>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                        {formatFileSize(file.size)}
                                                    </td>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                        {formatDate(file.upload_date)}
                                                    </td>
                                                    <td className="px-6 py-4 whitespace-nowrap">
                                                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                                                            file.indexed 
                                                                ? 'bg-green-100 text-green-800' 
                                                                : 'bg-yellow-100 text-yellow-800'
                                                        }`}>
                                                            {file.indexed ? 'Indexed' : 'Pending'}
                                                        </span>
                                                    </td>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                        <span className="inline-flex px-2 py-1 text-xs font-medium rounded-md bg-gray-100 text-gray-700">
                                                            {file.embedding_model || 'Unknown'}
                                                        </span>
                                                    </td>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                        <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-md ${getChunkingMethodStyle(file.chunking_method)}`}>
                                                            {formatChunkingMethod(file.chunking_method)}
                                                        </span>
                                                    </td>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                                                        <div className="relative inline-block text-left">
                                                            <button
                                                                type="button"
                                                                onClick={(e) => {
                                                                    e.stopPropagation();
                                                                    toggleDropdown(file.filename);
                                                                }}
                                                                className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                                                            >
                                                                Actions
                                                                <svg className="ml-2 -mr-0.5 h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                                                    <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                                                                </svg>
                                                            </button>

                                                            {openDropdown === file.filename && (
                                                                <div className="origin-top-right absolute right-0 mt-2 w-48 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 focus:outline-none z-10">
                                                                    <div className="py-1" role="menu">
                                                                        <button
                                                                            onClick={(e) => {
                                                                                e.stopPropagation();
                                                                                navigate(`/documents/${file.id}/chunks`);
                                                                            }}
                                                                            className="text-blue-600 block px-4 py-2 text-sm hover:bg-gray-100 w-full text-left flex items-center"
                                                                            role="menuitem"
                                                                        >
                                                                            📄 View Chunks
                                                                        </button>
                                                                        <button
                                                                            onClick={(e) => {
                                                                                e.stopPropagation();
                                                                                setOpenDropdown(null);
                                                                                handleDeleteFile(file.filename);
                                                                            }}
                                                                            className="text-red-600 block px-4 py-2 text-sm hover:bg-gray-100 w-full text-left"
                                                                            role="menuitem"
                                                                        >
                                                                            Delete
                                                                        </button>
                                                                    </div>
                                                                </div>
                                                            )}
                                                        </div>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )}
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
                                                if (['chunk_size', 'chunk_overlap', 'separators'].includes(key)) return null;
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

                                        {/* Save Button */}
                                        <div className="mt-6 pt-6 border-t border-gray-200">
                                            <button
                                                onClick={saveChunkingConfig}
                                                disabled={savingConfig}
                                                className="w-full bg-purple-600 hover:bg-purple-700 disabled:opacity-50 text-white px-4 py-2 rounded-lg transition-colors"
                                            >
                                                {savingConfig ? 'Saving...' : 'Save Configuration'}
                                            </button>
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
                    <div className="bg-white rounded-lg p-6 m-4 max-w-md w-full">
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
                        
                        <div className="mb-4">
                            <p className="text-sm text-gray-600 mb-2">
                                {warningDialog.message}
                            </p>
                            <div className="bg-gray-50 p-3 rounded-md">
                                <p className="text-sm font-medium text-gray-900">File: {warningDialog.fileName}</p>
                                <p className="text-sm text-gray-600">Extension: .{warningDialog.fileExtension}</p>
                                <p className="text-sm text-gray-600">Selected method: {warningDialog.method}</p>
                                <p className="text-sm text-gray-600">
                                    Supported formats: {warningDialog.supportedFormats.join(', ')}
                                </p>
                            </div>
                            <p className="text-sm text-gray-600 mt-2">
                                Do you want to continue uploading with the 'naive' chunking method instead?
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

        </div>
    );
};

export default KnowledgeHubPage;
