import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';

const KnowledgeHubPage = () => {
    const navigate = useNavigate();
    const [files, setFiles] = useState([]);
    const [uploadProgress, setUploadProgress] = useState({});
    const [loading, setLoading] = useState(true);
    const [vectorStoreStats, setVectorStoreStats] = useState({});
    const [isReingesting, setIsReingesting] = useState(false);
    const [activeTab, setActiveTab] = useState('documents');

    useEffect(() => {
        loadFiles();
        loadVectorStoreStats();
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

    const loadVectorStoreStats = async () => {
        try {
            const response = await api.getVectorStoreStats();
            setVectorStoreStats(response.stats || {});
        } catch (error) {
            console.error('Error loading vector store stats:', error);
        }
    };

    const handleFileUpload = async (event) => {
        const uploadedFiles = Array.from(event.target.files);
        
        for (const file of uploadedFiles) {
            const fileId = Date.now() + Math.random();
            setUploadProgress(prev => ({
                ...prev,
                [fileId]: { name: file.name, progress: 0 }
            }));

            try {
                await api.uploadFileWithProgress(file, false, "", (progress) => {
                    setUploadProgress(prev => ({
                        ...prev,
                        [fileId]: { name: file.name, progress }
                    }));
                });

                // Remove from progress tracking
                setUploadProgress(prev => {
                    const newProgress = { ...prev };
                    delete newProgress[fileId];
                    return newProgress;
                });

                // Show completion message
                console.log(`File ${file.name} uploaded successfully`);
            } catch (error) {
                console.error('Error uploading file:', error);
                setUploadProgress(prev => {
                    const newProgress = { ...prev };
                    delete newProgress[fileId];
                    return newProgress;
                });
            }
        }

        // Reload all data after all uploads complete
        await Promise.all([
            loadFiles(),
            loadVectorStoreStats()
        ]);
        
        // Show success notification
        if (uploadedFiles.length > 0) {
            alert(`${uploadedFiles.length} file(s) uploaded and processed successfully!`);
        }
    };

    const handleDeleteFile = async (filename) => {
        if (window.confirm(`Are you sure you want to delete "${filename}"?`)) {
            try {
                await api.deleteFile(filename);
                await Promise.all([
                    loadFiles(),
                    loadVectorStoreStats()
                ]);
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
                await Promise.all([
                    loadFiles(),
                    loadVectorStoreStats()
                ]);
            }, 2000);
            
            alert('Documents re-ingested successfully! Data will refresh shortly.');
        } catch (error) {
            console.error('Error re-ingesting documents:', error);
            alert('Error re-ingesting documents. Please try again.');
        } finally {
            setIsReingesting(false);
        }
    };

    const handleClearVectorStore = async () => {
        if (!window.confirm('Are you sure you want to clear the entire vector store? This action cannot be undone.')) {
            return;
        }

        try {
            await api.clearVectorStore();
            await Promise.all([
                loadFiles(),
                loadVectorStoreStats()
            ]);
            alert('Vector store cleared successfully!');
        } catch (error) {
            console.error('Error clearing vector store:', error);
            alert('Error clearing vector store. Please try again.');
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

    const refreshData = async () => {
        setLoading(true);
        try {
            await Promise.all([
                loadFiles(),
                loadVectorStoreStats()
            ]);
        } finally {
            setLoading(false);
        }
    };

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
                            onClick={() => setActiveTab('vectorstore')}
                            className={`w-full flex items-center px-3 py-2 rounded-lg transition-colors ${
                                activeTab === 'vectorstore' 
                                    ? 'bg-purple-50 text-purple-700' 
                                    : 'text-gray-700 hover:bg-gray-100'
                            }`}
                        >
                            <svg className="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                            </svg>
                            Vector Store
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
                            {activeTab === 'vectorstore' && (
                                <>
                                    <div className="flex justify-between">
                                        <span>Vectors:</span>
                                        <span className="font-medium">{vectorStoreStats.total_documents || 0}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Collections:</span>
                                        <span className="font-medium">{vectorStoreStats.total_collections || 0}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Size:</span>
                                        <span className="font-medium">{formatFileSize(vectorStoreStats.size_bytes || 0)}</span>
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
                                    {activeTab === 'vectorstore' && 'Vector Store Management'}
                                </h2>
                                <p className="mt-1 text-sm text-gray-500">
                                    {activeTab === 'documents' && 'Manage your uploaded documents for AI conversations'}
                                    {activeTab === 'vectorstore' && 'Monitor and manage your vector database'}
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
                                    {activeTab === 'vectorstore' && `${vectorStoreStats.total_documents || 0} vectors`}
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
                                <div className="overflow-hidden">
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
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium space-x-2">
                                                        <button
                                                            onClick={() => handleDeleteFile(file.filename)}
                                                            className="text-red-600 hover:text-red-900 transition-colors"
                                                        >
                                                            Delete
                                                        </button>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Vector Store Tab */}
                    {activeTab === 'vectorstore' && (
                        <div className="space-y-6">
                            {/* Vector Store Stats Cards */}
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                <div className="bg-white rounded-lg shadow p-6">
                                    <div className="flex items-center">
                                        <div className="flex-shrink-0">
                                            <div className="h-12 w-12 rounded-lg bg-blue-100 flex items-center justify-center">
                                                <svg className="h-6 w-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                                                </svg>
                                            </div>
                                        </div>
                                        <div className="ml-4">
                                            <h3 className="text-lg font-medium text-gray-900">Total Vectors</h3>
                                            <p className="text-3xl font-bold text-blue-600">{vectorStoreStats.total_documents || 0}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-white rounded-lg shadow p-6">
                                    <div className="flex items-center">
                                        <div className="flex-shrink-0">
                                            <div className="h-12 w-12 rounded-lg bg-purple-100 flex items-center justify-center">
                                                <svg className="h-6 w-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                                                </svg>
                                            </div>
                                        </div>
                                        <div className="ml-4">
                                            <h3 className="text-lg font-medium text-gray-900">Collections</h3>
                                            <p className="text-3xl font-bold text-purple-600">{vectorStoreStats.total_collections || 0}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-white rounded-lg shadow p-6">
                                    <div className="flex items-center">
                                        <div className="flex-shrink-0">
                                            <div className="h-12 w-12 rounded-lg bg-green-100 flex items-center justify-center">
                                                <svg className="h-6 w-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                                                </svg>
                                            </div>
                                        </div>
                                        <div className="ml-4">
                                            <h3 className="text-lg font-medium text-gray-900">Storage Size</h3>
                                            <p className="text-3xl font-bold text-green-600">{formatFileSize(vectorStoreStats.size_bytes || 0)}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Vector Store Actions */}
                            <div className="bg-white rounded-lg shadow">
                                <div className="px-6 py-4 border-b border-gray-200">
                                    <h3 className="text-lg font-medium text-gray-900">Vector Store Actions</h3>
                                    <p className="mt-1 text-sm text-gray-500">Manage your vector database and re-process documents</p>
                                </div>
                                <div className="p-6 space-y-4">
                                    <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg">
                                        <div>
                                            <h4 className="text-sm font-medium text-gray-900">Re-ingest All Documents</h4>
                                            <p className="text-sm text-gray-500">Recreate vectors for all uploaded documents</p>
                                        </div>
                                        <button
                                            onClick={handleReingestion}
                                            disabled={isReingesting}
                                            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors disabled:opacity-50"
                                        >
                                            {isReingesting ? 'Re-ingesting...' : 'Re-ingest'}
                                        </button>
                                    </div>

                                    <div className="flex items-center justify-between p-4 bg-red-50 rounded-lg">
                                        <div>
                                            <h4 className="text-sm font-medium text-gray-900">Clear Vector Store</h4>
                                            <p className="text-sm text-gray-500">Remove all vectors from the database</p>
                                        </div>
                                        <button
                                            onClick={handleClearVectorStore}
                                            className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors"
                                        >
                                            Clear Store
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                </div>
            </div>
        </div>
    );
};

export default KnowledgeHubPage;
