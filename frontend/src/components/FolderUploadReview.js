import React, { useState, useEffect } from 'react';
import { checkFileExists } from '../services/api';

// File type mapping and chunking method recommendations
const fileTypeMap = {
    pdf: { type: 'PDF Document', icon: '📄', defaultMethod: 'general', category: 'document' },
    pptx: { type: 'PowerPoint', icon: '📊', defaultMethod: 'presentation', category: 'presentation' },
    ppt: { type: 'PowerPoint', icon: '📊', defaultMethod: 'presentation', category: 'presentation' },
    docx: { type: 'Word Document', icon: '📝', defaultMethod: 'general', category: 'document' },
    doc: { type: 'Word Document', icon: '📝', defaultMethod: 'general', category: 'document' },
    txt: { type: 'Plain Text', icon: '📄', defaultMethod: 'general', category: 'text' },
    md: { type: 'Markdown', icon: '📝', defaultMethod: 'general', category: 'text' },
    jpg: { type: 'Image', icon: '🖼️', defaultMethod: 'picture', category: 'image' },
    jpeg: { type: 'Image', icon: '🖼️', defaultMethod: 'picture', category: 'image' },
    png: { type: 'Image', icon: '🖼️', defaultMethod: 'picture', category: 'image' },
    gif: { type: 'Image', icon: '🖼️', defaultMethod: 'picture', category: 'image' },
    csv: { type: 'Spreadsheet', icon: '📈', defaultMethod: 'table', category: 'data' },
    xlsx: { type: 'Excel', icon: '📈', defaultMethod: 'table', category: 'data' },
    xls: { type: 'Excel', icon: '📈', defaultMethod: 'table', category: 'data' },
    html: { type: 'HTML', icon: '🌐', defaultMethod: 'general', category: 'web' },
    htm: { type: 'HTML', icon: '🌐', defaultMethod: 'general', category: 'web' },
    json: { type: 'JSON', icon: '📋', defaultMethod: 'qa', category: 'data' },
    eml: { type: 'Email', icon: '📧', defaultMethod: 'email', category: 'email' }
};

const chunkingMethods = [
    { value: 'general', label: 'General', description: 'General document chunking for PDF, DOCX, MD, TXT' },
    { value: 'qa', label: 'Q&A', description: 'Question-answer optimized' },
    { value: 'presentation', label: 'Presentation', description: 'Slide-based chunking' },
    { value: 'resume', label: 'Resume', description: 'Resume/CV optimized' },
    { value: 'picture', label: 'Picture', description: 'OCR-based text extraction' },
    { value: 'table', label: 'Table', description: 'Table-aware chunking' },
    { value: 'email', label: 'Email', description: 'Email structure parsing' }
];

function getFileTypeAndMethod(filename) {
    const ext = filename.split('.').pop()?.toLowerCase() || '';
    const fileInfo = fileTypeMap[ext];
    return fileInfo || { type: 'Unknown', icon: '📄', defaultMethod: 'general', category: 'unknown' };
}

function getFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Utility function to calculate SHA-256 hash of a file
async function calculateFileHash(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = async (event) => {
            try {
                const arrayBuffer = event.target.result;
                const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer);
                const hashArray = Array.from(new Uint8Array(hashBuffer));
                const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
                resolve(hashHex);
            } catch (error) {
                reject(error);
            }
        };
        reader.onerror = () => reject(new Error('Failed to read file for hashing'));
        reader.readAsArrayBuffer(file);
    });
}

const FolderUploadReview = ({ 
    chunkingMethods: availableMethods = [], 
    onUpload, 
    onCancel,
    defaultConfigs = {},
    isVisible = false 
}) => {
    const [files, setFiles] = useState([]);
    const [bulkMethod, setBulkMethod] = useState('');
    const [selectedFiles, setSelectedFiles] = useState(new Set());
    const [showAdvancedConfig, setShowAdvancedConfig] = useState(false);
    const [folderName, setFolderName] = useState('');
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState({});
    const [dragActive, setDragActive] = useState(false);
    const [uploadError, setUploadError] = useState(null);
    const [filter, setFilter] = useState({ category: 'all', method: 'all' });
    const [searchTerm, setSearchTerm] = useState('');
    const [duplicateFiles, setDuplicateFiles] = useState(new Map());
    const [duplicateResolutions, setDuplicateResolutions] = useState(new Map());
    const [isCheckingDuplicates, setIsCheckingDuplicates] = useState(false);

    // Reset state when component becomes visible
    useEffect(() => {
        if (isVisible) {
            setFiles([]);
            setBulkMethod('');
            setSelectedFiles(new Set());
            setShowAdvancedConfig(false);
            setFolderName('');
            setIsUploading(false);
            setUploadProgress({});
            setDragActive(false);
            setUploadError(null);
            setFilter({ category: 'all', method: 'all' });
            setSearchTerm('');
            setDuplicateFiles(new Map());
            setDuplicateResolutions(new Map());
            setIsCheckingDuplicates(false);
        }
    }, [isVisible]);

    // Function to check for duplicates by comparing hashes
    const checkForDuplicates = async (fileList) => {
        setIsCheckingDuplicates(true);
        const duplicates = new Map();
        
        try {
            for (const file of fileList) {
                const hash = await calculateFileHash(file.file);
                
                // Check if a file with this hash already exists in the backend
                try {
                    const existingFile = await checkFileExists(file.name, hash);
                    if (existingFile) {
                        duplicates.set(file.name, {
                            existing: existingFile,
                            new: file,
                            hash: hash,
                            resolution: 'skip' // default to skip
                        });
                    }
                } catch (error) {
                    console.warn(`Could not check duplicate for ${file.name}:`, error);
                }
            }
        } catch (error) {
            console.error('Error checking for duplicates:', error);
        }
        
        setDuplicateFiles(duplicates);
        setIsCheckingDuplicates(false);
        return duplicates;
    };

    // Handle folder upload
    const handleFolderUpload = async (e) => {
        const fileList = Array.from(e.target.files);
        if (fileList.length === 0) return;

        // Extract folder name from first file's path
        const firstFilePath = fileList[0].webkitRelativePath;
        const extractedFolderName = firstFilePath.split('/')[0];
        setFolderName(extractedFolderName);

        const fileData = fileList.map((file, index) => {
            const fileInfo = getFileTypeAndMethod(file.name);
            return {
                id: `file-${index}-${Date.now()}`,
                name: file.name,
                path: file.webkitRelativePath,
                size: file.size,
                type: fileInfo.type,
                icon: fileInfo.icon,
                category: fileInfo.category,
                method: fileInfo.defaultMethod,
                config: { ...defaultConfigs[fileInfo.defaultMethod] } || {},
                file,
                namespace: extractedFolderName // Use folder name as namespace
            };
        }).filter(fileData => fileData.category !== 'unknown'); // Filter out unknown file types

        setFiles(fileData);
        
        // Check for duplicates
        await checkForDuplicates(fileData);
    };

    // Handle drag and drop
    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            // Check if items are folders/directories
            const items = Array.from(e.dataTransfer.items);
            const folderItems = items.filter(item => item.webkitGetAsEntry()?.isDirectory);
            
            if (folderItems.length > 0) {
                // Process dropped folders
                processDroppedFolders(e.dataTransfer.items);
            } else {
                setUploadError('Please drop a folder, not individual files.');
                setTimeout(() => setUploadError(null), 5000);
            }
        }
    };

    const processDroppedFolders = async (items) => {
        const files = [];
        
        for (let item of items) {
            const entry = item.webkitGetAsEntry();
            if (entry && entry.isDirectory) {
                const folderFiles = await readEntriesPromise(entry);
                files.push(...folderFiles);
                break; // Only process first folder
            }
        }
        
        if (files.length > 0) {
            // Convert to format expected by handleFolderUpload
            const mockEvent = {
                target: {
                    files: files
                }
            };
            await handleFolderUpload(mockEvent);
        }
    };

    const readEntriesPromise = (directoryEntry) => {
        return new Promise((resolve) => {
            const files = [];
            const reader = directoryEntry.createReader();
            
            const readEntries = () => {
                reader.readEntries((entries) => {
                    if (entries.length === 0) {
                        resolve(files);
                    } else {
                        entries.forEach(entry => {
                            if (entry.isFile) {
                                entry.file(file => {
                                    // Add webkitRelativePath property
                                    const relativePath = entry.fullPath.substring(1); // Remove leading slash
                                    Object.defineProperty(file, 'webkitRelativePath', {
                                        value: relativePath,
                                        writable: false
                                    });
                                    files.push(file);
                                });
                            }
                        });
                        readEntries(); // Continue reading
                    }
                });
            };
            
            readEntries();
        });
    };

    // Handle per-file method change
    const handleMethodChange = (fileId, method) => {
        setFiles(files => files.map(f => 
            f.id === fileId 
                ? { 
                    ...f, 
                    method, 
                    config: { ...defaultConfigs[method] } || {} 
                }
                : f
        ));
    };

    // Handle per-file config change
    const handleConfigChange = (fileId, configKey, value) => {
        setFiles(files => files.map(f => 
            f.id === fileId 
                ? { 
                    ...f, 
                    config: { ...f.config, [configKey]: value }
                }
                : f
        ));
    };

    // Bulk apply method
    const handleBulkApply = () => {
        if (!bulkMethod) return;
        
        const targetFiles = selectedFiles.size > 0 
            ? files.filter(f => selectedFiles.has(f.id))
            : files;

        setFiles(files => files.map(f => 
            targetFiles.find(tf => tf.id === f.id)
                ? { 
                    ...f, 
                    method: bulkMethod, 
                    config: { ...defaultConfigs[bulkMethod] } || {} 
                }
                : f
        ));
        
        setBulkMethod('');
        setSelectedFiles(new Set());
    };

    // Handle file selection for bulk operations
    const handleFileSelection = (fileId, isSelected) => {
        setSelectedFiles(prev => {
            const newSet = new Set(prev);
            if (isSelected) {
                newSet.add(fileId);
            } else {
                newSet.delete(fileId);
            }
            return newSet;
        });
    };

    // Select all files
    const handleSelectAll = () => {
        if (selectedFiles.size === files.length) {
            setSelectedFiles(new Set());
        } else {
            setSelectedFiles(new Set(files.map(f => f.id)));
        }
    };

    // Process uploads
    const handleUploadConfirm = async () => {
        if (files.length === 0 || isUploading) return;

        setIsUploading(true);
        
        try {
            // Filter files based on duplicate resolution
            const filesToUpload = files.filter(file => {
                const duplicateInfo = duplicateFiles.get(file.name);
                if (!duplicateInfo) {
                    return true; // Not a duplicate, include it
                }
                
                const resolution = duplicateResolutions.get(file.name) || 'skip';
                return resolution === 'overwrite'; // Only include if we're overwriting
            });

            // Add hash information to files for backend processing
            const filesWithHashes = await Promise.all(filesToUpload.map(async (file) => {
                const duplicateInfo = duplicateFiles.get(file.name);
                const hash = duplicateInfo ? duplicateInfo.hash : await calculateFileHash(file.file);
                
                return {
                    ...file,
                    hash: hash,
                    isOverwrite: duplicateFiles.has(file.name) && duplicateResolutions.get(file.name) === 'overwrite'
                };
            }));

            await onUpload(filesWithHashes, {
                folderName,
                duplicateResolutions: Object.fromEntries(duplicateResolutions),
                onProgress: (fileId, progress) => {
                    setUploadProgress(prev => ({
                        ...prev,
                        [fileId]: progress
                    }));
                }
            });
            
            // Clear progress after successful upload
            setUploadProgress({});
            
        } catch (error) {
            console.error('Upload failed:', error);
            // Clear progress on error as well
            setUploadProgress({});
        } finally {
            setIsUploading(false);
        }
    };

    // Filter and search files
    const filteredFiles = files.filter(file => {
        // Hide unknown file types
        if (file.category === 'unknown') {
            return false;
        }
        
        const category = file.category;
        
        // Category filter
        if (filter.category !== 'all' && category !== filter.category) {
            return false;
        }
        
        // Method filter
        if (filter.method !== 'all' && file.method !== filter.method) {
            return false;
        }
        
        // Search filter
        if (searchTerm && !file.name.toLowerCase().includes(searchTerm.toLowerCase())) {
            return false;
        }
        
        return true;
    });

    // Get file statistics
    const stats = {
        totalFiles: filteredFiles.length,
        totalSize: filteredFiles.reduce((sum, f) => sum + f.size, 0),
        categories: filteredFiles.reduce((acc, f) => {
            acc[f.category] = (acc[f.category] || 0) + 1;
            return acc;
        }, {}),
        methods: filteredFiles.reduce((acc, f) => {
            acc[f.method] = (acc[f.method] || 0) + 1;
            return acc;
        }, {})
    };

    if (!isVisible) return null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] overflow-hidden flex flex-col">
                {/* Header */}
                <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
                    <div className="flex items-center justify-between">
                        <div>
                            <h2 className="text-xl font-semibold text-gray-900">
                                🗂️ Folder Upload & Chunking Review
                            </h2>
                            {folderName && (
                                <p className="text-sm text-gray-600 mt-1">
                                    Folder: <span className="font-medium">{folderName}</span>
                                </p>
                            )}
                        </div>
                        <button
                            onClick={onCancel}
                            className="text-gray-400 hover:text-gray-600 p-2"
                            disabled={isUploading}
                        >
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>
                </div>

                {/* Folder Upload Section */}
                {files.length === 0 && (
                    <div 
                        className={`flex-1 flex flex-col items-center justify-center p-8 transition-colors ${
                            dragActive ? 'bg-blue-50 border-blue-300' : ''
                        }`}
                        onDragEnter={handleDrag}
                        onDragLeave={handleDrag}
                        onDragOver={handleDrag}
                        onDrop={handleDrop}
                    >
                        <div className="text-center">
                            <svg className={`w-16 h-16 mx-auto mb-4 transition-colors ${
                                dragActive ? 'text-blue-500' : 'text-gray-400'
                            }`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                            </svg>
                            <h3 className="text-lg font-medium text-gray-900 mb-2">Upload a Folder</h3>
                            <p className="text-gray-600 mb-6 max-w-md">
                                {dragActive 
                                    ? 'Drop your folder here!' 
                                    : 'Select a folder to upload or drag and drop. Files will be automatically analyzed and assigned optimal chunking methods.'
                                }
                            </p>
                            
                            {uploadError && (
                                <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                                    {uploadError}
                                </div>
                            )}
                            
                            <label className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer transition-colors">
                                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                                </svg>
                                Choose Folder
                                <input
                                    type="file"
                                    webkitdirectory="true"
                                    multiple
                                    onChange={handleFolderUpload}
                                    className="hidden"
                                />
                            </label>
                        </div>
                    </div>
                )}

                {/* File Review Section */}
                {files.length > 0 && (
                    <div className="flex-1 flex flex-col overflow-hidden">
                        {/* Stats Section */}
                        <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                <div className="text-center">
                                    <div className="text-2xl font-bold text-blue-600">{stats.totalFiles}</div>
                                    <div className="text-sm text-gray-600">Files</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-2xl font-bold text-green-600">{getFileSize(stats.totalSize)}</div>
                                    <div className="text-sm text-gray-600">Total Size</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-2xl font-bold text-purple-600">{Object.keys(stats.categories).length}</div>
                                    <div className="text-sm text-gray-600">File Types</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-2xl font-bold text-orange-600">{Object.keys(stats.methods).length}</div>
                                    <div className="text-sm text-gray-600">Chunk Methods</div>
                                </div>
                            </div>
                        </div>

                        {/* Bulk Operations */}
                        <div className="px-6 py-3 bg-white border-b border-gray-200">
                            <div className="flex flex-wrap items-center gap-4">
                                <button
                                    onClick={handleSelectAll}
                                    className="text-sm text-blue-600 hover:text-blue-800"
                                >
                                    {selectedFiles.size === files.length ? 'Deselect All' : 'Select All'}
                                </button>
                                
                                <div className="flex items-center gap-2">
                                    <select 
                                        value={bulkMethod} 
                                        onChange={e => setBulkMethod(e.target.value)}
                                        className="text-sm border border-gray-300 rounded px-2 py-1"
                                    >
                                        <option value="">Bulk Apply Method</option>
                                        {availableMethods.map(method => (
                                            <option key={method} value={method}>
                                                {method.charAt(0).toUpperCase() + method.slice(1)}
                                            </option>
                                        ))}
                                    </select>
                                    <button
                                        onClick={handleBulkApply}
                                        disabled={!bulkMethod}
                                        className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Apply{selectedFiles.size > 0 ? ` to ${selectedFiles.size}` : ' to All'}
                                    </button>
                                </div>

                                {duplicateFiles.size > 0 && (
                                    <div className="flex items-center gap-2">
                                        <span className="text-sm text-gray-600">Duplicates:</span>
                                        <button
                                            onClick={() => {
                                                const newResolutions = new Map(duplicateResolutions);
                                                duplicateFiles.forEach((_, filename) => {
                                                    newResolutions.set(filename, 'skip');
                                                });
                                                setDuplicateResolutions(newResolutions);
                                            }}
                                            className="px-3 py-1 text-sm bg-yellow-100 text-yellow-700 rounded hover:bg-yellow-200"
                                        >
                                            Skip All
                                        </button>
                                        <button
                                            onClick={() => {
                                                const newResolutions = new Map(duplicateResolutions);
                                                duplicateFiles.forEach((_, filename) => {
                                                    newResolutions.set(filename, 'overwrite');
                                                });
                                                setDuplicateResolutions(newResolutions);
                                            }}
                                            className="px-3 py-1 text-sm bg-orange-100 text-orange-700 rounded hover:bg-orange-200"
                                        >
                                            Overwrite All
                                        </button>
                                    </div>
                                )}

                                <button
                                    onClick={() => setShowAdvancedConfig(!showAdvancedConfig)}
                                    className="text-sm text-gray-600 hover:text-gray-800 flex items-center"
                                >
                                    <svg className={`w-4 h-4 mr-1 transition-transform ${showAdvancedConfig ? 'rotate-90' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                    </svg>
                                    Advanced Config
                                </button>
                            </div>
                        </div>

                        {/* Filter and Search Controls */}
                        <div className="px-6 py-3 bg-gray-50 border-b border-gray-200">
                            <div className="flex flex-wrap items-center gap-4">
                                {/* Search */}
                                <div className="flex items-center gap-2">
                                    <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                                    </svg>
                                    <input
                                        type="text"
                                        placeholder="Search files..."
                                        value={searchTerm}
                                        onChange={(e) => setSearchTerm(e.target.value)}
                                        className="text-sm border border-gray-300 rounded px-3 py-1 w-48"
                                    />
                                </div>

                                {/* Category Filter */}
                                <div className="flex items-center gap-2">
                                    <label className="text-sm text-gray-600">Category:</label>
                                    <select
                                        value={filter.category}
                                        onChange={(e) => setFilter(prev => ({ ...prev, category: e.target.value }))}
                                        className="text-sm border border-gray-300 rounded px-2 py-1"
                                    >
                                        <option value="all">All Categories</option>
                                        <option value="document">📄 Document</option>
                                        <option value="presentation">📊 Presentation</option>
                                        <option value="image">🖼️ Image</option>
                                        <option value="data">📈 Data</option>
                                        <option value="text">📝 Text</option>
                                        <option value="web">🌐 Web</option>
                                        <option value="email">📧 Email</option>
                                    </select>
                                </div>

                                {/* Method Filter */}
                                <div className="flex items-center gap-2">
                                    <label className="text-sm text-gray-600">Method:</label>
                                    <select
                                        value={filter.method}
                                        onChange={(e) => setFilter(prev => ({ ...prev, method: e.target.value }))}
                                        className="text-sm border border-gray-300 rounded px-2 py-1"
                                    >
                                        <option value="all">All Methods</option>
                                        {availableMethods.map(method => (
                                            <option key={method} value={method}>
                                                {method.charAt(0).toUpperCase() + method.slice(1)}
                                            </option>
                                        ))}
                                    </select>
                                </div>

                                {/* Clear Filters */}
                                {(filter.category !== 'all' || filter.method !== 'all' || searchTerm) && (
                                    <button
                                        onClick={() => {
                                            setFilter({ category: 'all', method: 'all' });
                                            setSearchTerm('');
                                        }}
                                        className="text-sm text-gray-500 hover:text-gray-700 underline"
                                    >
                                        Clear Filters
                                    </button>
                                )}

                                {/* Results Count */}
                                <div className="text-sm text-gray-500 ml-auto">
                                    Showing {filteredFiles.length} of {files.length} files
                                </div>
                            </div>
                        </div>

                        {/* Files Table */}
                        <div className="flex-1 overflow-auto">
                            {filteredFiles.length === 0 ? (
                                <div className="flex flex-col items-center justify-center py-12">
                                    <svg className="w-12 h-12 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                                    </svg>
                                    <h3 className="text-lg font-medium text-gray-900 mb-2">No files match your filters</h3>
                                    <p className="text-gray-500 text-sm mb-4">Try adjusting your search or filter criteria</p>
                                    <button
                                        onClick={() => {
                                            setFilter({ category: 'all', method: 'all' });
                                            setSearchTerm('');
                                        }}
                                        className="text-sm text-blue-600 hover:text-blue-800 underline"
                                    >
                                        Clear all filters
                                    </button>
                                </div>
                            ) : (
                                <table className="min-w-full divide-y divide-gray-200">
                                <thead className="bg-gray-50 sticky top-0">
                                    <tr>
                                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-12">
                                            <input
                                                type="checkbox"
                                                checked={selectedFiles.size === files.length && files.length > 0}
                                                onChange={handleSelectAll}
                                                className="rounded border-gray-300"
                                            />
                                        </th>
                                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                            File
                                        </th>
                                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                            Type
                                        </th>
                                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                            Size
                                        </th>
                                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                            Suggested Method
                                        </th>
                                        <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                            Override Method
                                        </th>
                                        {showAdvancedConfig && (
                                            <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                Config
                                            </th>
                                        )}
                                    </tr>
                                </thead>
                                <tbody className="bg-white divide-y divide-gray-200">
                                    {filteredFiles.map((file) => (
                                        <tr key={file.id} className={`hover:bg-gray-50 ${selectedFiles.has(file.id) ? 'bg-blue-50' : ''}`}>
                                            <td className="px-3 py-4">
                                                <input
                                                    type="checkbox"
                                                    checked={selectedFiles.has(file.id)}
                                                    onChange={(e) => handleFileSelection(file.id, e.target.checked)}
                                                    className="rounded border-gray-300"
                                                />
                                            </td>
                                            <td className="px-3 py-4">
                                                <div className="flex items-center">
                                                    <span className="text-lg mr-2">{file.icon}</span>
                                                    <div className="flex-1">
                                                        <div className="flex items-center gap-2">
                                                            <div className="text-sm font-medium text-gray-900 truncate max-w-xs" title={file.name}>
                                                                {file.name}
                                                            </div>
                                                            {duplicateFiles.has(file.name) && (
                                                                <span className="inline-flex px-2 py-1 text-xs font-medium rounded-full bg-yellow-100 text-yellow-700">
                                                                    Duplicate
                                                                </span>
                                                            )}
                                                        </div>
                                                        <div className="text-xs text-gray-500 truncate max-w-xs" title={file.path}>
                                                            {file.path}
                                                        </div>
                                                        {duplicateFiles.has(file.name) && (
                                                            <div className="mt-1">
                                                                <select
                                                                    value={duplicateResolutions.get(file.name) || 'skip'}
                                                                    onChange={(e) => setDuplicateResolutions(prev => 
                                                                        new Map(prev.set(file.name, e.target.value))
                                                                    )}
                                                                    className="text-xs border border-yellow-300 rounded px-1 py-0.5 bg-yellow-50"
                                                                >
                                                                    <option value="skip">Skip (Keep existing)</option>
                                                                    <option value="overwrite">Overwrite</option>
                                                                </select>
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            </td>
                                            <td className="px-3 py-4 text-sm text-gray-900">
                                                <span className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                                                    file.category === 'document' ? 'bg-blue-100 text-blue-700' :
                                                    file.category === 'image' ? 'bg-green-100 text-green-700' :
                                                    file.category === 'data' ? 'bg-purple-100 text-purple-700' :
                                                    file.category === 'presentation' ? 'bg-orange-100 text-orange-700' :
                                                    'bg-gray-100 text-gray-700'
                                                }`}>
                                                    {file.type}
                                                </span>
                                            </td>
                                            <td className="px-3 py-4 text-sm text-gray-500">
                                                {getFileSize(file.size)}
                                            </td>
                                            <td className="px-3 py-4 text-sm text-gray-900">
                                                <span className="inline-flex px-2 py-1 text-xs font-medium rounded bg-green-100 text-green-700">
                                                    {getFileTypeAndMethod(file.name).defaultMethod.charAt(0).toUpperCase() + getFileTypeAndMethod(file.name).defaultMethod.slice(1)}
                                                </span>
                                            </td>
                                            <td className="px-3 py-4">
                                                <select
                                                    value={file.method}
                                                    onChange={(e) => handleMethodChange(file.id, e.target.value)}
                                                    className="text-sm border border-gray-300 rounded px-2 py-1 w-full"
                                                >
                                                    {availableMethods.map(method => (
                                                        <option key={method} value={method}>
                                                            {method.charAt(0).toUpperCase() + method.slice(1)}
                                                        </option>
                                                    ))}
                                                </select>
                                            </td>
                                            {showAdvancedConfig && (
                                                <td className="px-3 py-4">
                                                    <div className="space-y-2">
                                                        {Object.entries(file.config).map(([key, value]) => (
                                                            <div key={key} className="flex items-center space-x-2">
                                                                <label className="text-xs text-gray-600 w-20 truncate" title={key}>
                                                                    {key.replace(/_/g, ' ')}:
                                                                </label>
                                                                <input
                                                                    type={typeof value === 'number' ? 'number' : typeof value === 'boolean' ? 'checkbox' : 'text'}
                                                                    value={typeof value === 'boolean' ? undefined : value}
                                                                    checked={typeof value === 'boolean' ? value : undefined}
                                                                    onChange={(e) => handleConfigChange(
                                                                        file.id, 
                                                                        key, 
                                                                        typeof value === 'boolean' ? e.target.checked : 
                                                                        typeof value === 'number' ? parseFloat(e.target.value) : 
                                                                        e.target.value
                                                                    )}
                                                                    className="text-xs border border-gray-300 rounded px-1 py-0.5 w-16"
                                                                />
                                                            </div>
                                                        ))}
                                                    </div>
                                                </td>
                                            )}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                            )}
                        </div>

                        {/* Upload Progress */}
                        {isUploading && Object.keys(uploadProgress).length > 0 && (
                            <div className="px-6 py-4 bg-gray-50 border-t border-gray-200">
                                <div className="space-y-2">
                                    {files.map((file) => (
                                        <div key={file.id} className="flex items-center space-x-3">
                                            <span className="text-sm text-gray-600 flex-1 truncate">
                                                {file.name}
                                            </span>
                                            <div className="w-32 bg-gray-200 rounded-full h-2">
                                                <div
                                                    className="bg-blue-600 h-2 rounded-full transition-all"
                                                    style={{ width: `${uploadProgress[file.id] || 0}%` }}
                                                />
                                            </div>
                                            <span className="text-sm text-gray-500 w-12">
                                                {uploadProgress[file.id] || 0}%
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Footer Actions */}
                        <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
                            <div className="flex items-center justify-between">
                                <div className="text-sm text-gray-600">
                                    {files.length} files ready • {getFileSize(files.reduce((sum, f) => sum + f.size, 0))} total
                                    {filteredFiles.length < files.length && (
                                        <span className="ml-2 text-blue-600">
                                            ({filteredFiles.length} visible)
                                        </span>
                                    )}
                                    {duplicateFiles.size > 0 && (
                                        <span className="ml-2 text-yellow-600">
                                            • {duplicateFiles.size} duplicate{duplicateFiles.size !== 1 ? 's' : ''} found
                                        </span>
                                    )}
                                    {isCheckingDuplicates && (
                                        <span className="ml-2 text-blue-600">
                                            • Checking for duplicates...
                                        </span>
                                    )}
                                </div>
                                <div className="flex space-x-3">
                                    <button
                                        onClick={onCancel}
                                        disabled={isUploading}
                                        className="px-4 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        onClick={handleUploadConfirm}
                                        disabled={isUploading || files.length === 0}
                                        className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {isUploading ? 'Uploading...' : (() => {
                                            const skippedCount = Array.from(duplicateFiles.keys()).filter(filename => 
                                                (duplicateResolutions.get(filename) || 'skip') === 'skip'
                                            ).length;
                                            const uploadCount = files.length - skippedCount;
                                            
                                            if (skippedCount > 0) {
                                                return `Upload ${uploadCount} Files (${skippedCount} skipped)`;
                                            }
                                            return `Upload ${files.length} Files`;
                                        })()}
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default FolderUploadReview;
