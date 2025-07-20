import React, { useState, useEffect } from 'react';
import { checkFileExists } from '../services/api';

// File type mapping and chunking method recommendations
const fileTypeMap = {
    pdf: { type: 'PDF Document', icon: '📄', defaultMethod: 'qa', category: 'document' },
    pptx: { type: 'PowerPoint', icon: '📊', defaultMethod: 'presentation', category: 'presentation' },
    ppt: { type: 'PowerPoint', icon: '📊', defaultMethod: 'presentation', category: 'presentation' },
    docx: { type: 'Word Document', icon: '📝', defaultMethod: 'resume', category: 'document' },
    doc: { type: 'Word Document', icon: '📝', defaultMethod: 'resume', category: 'document' },
    txt: { type: 'Plain Text', icon: '📄', defaultMethod: 'naive', category: 'text' },
    md: { type: 'Markdown', icon: '📝', defaultMethod: 'naive', category: 'text' },
    jpg: { type: 'Image', icon: '🖼️', defaultMethod: 'picture', category: 'image' },
    jpeg: { type: 'Image', icon: '🖼️', defaultMethod: 'picture', category: 'image' },
    png: { type: 'Image', icon: '🖼️', defaultMethod: 'picture', category: 'image' },
    gif: { type: 'Image', icon: '🖼️', defaultMethod: 'picture', category: 'image' },
    csv: { type: 'Spreadsheet', icon: '📈', defaultMethod: 'table', category: 'data' },
    xlsx: { type: 'Excel', icon: '📈', defaultMethod: 'table', category: 'data' },
    xls: { type: 'Excel', icon: '📈', defaultMethod: 'table', category: 'data' },
    html: { type: 'HTML', icon: '🌐', defaultMethod: 'naive', category: 'web' },
    htm: { type: 'HTML', icon: '🌐', defaultMethod: 'naive', category: 'web' },
    json: { type: 'JSON', icon: '📋', defaultMethod: 'naive', category: 'data' },
    eml: { type: 'Email', icon: '📧', defaultMethod: 'email', category: 'email' }
};

function getFileTypeAndMethod(filename) {
    const ext = filename.split('.').pop()?.toLowerCase() || '';
    const fileInfo = fileTypeMap[ext];
    return fileInfo || { type: 'Unknown', icon: '📄', defaultMethod: 'naive', category: 'unknown' };
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

const FileUploadReview = ({ 
    chunkingMethods: availableMethods = [], 
    onUpload, 
    onCancel,
    defaultConfigs = {},
    isVisible = false,
    selectedFiles = []
}) => {
    const [files, setFiles] = useState([]);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState({});
    const [duplicateFiles, setDuplicateFiles] = useState(new Map());
    const [duplicateResolutions, setDuplicateResolutions] = useState(new Map());
    const [isCheckingDuplicates, setIsCheckingDuplicates] = useState(false);

    // Process selected files when component becomes visible
    useEffect(() => {
        if (isVisible && selectedFiles.length > 0) {
            // Reset all state before processing new files
            setFiles([]);
            setIsUploading(false);
            setUploadProgress({});
            setDuplicateFiles(new Map());
            setDuplicateResolutions(new Map());
            setIsCheckingDuplicates(false);
            
            processFiles(selectedFiles);
        } else if (!isVisible) {
            // Reset state when component is hidden
            setFiles([]);
            setIsUploading(false);
            setUploadProgress({});
            setDuplicateFiles(new Map());
            setDuplicateResolutions(new Map());
            setIsCheckingDuplicates(false);
        }
    }, [isVisible, selectedFiles]);

    // Process files and check for duplicates
    const processFiles = async (fileList) => {
        const fileData = fileList.map((file, index) => {
            const fileInfo = getFileTypeAndMethod(file.name);
            return {
                id: `file-${index}-${Date.now()}`,
                name: file.name,
                size: file.size,
                type: fileInfo.type,
                icon: fileInfo.icon,
                category: fileInfo.category,
                method: fileInfo.defaultMethod,
                config: { ...defaultConfigs[fileInfo.defaultMethod] } || {},
                file
            };
        }).filter(fileData => fileData.category !== 'unknown'); // Filter out unknown file types

        setFiles(fileData);
        
        // Check for duplicates
        await checkForDuplicates(fileData);
    };

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

    if (!isVisible || files.length === 0) return null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
                {/* Header */}
                <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
                    <div className="flex items-center justify-between">
                        <div>
                            <h2 className="text-xl font-semibold text-gray-900">
                                📄 File Upload Review
                            </h2>
                            <p className="text-sm text-gray-600 mt-1">
                                Review and configure {files.length} file{files.length !== 1 ? 's' : ''} before uploading
                            </p>
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

                {/* File Review Section */}
                <div className="flex-1 flex flex-col overflow-hidden">
                    {/* Duplicate Actions */}
                    {duplicateFiles.size > 0 && (
                        <div className="px-6 py-3 bg-yellow-50 border-b border-yellow-200">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <svg className="w-5 h-5 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16c-.77.833.192 2.5 1.732 2.5z" />
                                    </svg>
                                    <span className="text-sm font-medium text-yellow-800">
                                        {duplicateFiles.size} duplicate{duplicateFiles.size !== 1 ? 's' : ''} found
                                    </span>
                                </div>
                                <div className="flex items-center gap-2">
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
                            </div>
                        </div>
                    )}

                    {/* Files Table */}
                    <div className="flex-1 overflow-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50 sticky top-0">
                                <tr>
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
                                        Chunking Method
                                    </th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {files.map((file) => (
                                    <tr key={file.id} className="hover:bg-gray-50">
                                        <td className="px-3 py-4">
                                            <div className="flex items-center">
                                                <span className="text-lg mr-2">{file.icon}</span>
                                                <div className="flex-1">
                                                    <div className="flex items-center gap-2">
                                                        <div className="text-sm font-medium text-gray-900" title={file.name}>
                                                            {file.name}
                                                        </div>
                                                        {duplicateFiles.has(file.name) && (
                                                            <span className="inline-flex px-2 py-1 text-xs font-medium rounded-full bg-yellow-100 text-yellow-700">
                                                                Duplicate
                                                            </span>
                                                        )}
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
                                    </tr>
                                ))}
                            </tbody>
                        </table>
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
                                {files.length} file{files.length !== 1 ? 's' : ''} ready • {getFileSize(files.reduce((sum, f) => sum + f.size, 0))} total
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
                                            return `Upload ${uploadCount} File${uploadCount !== 1 ? 's' : ''} (${skippedCount} skipped)`;
                                        }
                                        return `Upload ${files.length} File${files.length !== 1 ? 's' : ''}`;
                                    })()}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default FileUploadReview;
