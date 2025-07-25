import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

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

// Chunking methods available for reingestion
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

const DocumentReingestionModal = ({ 
    selectedDocuments, 
    onReingestion, 
    onCancel,
    isVisible = false,
    isProcessing = false,
    documents = [],
    defaultConfigs = {}
}) => {
    const [documentsInfo, setDocumentsInfo] = useState([]);
    const [bulkMethod, setBulkMethod] = useState('');
    const [showAdvancedConfig, setShowAdvancedConfig] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');

    // Extract document information for display when modal opens
    useEffect(() => {
        if (selectedDocuments && selectedDocuments.size > 0 && documents.length > 0 && isVisible) {
            const selectedDocIds = Array.from(selectedDocuments);
            const docs = documents.filter(doc => selectedDocIds.includes(doc.id)).map(doc => {
                const fileTypeInfo = getFileTypeAndMethod(doc.filename);
                return {
                    id: doc.id,
                    filename: doc.filename,
                    size: doc.size || 0,
                    content_type: doc.content_type,
                    currentChunkingMethod: doc.chunking_method || 'general',
                    // Set the method to current chunking method or recommended default
                    method: doc.chunking_method || fileTypeInfo.defaultMethod,
                    config: defaultConfigs[doc.chunking_method || fileTypeInfo.defaultMethod] || {},
                    fileType: fileTypeInfo.type,
                    icon: fileTypeInfo.icon,
                    category: fileTypeInfo.category
                };
            });
            setDocumentsInfo(docs);
        }
    }, [selectedDocuments, documents, isVisible, defaultConfigs]);

    // Handle per-document method change
    const handleMethodChange = (docId, method) => {
        setDocumentsInfo(docs => docs.map(d => 
            d.id === docId 
                ? { 
                    ...d, 
                    method, 
                    config: { ...defaultConfigs[method] } || {} 
                }
                : d
        ));
    };

    // Handle per-document config change
    const handleConfigChange = (docId, configKey, value) => {
        setDocumentsInfo(docs => docs.map(d => 
            d.id === docId 
                ? { 
                    ...d, 
                    config: { ...d.config, [configKey]: value } 
                }
                : d
        ));
    };

    // Handle bulk method change
    const handleBulkMethodChange = (method) => {
        setBulkMethod(method);
        if (method) {
            setDocumentsInfo(docs => docs.map(d => ({
                ...d,
                method,
                config: { ...defaultConfigs[method] } || {}
            })));
        }
    };

    const handleSubmit = () => {
        // Prepare the reingestion data with per-document configurations
        const reingestionData = documentsInfo.map(doc => ({
            document_id: doc.id,
            chunking_method: doc.method,
            chunking_config: doc.config
        }));

        onReingestion(reingestionData);
    };

    // Filter documents based on search term
    const getFilteredDocuments = () => {
        if (!searchTerm) return documentsInfo;
        return documentsInfo.filter(doc => 
            doc.filename.toLowerCase().includes(searchTerm.toLowerCase())
        );
    };

    if (!isVisible) return null;

    return (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
            <div className="relative top-10 mx-auto p-6 border w-full max-w-6xl shadow-lg rounded-lg bg-white max-h-[90vh] overflow-y-auto">
                <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xl font-semibold text-gray-900">
                        Reingest Selected Documents ({documentsInfo.length})
                    </h3>
                    <button
                        onClick={onCancel}
                        className="text-gray-400 hover:text-gray-600 transition-colors"
                        disabled={isProcessing}
                    >
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {/* Instructions */}
                <div className="mb-6 p-4 bg-blue-50 rounded-lg">
                    <div className="flex items-start space-x-3">
                        <div className="flex-shrink-0">
                            <svg className="w-5 h-5 text-blue-400 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                            </svg>
                        </div>
                        <div>
                            <h4 className="text-sm font-medium text-blue-800">Document Reingestion</h4>
                            <p className="text-sm text-blue-700 mt-1">
                                Select chunking methods for each document. Documents will be re-processed using the new settings.
                                Existing chunks will be removed and replaced with new ones.
                            </p>
                        </div>
                    </div>
                </div>

                {/* Search and Bulk Actions */}
                <div className="mb-6 space-y-4">
                    {/* Search */}
                    <div className="flex items-center space-x-4">
                        <div className="flex-1">
                            <div className="relative">
                                <svg className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                                </svg>
                                <input
                                    type="text"
                                    placeholder="Search documents..."
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                    className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                />
                            </div>
                        </div>
                    </div>

                    {/* Bulk Method Selection */}
                    <div className="flex items-center space-x-4">
                        <label className="text-sm font-medium text-gray-700">Apply to all:</label>
                        <select
                            value={bulkMethod}
                            onChange={(e) => handleBulkMethodChange(e.target.value)}
                            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            disabled={isProcessing}
                        >
                            <option value="">Select method for all documents</option>
                            {chunkingMethods.map(method => (
                                <option key={method.value} value={method.value}>
                                    {method.label} - {method.description}
                                </option>
                            ))}
                        </select>
                    </div>
                </div>

                {/* Documents List */}
                <div className="bg-gray-50 rounded-lg overflow-hidden">
                    <div className="max-h-96 overflow-y-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-100 sticky top-0">
                                <tr>
                                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Document
                                    </th>
                                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Current Method
                                    </th>
                                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        New Method
                                    </th>
                                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                        Size
                                    </th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {getFilteredDocuments().map((doc) => (
                                    <tr key={doc.id} className="hover:bg-gray-50">
                                        <td className="px-4 py-3">
                                            <div className="flex items-center space-x-3">
                                                <span className="text-2xl">{doc.icon}</span>
                                                <div>
                                                    <div className="font-medium text-gray-900">{doc.filename}</div>
                                                    <div className="text-sm text-gray-500">{doc.fileType}</div>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="px-4 py-3">
                                            <span className="inline-flex px-2 py-1 text-xs font-medium bg-gray-100 text-gray-800 rounded-full">
                                                {doc.currentChunkingMethod}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3">
                                            <select
                                                value={doc.method}
                                                onChange={(e) => handleMethodChange(doc.id, e.target.value)}
                                                className="px-3 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                                disabled={isProcessing}
                                            >
                                                {chunkingMethods.map(method => (
                                                    <option key={method.value} value={method.value}>
                                                        {method.label}
                                                    </option>
                                                ))}
                                            </select>
                                        </td>
                                        <td className="px-4 py-3 text-sm text-gray-500">
                                            {getFileSize(doc.size)}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Advanced Configuration Toggle */}
                <div className="mt-6">
                    <button
                        type="button"
                        onClick={() => setShowAdvancedConfig(!showAdvancedConfig)}
                        className="flex items-center space-x-2 text-sm text-gray-600 hover:text-gray-800"
                        disabled={isProcessing}
                    >
                        <svg 
                            className={`w-4 h-4 transform transition-transform ${showAdvancedConfig ? 'rotate-90' : ''}`} 
                            fill="none" 
                            stroke="currentColor" 
                            viewBox="0 0 24 24"
                        >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                        <span>Advanced Configuration</span>
                    </button>
                </div>

                {/* Advanced Configuration Panel */}
                {showAdvancedConfig && (
                    <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                        <p className="text-sm text-gray-600 mb-4">
                            Advanced chunking configuration will use the default settings for each selected method. 
                            You can modify these in the Chunking Settings tab after reingestion.
                        </p>
                    </div>
                )}

                {/* Action Buttons */}
                <div className="mt-8 flex items-center justify-end space-x-4">
                    <button
                        onClick={onCancel}
                        className="px-6 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                        disabled={isProcessing}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSubmit}
                        className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        disabled={isProcessing || documentsInfo.length === 0}
                    >
                        {isProcessing ? (
                            <span className="flex items-center space-x-2">
                                <svg className="animate-spin -ml-1 mr-3 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                <span>Reingesting...</span>
                            </span>
                        ) : (
                            `Reingest ${documentsInfo.length} Document${documentsInfo.length !== 1 ? 's' : ''}`
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default DocumentReingestionModal;
