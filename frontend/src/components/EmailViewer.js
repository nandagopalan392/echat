import React, { useState, useEffect } from 'react';

const EmailViewer = ({
    document,
    documentPreview,
    selectedChunkId,
    chunks = [],
    onChunkHighlight,
    highlightMode = 'line',
    className = ''
}) => {
    const [viewMode, setViewMode] = useState('formatted'); // 'formatted' or 'raw'
    const [showHeaders, setShowHeaders] = useState(false);

    const emailData = documentPreview || {};
    const headers = emailData.headers || {};
    const body = emailData.body || '';
    const attachments = emailData.attachments || [];

    // Get chunks for different email parts
    const getChunksForPart = (partName) => {
        return chunks.filter(chunk => 
            chunk.metadata?.email_part === partName ||
            chunk.content?.toLowerCase().includes(partName.toLowerCase())
        );
    };

    const headerChunks = getChunksForPart('headers');
    const bodyChunks = getChunksForPart('body');
    const attachmentChunks = getChunksForPart('attachments');

    const handleChunkClick = (chunkId) => {
        if (onChunkHighlight) {
            onChunkHighlight(chunkId);
        }
    };

    const formatDate = (dateStr) => {
        if (!dateStr) return 'Unknown';
        try {
            return new Date(dateStr).toLocaleString();
        } catch {
            return dateStr;
        }
    };

    const renderHeaders = () => {
        const mainHeaders = ['from', 'to', 'cc', 'bcc', 'subject', 'date'];
        const otherHeaders = Object.keys(headers).filter(key => !mainHeaders.includes(key.toLowerCase()));

        return (
            <div className="space-y-3">
                {/* Main headers */}
                <div className="space-y-2">
                    {mainHeaders.map(headerName => {
                        const value = headers[headerName] || headers[headerName.charAt(0).toUpperCase() + headerName.slice(1)];
                        if (!value) return null;

                        return (
                            <div key={headerName} className="flex flex-col sm:flex-row">
                                <div className="w-20 flex-shrink-0 text-sm font-medium text-gray-600 capitalize">
                                    {headerName}:
                                </div>
                                <div className="flex-1 text-sm text-gray-900">
                                    {headerName === 'date' ? formatDate(value) : value}
                                </div>
                            </div>
                        );
                    })}
                </div>

                {/* Additional headers (collapsible) */}
                {otherHeaders.length > 0 && (
                    <div>
                        <button
                            onClick={() => setShowHeaders(!showHeaders)}
                            className="flex items-center space-x-2 text-sm text-blue-600 hover:text-blue-700"
                        >
                            <span>{showHeaders ? 'Hide' : 'Show'} additional headers</span>
                            <svg 
                                className={`w-4 h-4 transform transition-transform ${showHeaders ? 'rotate-180' : ''}`} 
                                fill="none" 
                                stroke="currentColor" 
                                viewBox="0 0 24 24"
                            >
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                            </svg>
                        </button>

                        {showHeaders && (
                            <div className="mt-3 pl-4 border-l-2 border-gray-200 space-y-1">
                                {otherHeaders.map(headerName => (
                                    <div key={headerName} className="flex flex-col sm:flex-row">
                                        <div className="w-32 flex-shrink-0 text-xs font-medium text-gray-500">
                                            {headerName}:
                                        </div>
                                        <div className="flex-1 text-xs text-gray-700 font-mono break-all">
                                            {headers[headerName]}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}
            </div>
        );
    };

    const renderBody = () => {
        if (!body) {
            return (
                <div className="text-center py-8 text-gray-500">
                    No email body content available
                </div>
            );
        }

        return (
            <div className="prose max-w-none">
                {viewMode === 'formatted' ? (
                    <div 
                        className="whitespace-pre-wrap text-sm text-gray-800"
                        dangerouslySetInnerHTML={{
                            __html: body.replace(/\n/g, '<br>').replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" class="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer">$1</a>')
                        }}
                    />
                ) : (
                    <pre className="whitespace-pre-wrap text-sm text-gray-700 font-mono bg-gray-50 p-4 rounded-lg overflow-auto">
                        {body}
                    </pre>
                )}
            </div>
        );
    };

    const renderAttachments = () => {
        if (attachments.length === 0) return null;

        return (
            <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-900">
                    Attachments ({attachments.length})
                </h4>
                <div className="space-y-2">
                    {attachments.map((attachment, index) => (
                        <div key={index} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                            <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                            </svg>
                            <div className="flex-1">
                                <div className="text-sm font-medium text-gray-900">
                                    {attachment.filename || `Attachment ${index + 1}`}
                                </div>
                                {attachment.size && (
                                    <div className="text-xs text-gray-500">
                                        {attachment.size} bytes
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    const renderChunksPanel = () => {
        if (chunks.length === 0) return null;

        const chunkGroups = [
            { name: 'Headers', chunks: headerChunks, color: 'blue' },
            { name: 'Body', chunks: bodyChunks, color: 'green' },
            { name: 'Attachments', chunks: attachmentChunks, color: 'purple' },
            { name: 'Other', chunks: chunks.filter(chunk => 
                !headerChunks.includes(chunk) && 
                !bodyChunks.includes(chunk) && 
                !attachmentChunks.includes(chunk)
            ), color: 'gray' }
        ].filter(group => group.chunks.length > 0);

        return (
            <div className="bg-white border-l border-gray-200 w-80 flex-shrink-0 flex flex-col">
                <div className="p-4 border-b border-gray-200">
                    <h4 className="text-sm font-medium text-gray-900">
                        Email Chunks
                    </h4>
                    <p className="text-xs text-gray-500 mt-1">
                        {chunks.length} chunk{chunks.length !== 1 ? 's' : ''}
                    </p>
                </div>
                <div className="flex-1 overflow-auto p-4 space-y-4">
                    {chunkGroups.map(group => (
                        <div key={group.name}>
                            <h5 className="text-xs font-semibold text-gray-700 uppercase tracking-wide mb-2">
                                {group.name} ({group.chunks.length})
                            </h5>
                            <div className="space-y-2">
                                {group.chunks.map(chunk => (
                                    <div
                                        key={chunk.id}
                                        onClick={() => handleChunkClick(chunk.id)}
                                        className={`
                                            p-3 rounded-lg border cursor-pointer transition-colors
                                            ${chunk.id === selectedChunkId 
                                                ? `border-${group.color}-300 bg-${group.color}-50` 
                                                : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                                            }
                                        `}
                                    >
                                        <div className="flex items-center justify-between mb-2">
                                            <span className="text-xs font-medium text-gray-500">
                                                Chunk {chunk.chunk_number}
                                            </span>
                                            <span className="text-xs text-gray-400">
                                                {chunk.content?.length || 0} chars
                                            </span>
                                        </div>
                                        <div className="text-sm text-gray-700 line-clamp-3">
                                            {chunk.content}
                                        </div>
                                        {chunk.metadata?.email_part && (
                                            <div className="mt-2 text-xs text-gray-500">
                                                Part: {chunk.metadata.email_part}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    if (!documentPreview) {
        return (
            <div className="h-full flex items-center justify-center">
                <div className="text-center">
                    <div className="text-gray-400 mb-4">
                        <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                        </svg>
                    </div>
                    <p className="text-gray-600">Loading email...</p>
                </div>
            </div>
        );
    }

    return (
        <div className={`h-full flex flex-col ${className}`}>
            {/* Header with controls */}
            <div className="flex-shrink-0 bg-white border-b border-gray-200 p-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                        <h3 className="text-lg font-medium text-gray-900">
                            {document?.filename || 'Email'}
                        </h3>
                        <span className="text-sm text-gray-500">
                            {headers.subject || 'No subject'}
                        </span>
                    </div>

                    <div className="flex items-center space-x-3">
                        {/* View mode toggle */}
                        <div className="flex rounded-lg border border-gray-300 overflow-hidden">
                            <button
                                onClick={() => setViewMode('formatted')}
                                className={`px-3 py-1 text-sm ${
                                    viewMode === 'formatted'
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-white text-gray-700 hover:bg-gray-50'
                                }`}
                            >
                                Formatted
                            </button>
                            <button
                                onClick={() => setViewMode('raw')}
                                className={`px-3 py-1 text-sm ${
                                    viewMode === 'raw'
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-white text-gray-700 hover:bg-gray-50'
                                }`}
                            >
                                Raw
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main content area */}
            <div className="flex-1 flex overflow-hidden">
                {/* Email content */}
                <div className="flex-1 overflow-auto bg-white">
                    <div className="p-6 space-y-6">
                        {/* Headers section */}
                        <div className="bg-gray-50 rounded-lg p-4">
                            <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
                                </svg>
                                Email Headers
                                {headerChunks.length > 0 && (
                                    <span className="ml-2 inline-flex items-center px-2 py-1 text-xs font-medium rounded-md bg-blue-100 text-blue-700">
                                        {headerChunks.length} chunk{headerChunks.length !== 1 ? 's' : ''}
                                    </span>
                                )}
                            </h4>
                            {renderHeaders()}
                        </div>

                        {/* Body section */}
                        <div>
                            <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                                Email Body
                                {bodyChunks.length > 0 && (
                                    <span className="ml-2 inline-flex items-center px-2 py-1 text-xs font-medium rounded-md bg-green-100 text-green-700">
                                        {bodyChunks.length} chunk{bodyChunks.length !== 1 ? 's' : ''}
                                    </span>
                                )}
                            </h4>
                            {renderBody()}
                        </div>

                        {/* Attachments section */}
                        {attachments.length > 0 && (
                            <div>
                                <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                                    </svg>
                                    Attachments
                                    {attachmentChunks.length > 0 && (
                                        <span className="ml-2 inline-flex items-center px-2 py-1 text-xs font-medium rounded-md bg-purple-100 text-purple-700">
                                            {attachmentChunks.length} chunk{attachmentChunks.length !== 1 ? 's' : ''}
                                        </span>
                                    )}
                                </h4>
                                {renderAttachments()}
                            </div>
                        )}
                    </div>
                </div>

                {/* Chunks panel */}
                {renderChunksPanel()}
            </div>
        </div>
    );
};

export default EmailViewer;
