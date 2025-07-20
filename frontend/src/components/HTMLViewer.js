import React, { useState, useEffect, useRef } from 'react';

const HTMLViewer = ({
    document,
    documentPreview,
    selectedChunkId,
    chunks = [],
    onChunkHighlight,
    highlightMode = 'line',
    className = ''
}) => {
    const [viewMode, setViewMode] = useState('rendered'); // 'rendered', 'source'
    const [showMetadata, setShowMetadata] = useState(false);
    const iframeRef = useRef(null);

    const htmlContent = documentPreview?.content || documentPreview?.html || '';
    const metadata = documentPreview?.metadata || {};
    const title = metadata.title || document?.filename || 'HTML Document';

    // Get chunks for different HTML parts
    const getChunksForPart = (partName) => {
        return chunks.filter(chunk => 
            chunk.metadata?.html_part === partName ||
            chunk.content?.toLowerCase().includes(`<${partName}`)
        );
    };

    const headChunks = getChunksForPart('head');
    const bodyChunks = getChunksForPart('body');
    const scriptChunks = getChunksForPart('script');
    const styleChunks = getChunksForPart('style');

    const handleChunkClick = (chunkId) => {
        if (onChunkHighlight) {
            onChunkHighlight(chunkId);
        }
    };

    // Create a safe HTML content for iframe
    const createSafeHtmlContent = () => {
        // Remove any script tags and dangerous attributes for security
        let safeContent = htmlContent
            .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
            .replace(/javascript:/gi, '')
            .replace(/on\w+\s*=/gi, '');
        
        // If it's a partial HTML (no html/body tags), wrap it
        if (!safeContent.toLowerCase().includes('<html')) {
            safeContent = `
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>${title}</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                        img { max-width: 100%; height: auto; }
                        table { border-collapse: collapse; width: 100%; }
                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        th { background-color: #f2f2f2; }
                    </style>
                </head>
                <body>
                    ${safeContent}
                </body>
                </html>
            `;
        }
        
        return safeContent;
    };

    const renderSourceView = () => {
        return (
            <div className="h-full overflow-auto bg-gray-50 p-4">
                <pre className="whitespace-pre-wrap text-sm text-gray-700 font-mono bg-white p-4 rounded-lg border">
                    {htmlContent}
                </pre>
            </div>
        );
    };

    const renderRenderedView = () => {
        const safeContent = createSafeHtmlContent();
        
        return (
            <div className="h-full overflow-auto bg-white">
                <iframe
                    ref={iframeRef}
                    srcDoc={safeContent}
                    className="w-full h-full border-0"
                    title="HTML Content"
                    sandbox="allow-same-origin allow-popups"
                />
            </div>
        );
    };

    const renderMetadata = () => {
        if (!metadata || Object.keys(metadata).length === 0) {
            return (
                <div className="text-sm text-gray-500">
                    No metadata available
                </div>
            );
        }

        return (
            <div className="space-y-2">
                {Object.entries(metadata).map(([key, value]) => (
                    <div key={key} className="flex flex-col sm:flex-row">
                        <div className="w-24 flex-shrink-0 text-xs font-medium text-gray-600 capitalize">
                            {key.replace(/_/g, ' ')}:
                        </div>
                        <div className="flex-1 text-xs text-gray-900 font-mono">
                            {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                        </div>
                    </div>
                ))}
            </div>
        );
    };

    const renderChunksPanel = () => {
        if (chunks.length === 0) return null;

        const chunkGroups = [
            { name: 'Head', chunks: headChunks, color: 'blue' },
            { name: 'Body', chunks: bodyChunks, color: 'green' },
            { name: 'Scripts', chunks: scriptChunks, color: 'red' },
            { name: 'Styles', chunks: styleChunks, color: 'purple' },
            { name: 'Other', chunks: chunks.filter(chunk => 
                !headChunks.includes(chunk) && 
                !bodyChunks.includes(chunk) && 
                !scriptChunks.includes(chunk) && 
                !styleChunks.includes(chunk)
            ), color: 'gray' }
        ].filter(group => group.chunks.length > 0);

        return (
            <div className="bg-white border-l border-gray-200 w-80 flex-shrink-0 flex flex-col">
                <div className="p-4 border-b border-gray-200">
                    <h4 className="text-sm font-medium text-gray-900">
                        HTML Chunks
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
                                        <div className="text-sm text-gray-700 line-clamp-3 font-mono">
                                            {chunk.content}
                                        </div>
                                        {chunk.metadata?.html_part && (
                                            <div className="mt-2 text-xs text-gray-500">
                                                Part: {chunk.metadata.html_part}
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
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                        </svg>
                    </div>
                    <p className="text-gray-600">Loading HTML document...</p>
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
                            {title}
                        </h3>
                        {metadata.description && (
                            <span className="text-sm text-gray-500 max-w-md truncate">
                                {metadata.description}
                            </span>
                        )}
                    </div>

                    <div className="flex items-center space-x-3">
                        {/* Metadata toggle */}
                        <button
                            onClick={() => setShowMetadata(!showMetadata)}
                            className={`px-3 py-1 text-sm rounded-md border ${
                                showMetadata
                                    ? 'bg-gray-600 text-white border-gray-600'
                                    : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                            }`}
                        >
                            Metadata
                        </button>

                        {/* View mode toggle */}
                        <div className="flex rounded-lg border border-gray-300 overflow-hidden">
                            <button
                                onClick={() => setViewMode('rendered')}
                                className={`px-3 py-1 text-sm ${
                                    viewMode === 'rendered'
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-white text-gray-700 hover:bg-gray-50'
                                }`}
                            >
                                Rendered
                            </button>
                            <button
                                onClick={() => setViewMode('source')}
                                className={`px-3 py-1 text-sm ${
                                    viewMode === 'source'
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-white text-gray-700 hover:bg-gray-50'
                                }`}
                            >
                                Source
                            </button>
                        </div>
                    </div>
                </div>

                {/* Metadata section */}
                {showMetadata && (
                    <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                        <h4 className="text-sm font-semibold text-gray-900 mb-3">
                            Document Metadata
                        </h4>
                        {renderMetadata()}
                    </div>
                )}
            </div>

            {/* Main content area */}
            <div className="flex-1 flex overflow-hidden">
                {/* HTML content */}
                <div className="flex-1 overflow-hidden">
                    {viewMode === 'rendered' ? renderRenderedView() : renderSourceView()}
                </div>

                {/* Chunks panel */}
                {renderChunksPanel()}
            </div>

            {/* Footer with security notice */}
            {viewMode === 'rendered' && (
                <div className="flex-shrink-0 bg-yellow-50 border-t border-yellow-200 px-4 py-2">
                    <div className="flex items-center space-x-2 text-xs text-yellow-700">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.464 0L4.35 16.5c-.77.833.192 2.5 1.732 2.5z" />
                        </svg>
                        <span>
                            HTML is rendered in a sandboxed iframe with scripts disabled for security
                        </span>
                    </div>
                </div>
            )}
        </div>
    );
};

export default HTMLViewer;
