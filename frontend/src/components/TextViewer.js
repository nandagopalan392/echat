import React, { useState, useEffect, useRef } from 'react';

const TextViewer = ({
    document,
    documentPreview,
    selectedChunkId,
    chunks = [],
    onChunkHighlight,
    highlightMode = 'line',
    className = ''
}) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [searchResults, setSearchResults] = useState([]);
    const [currentSearchIndex, setCurrentSearchIndex] = useState(0);
    const [showLineNumbers, setShowLineNumbers] = useState(true);
    const [fontSize, setFontSize] = useState(14);
    const textRef = useRef(null);

    const content = documentPreview?.content || documentPreview?.text || '';
    const lines = content.split('\n');

    // Search functionality
    useEffect(() => {
        if (searchTerm && content) {
            const regex = new RegExp(searchTerm, 'gi');
            const matches = [...content.matchAll(regex)];
            setSearchResults(matches.map(match => ({
                index: match.index,
                text: match[0],
                line: content.substring(0, match.index).split('\n').length - 1
            })));
            setCurrentSearchIndex(0);
        } else {
            setSearchResults([]);
            setCurrentSearchIndex(0);
        }
    }, [searchTerm, content]);

    // Auto-scroll to selected chunk
    useEffect(() => {
        if (selectedChunkId && textRef.current) {
            const selectedChunk = chunks.find(chunk => chunk.id === selectedChunkId);
            if (selectedChunk && selectedChunk.metadata?.line_start) {
                const lineElement = textRef.current.querySelector(`[data-line="${selectedChunk.metadata.line_start}"]`);
                if (lineElement) {
                    lineElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }
        }
    }, [selectedChunkId, chunks]);

    const getChunkForLine = (lineNumber) => {
        return chunks.find(chunk => {
            const lineStart = chunk.metadata?.line_start || 0;
            const lineEnd = chunk.metadata?.line_end || lineStart;
            return lineNumber >= lineStart && lineNumber <= lineEnd;
        });
    };

    const isLineInSelectedChunk = (lineNumber) => {
        if (!selectedChunkId) return false;
        const selectedChunk = chunks.find(chunk => chunk.id === selectedChunkId);
        if (!selectedChunk) return false;
        
        const lineStart = selectedChunk.metadata?.line_start || 0;
        const lineEnd = selectedChunk.metadata?.line_end || lineStart;
        return lineNumber >= lineStart && lineNumber <= lineEnd;
    };

    const isLineInSearch = (lineNumber) => {
        if (!searchTerm || searchResults.length === 0) return false;
        return searchResults.some(result => result.line === lineNumber);
    };

    const handleLineClick = (lineNumber) => {
        const chunk = getChunkForLine(lineNumber);
        if (chunk && onChunkHighlight) {
            onChunkHighlight(chunk.id);
        }
    };

    const handleSearch = (direction) => {
        if (searchResults.length === 0) return;
        
        let newIndex;
        if (direction === 'next') {
            newIndex = (currentSearchIndex + 1) % searchResults.length;
        } else {
            newIndex = currentSearchIndex === 0 ? searchResults.length - 1 : currentSearchIndex - 1;
        }
        
        setCurrentSearchIndex(newIndex);
        
        // Scroll to the search result
        if (textRef.current) {
            const result = searchResults[newIndex];
            const lineElement = textRef.current.querySelector(`[data-line="${result.line}"]`);
            if (lineElement) {
                lineElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
    };

    const highlightSearchTerms = (text, lineNumber) => {
        if (!searchTerm) return text;
        
        const regex = new RegExp(`(${searchTerm})`, 'gi');
        const parts = text.split(regex);
        
        return parts.map((part, index) => {
            if (regex.test(part)) {
                const isCurrentResult = searchResults[currentSearchIndex]?.line === lineNumber;
                return (
                    <span
                        key={index}
                        className={`${isCurrentResult ? 'bg-yellow-300' : 'bg-yellow-200'} px-1 rounded`}
                    >
                        {part}
                    </span>
                );
            }
            return part;
        });
    };

    const renderChunksPanel = () => {
        if (chunks.length === 0) return null;

        return (
            <div className="bg-white border-l border-gray-200 w-80 flex-shrink-0 flex flex-col">
                <div className="p-4 border-b border-gray-200">
                    <h4 className="text-sm font-medium text-gray-900">
                        Text Chunks
                    </h4>
                    <p className="text-xs text-gray-500 mt-1">
                        {chunks.length} chunk{chunks.length !== 1 ? 's' : ''}
                    </p>
                </div>
                <div className="flex-1 overflow-auto p-4 space-y-3">
                    {chunks.map(chunk => (
                        <div
                            key={chunk.id}
                            onClick={() => onChunkHighlight && onChunkHighlight(chunk.id)}
                            className={`
                                p-3 rounded-lg border cursor-pointer transition-colors
                                ${chunk.id === selectedChunkId 
                                    ? 'border-blue-300 bg-blue-50' 
                                    : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                                }
                            `}
                        >
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-xs font-medium text-gray-500">
                                    Chunk {chunk.chunk_number}
                                </span>
                                <span className="text-xs text-gray-400">
                                    {chunk.metadata?.line_start && chunk.metadata?.line_end 
                                        ? `Lines ${chunk.metadata.line_start}-${chunk.metadata.line_end}`
                                        : `${chunk.content?.length || 0} chars`
                                    }
                                </span>
                            </div>
                            <div className="text-sm text-gray-700 line-clamp-4 font-mono">
                                {chunk.content}
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
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                    </div>
                    <p className="text-gray-600">Loading text document...</p>
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
                            {document?.filename || 'Text Document'}
                        </h3>
                        <span className="text-sm text-gray-500">
                            {lines.length} line{lines.length !== 1 ? 's' : ''}
                        </span>
                    </div>

                    <div className="flex items-center space-x-3">
                        {/* Search */}
                        <div className="flex items-center space-x-2">
                            <div className="relative">
                                <input
                                    type="text"
                                    placeholder="Search..."
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                    className="pl-8 pr-3 py-1 text-sm border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                                />
                                <svg className="absolute left-2 top-1.5 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                                </svg>
                            </div>
                            
                            {searchResults.length > 0 && (
                                <div className="flex items-center space-x-1">
                                    <span className="text-xs text-gray-500">
                                        {currentSearchIndex + 1} of {searchResults.length}
                                    </span>
                                    <button
                                        onClick={() => handleSearch('prev')}
                                        className="p-1 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded"
                                    >
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                                        </svg>
                                    </button>
                                    <button
                                        onClick={() => handleSearch('next')}
                                        className="p-1 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded"
                                    >
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                        </svg>
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* Font size controls */}
                        <div className="flex items-center space-x-2">
                            <button
                                onClick={() => setFontSize(Math.max(10, fontSize - 2))}
                                className="p-1 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded"
                                title="Decrease font size"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                                </svg>
                            </button>
                            <span className="text-xs text-gray-500 min-w-[2rem] text-center">
                                {fontSize}px
                            </span>
                            <button
                                onClick={() => setFontSize(Math.min(24, fontSize + 2))}
                                className="p-1 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded"
                                title="Increase font size"
                            >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                                </svg>
                            </button>
                        </div>

                        {/* Line numbers toggle */}
                        <button
                            onClick={() => setShowLineNumbers(!showLineNumbers)}
                            className={`px-3 py-1 text-sm rounded-md border ${
                                showLineNumbers
                                    ? 'bg-blue-600 text-white border-blue-600'
                                    : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                            }`}
                        >
                            Line #
                        </button>
                    </div>
                </div>
            </div>

            {/* Main content area */}
            <div className="flex-1 flex overflow-hidden">
                {/* Text content */}
                <div className="flex-1 overflow-auto bg-gray-50">
                    <div 
                        ref={textRef}
                        className="p-4 font-mono text-gray-800"
                        style={{ fontSize: `${fontSize}px` }}
                    >
                        {lines.map((line, index) => {
                            const lineNumber = index + 1;
                            const chunk = getChunkForLine(lineNumber);
                            const isSelected = isLineInSelectedChunk(lineNumber);
                            const hasSearch = isLineInSearch(lineNumber);
                            
                            return (
                                <div
                                    key={lineNumber}
                                    data-line={lineNumber}
                                    onClick={() => handleLineClick(lineNumber)}
                                    className={`
                                        flex hover:bg-gray-100 transition-colors
                                        ${isSelected ? 'bg-blue-50 border-l-4 border-blue-500' : ''}
                                        ${chunk ? 'cursor-pointer' : ''}
                                        ${hasSearch ? 'bg-yellow-50' : ''}
                                    `}
                                >
                                    {showLineNumbers && (
                                        <div className="flex-shrink-0 w-12 text-right pr-4 text-gray-400 select-none text-xs leading-6 border-r border-gray-200 bg-gray-50">
                                            {lineNumber}
                                        </div>
                                    )}
                                    <div className="flex-1 px-4 py-1 whitespace-pre-wrap break-words leading-6">
                                        {highlightSearchTerms(line, index)}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Chunks panel */}
                {renderChunksPanel()}
            </div>
        </div>
    );
};

export default TextViewer;
