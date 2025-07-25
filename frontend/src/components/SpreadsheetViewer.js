import React, { useState, useEffect } from 'react';

const SpreadsheetViewer = ({
    document,
    documentPreview,
    selectedChunkId,
    chunks = [],
    onChunkHighlight,
    highlightMode = 'line',
    className = ''
}) => {
    const [currentSheet, setCurrentSheet] = useState(0);
    const [viewMode, setViewMode] = useState('table'); // 'table' or 'raw'

    const sheets = documentPreview?.sheets || [];
    const sheetNames = Object.keys(sheets);

    // Auto-select sheet that contains the selected chunk
    useEffect(() => {
        if (selectedChunkId && sheetNames.length > 0) {
            const selectedChunk = chunks.find(chunk => chunk.id === selectedChunkId);
            if (selectedChunk && selectedChunk.metadata?.sheet_name) {
                const sheetIndex = sheetNames.indexOf(selectedChunk.metadata.sheet_name);
                if (sheetIndex >= 0) {
                    setCurrentSheet(sheetIndex);
                }
            }
        }
    }, [selectedChunkId, sheetNames, chunks]);

    const getCurrentSheetData = () => {
        if (sheetNames.length === 0) return null;
        const sheetName = sheetNames[currentSheet];
        return sheets[sheetName];
    };

    const getSheetChunks = (sheetName) => {
        return chunks.filter(chunk => 
            chunk.metadata?.sheet_name === sheetName ||
            chunk.content?.includes(`Sheet: ${sheetName}`)
        );
    };

    const handleChunkClick = (chunkId) => {
        if (onChunkHighlight) {
            onChunkHighlight(chunkId);
        }
    };

    const renderTableView = (sheetData, sheetName) => {
        if (!sheetData || !Array.isArray(sheetData) || sheetData.length === 0) {
            return (
                <div className="text-center py-8 text-gray-500">
                    No data available for this sheet
                </div>
            );
        }

        // Get headers from first row
        const headers = sheetData[0] || [];
        const rows = sheetData.slice(1);

        return (
            <div className="overflow-auto">
                <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50 sticky top-0">
                        <tr>
                            <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-r border-gray-200">
                                #
                            </th>
                            {headers.map((header, index) => (
                                <th
                                    key={index}
                                    className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-r border-gray-200"
                                >
                                    {header || `Column ${index + 1}`}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {rows.map((row, rowIndex) => (
                            <tr key={rowIndex} className="hover:bg-gray-50">
                                <td className="px-3 py-2 text-sm text-gray-500 border-r border-gray-200 bg-gray-50">
                                    {rowIndex + 1}
                                </td>
                                {headers.map((_, colIndex) => (
                                    <td
                                        key={colIndex}
                                        className="px-3 py-2 text-sm text-gray-900 border-r border-gray-200"
                                    >
                                        {row[colIndex] || ''}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        );
    };

    const renderRawView = (sheetData, sheetName) => {
        const rawContent = Array.isArray(sheetData) 
            ? sheetData.map(row => Array.isArray(row) ? row.join('\t') : row).join('\n')
            : JSON.stringify(sheetData, null, 2);

        return (
            <pre className="whitespace-pre-wrap text-sm text-gray-700 font-mono bg-gray-50 p-4 rounded-lg overflow-auto">
                {rawContent}
            </pre>
        );
    };

    const renderChunksPanel = (sheetName) => {
        const sheetChunks = getSheetChunks(sheetName);
        
        if (sheetChunks.length === 0) return null;

        return (
            <div className="bg-white border-l border-gray-200 w-80 flex-shrink-0 flex flex-col">
                <div className="p-4 border-b border-gray-200">
                    <h4 className="text-sm font-medium text-gray-900">
                        Chunks in {sheetName}
                    </h4>
                    <p className="text-xs text-gray-500 mt-1">
                        {sheetChunks.length} chunk{sheetChunks.length !== 1 ? 's' : ''}
                    </p>
                </div>
                <div className="flex-1 overflow-auto p-4 space-y-3">
                    {sheetChunks.map(chunk => (
                        <div
                            key={chunk.id}
                            onClick={() => handleChunkClick(chunk.id)}
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
                                    {chunk.content?.length || 0} chars
                                </span>
                            </div>
                            <div className="text-sm text-gray-700 line-clamp-3">
                                {chunk.content}
                            </div>
                            {chunk.metadata?.sheet_name && (
                                <div className="mt-2 text-xs text-gray-500">
                                    Sheet: {chunk.metadata.sheet_name}
                                </div>
                            )}
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
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2-2V7a2 2 0 012-2h2a2 2 0 002 2v2a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 00-2 2h-2a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2v-2a2 2 0 00-2-2h-2a2 2 0 00-2-2V9a2 2 0 012-2h2a2 2 0 012 2v10a2 2 0 01-2 2H9a2 2 0 01-2-2z" />
                        </svg>
                    </div>
                    <p className="text-gray-600">Loading spreadsheet...</p>
                </div>
            </div>
        );
    }

    const currentSheetName = sheetNames[currentSheet];
    const currentSheetData = getCurrentSheetData();
    const hasChunks = chunks.length > 0;

    return (
        <div className={`h-full flex flex-col ${className}`}>
            {/* Header with controls */}
            <div className="flex-shrink-0 bg-white border-b border-gray-200 p-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                        <h3 className="text-lg font-medium text-gray-900">
                            {document?.filename || 'Spreadsheet'}
                        </h3>
                        <span className="text-sm text-gray-500">
                            {sheetNames.length} sheet{sheetNames.length !== 1 ? 's' : ''}
                        </span>
                    </div>

                    <div className="flex items-center space-x-3">
                        {/* View mode toggle */}
                        <div className="flex rounded-lg border border-gray-300 overflow-hidden">
                            <button
                                onClick={() => setViewMode('table')}
                                className={`px-3 py-1 text-sm ${
                                    viewMode === 'table'
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-white text-gray-700 hover:bg-gray-50'
                                }`}
                            >
                                Table
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

                {/* Sheet tabs */}
                {sheetNames.length > 1 && (
                    <div className="mt-3 flex space-x-2 overflow-x-auto">
                        {sheetNames.map((sheetName, index) => {
                            const sheetChunks = getSheetChunks(sheetName);
                            const hasSelectedChunk = sheetChunks.some(chunk => chunk.id === selectedChunkId);
                            
                            return (
                                <button
                                    key={index}
                                    onClick={() => setCurrentSheet(index)}
                                    className={`
                                        flex-shrink-0 px-3 py-2 text-sm rounded-lg border transition-colors
                                        ${index === currentSheet
                                            ? 'bg-blue-600 text-white border-blue-600'
                                            : 'bg-white text-gray-700 border-gray-200 hover:border-gray-300'
                                        }
                                        ${hasSelectedChunk ? 'ring-2 ring-blue-300' : ''}
                                    `}
                                >
                                    <span>{sheetName}</span>
                                    {sheetChunks.length > 0 && (
                                        <span className={`ml-2 inline-flex items-center px-1.5 py-0.5 text-xs font-medium rounded-full ${
                                            index === currentSheet
                                                ? 'bg-blue-500 text-white'
                                                : 'bg-gray-100 text-gray-600'
                                        }`}>
                                            {sheetChunks.length}
                                        </span>
                                    )}
                                </button>
                            );
                        })}
                    </div>
                )}
            </div>

            {/* Main content area */}
            <div className="flex-1 flex overflow-hidden">
                {/* Sheet content */}
                <div className="flex-1 overflow-auto bg-white">
                    {currentSheetData ? (
                        <div className="h-full">
                            {viewMode === 'table' 
                                ? renderTableView(currentSheetData, currentSheetName)
                                : renderRawView(currentSheetData, currentSheetName)
                            }
                        </div>
                    ) : (
                        <div className="h-full flex items-center justify-center text-gray-500">
                            <div className="text-center">
                                <svg className="w-12 h-12 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                                <p>No data available for this sheet</p>
                            </div>
                        </div>
                    )}
                </div>

                {/* Chunks panel */}
                {hasChunks && renderChunksPanel(currentSheetName)}
            </div>
        </div>
    );
};

export default SpreadsheetViewer;
