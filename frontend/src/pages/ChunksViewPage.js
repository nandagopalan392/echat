import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { api } from '../services/api';
import DocumentViewer from '../components/DocumentViewer';

const ChunksViewPage = () => {
    const { docId } = useParams();
    const navigate = useNavigate();
    
    // State for document and chunks
    const [document, setDocument] = useState(null);
    const [chunks, setChunks] = useState([]);
    const [documentPreview, setDocumentPreview] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    
    // UI state
    const [selectedChunkId, setSelectedChunkId] = useState(null);
    const [sidebarWidth, setSidebarWidth] = useState(400);
    const [isResizing, setIsResizing] = useState(false);
    const [highlightMode, setHighlightMode] = useState('line');

    useEffect(() => {
        loadDocumentData();
    }, [docId]);

    const loadDocumentData = async () => {
        try {
            setLoading(true);
            setError(null);

            // Load document info and chunks
            const documentsResponse = await api.getDocuments();
            const foundDoc = documentsResponse.find(doc => doc.id === parseInt(docId));
            
            if (!foundDoc) {
                setError('Document not found');
                return;
            }
            
            setDocument(foundDoc);

            // Load chunks
            const chunksResponse = await api.getDocumentChunks(foundDoc.filename);
            setChunks(chunksResponse.chunks || []);

            // Load document preview for all supported formats
            try {
                console.log('DEBUG: ChunksViewPage loading preview for document:', foundDoc.id);
                const previewResponse = await api.getDocumentPreview(foundDoc.id);
                console.log('DEBUG: ChunksViewPage received preview:', previewResponse);
                setDocumentPreview(previewResponse);
            } catch (previewError) {
                console.error('Error loading document preview:', previewError);
                // Continue without preview - DocumentViewer will handle gracefully
            }

        } catch (err) {
            console.error('Error loading document data:', err);
            setError('Failed to load document data');
        } finally {
            setLoading(false);
        }
    };

    const handleChunkClick = (chunkId) => {
        setSelectedChunkId(selectedChunkId === chunkId ? null : chunkId);
    };

    const handleChunkHighlight = (chunkId, pageNumber) => {
        setSelectedChunkId(chunkId);
        console.log(`Highlighting chunk ${chunkId} on page ${pageNumber}`);
    };

    const handleMouseDown = (e) => {
        setIsResizing(true);
        e.preventDefault();
    };

    const handleMouseMove = (e) => {
        if (!isResizing) return;
        const newWidth = window.innerWidth - e.clientX;
        setSidebarWidth(Math.max(300, Math.min(800, newWidth)));
    };

    const handleMouseUp = () => {
        setIsResizing(false);
    };

    useEffect(() => {
        if (isResizing) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            return () => {
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', handleMouseUp);
            };
        }
    }, [isResizing]);

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="flex items-center space-x-3">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                    <span className="text-gray-600">Loading document...</span>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="text-center">
                    <div className="text-red-500 mb-4">
                        <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                    </div>
                    <h2 className="text-xl font-semibold text-gray-900 mb-2">Error Loading Document</h2>
                    <p className="text-gray-600 mb-4">{error}</p>
                    <button
                        onClick={() => navigate('/knowledge-hub')}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                    >
                        Back to Knowledge Hub
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50 flex flex-col">
            {/* Header with Breadcrumb and Settings */}
            <div className="bg-white shadow-sm border-b">
                <div className="px-6 py-4">
                    {/* Breadcrumb */}
                    <nav className="flex items-center space-x-2 text-sm text-gray-500 mb-4">
                        <Link to="/knowledge-hub" className="hover:text-gray-700">
                            Home
                        </Link>
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                        <Link to="/knowledge-hub" className="hover:text-gray-700">
                            Documents
                        </Link>
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                        <span className="text-gray-900 font-medium">{document?.filename}</span>
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                        <span className="text-gray-900">Chunks</span>
                    </nav>

                    {/* Header content */}
                    <div className="flex items-center justify-between">
                        <div>
                            <h1 className="text-2xl font-bold text-gray-900">
                                {document?.filename}
                            </h1>
                            <div className="flex items-center space-x-4 mt-2">
                                <span className="text-sm text-gray-500">
                                    {chunks.length} chunks
                                </span>
                                {document?.content_type && (
                                    <span className="inline-flex items-center px-2 py-1 text-xs font-medium rounded-md bg-gray-100 text-gray-700">
                                        {document.content_type}
                                    </span>
                                )}
                                {document?.is_image && (
                                    <span className="inline-flex items-center px-2 py-1 text-xs font-medium rounded-md bg-indigo-100 text-indigo-700">
                                        📸 Image
                                    </span>
                                )}
                            </div>
                        </div>

                        {/* Action buttons */}
                        <div className="flex items-center space-x-3">
                            <button
                                onClick={() => navigate('/knowledge-hub')}
                                className="flex items-center px-3 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                            >
                                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                                </svg>
                                Back to Hub
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content Area */}
            <div className="flex-1 flex overflow-hidden">
                {/* Document Viewer (Left) */}
                <div className="flex-1 bg-white">
                    <DocumentViewer
                        document={document}
                        documentPreview={documentPreview}
                        selectedChunkId={selectedChunkId}
                        chunks={chunks}
                        onChunkHighlight={handleChunkHighlight}
                        highlightMode={highlightMode}
                        className="h-full"
                    />
                </div>

                {/* Resizer */}
                <div
                    className="w-1 bg-gray-200 hover:bg-gray-300 cursor-col-resize"
                    onMouseDown={handleMouseDown}
                />

                {/* Chunks Sidebar (Right) */}
                <div 
                    className="bg-white border-l overflow-hidden flex flex-col"
                    style={{ width: `${sidebarWidth}px` }}
                >
                    {/* Sidebar Header */}
                    <div className="p-4 border-b bg-gray-50">
                        <div className="flex items-center justify-between mb-2">
                            <h2 className="text-lg font-semibold text-gray-900">Document Chunks</h2>
                            {selectedChunkId && (
                                <div className="flex items-center space-x-2">
                                    <span className="text-xs text-gray-700">Highlight:</span>
                                    <select
                                        value={highlightMode}
                                        onChange={(e) => setHighlightMode(e.target.value)}
                                        className="text-xs border border-gray-300 rounded px-2 py-1 bg-white"
                                    >
                                        <option value="line">Line</option>
                                        <option value="area">Area</option>
                                        <option value="full">Full</option>
                                        <option value="outline">Outline</option>
                                    </select>
                                </div>
                            )}
                        </div>
                        <p className="text-sm text-gray-600">
                            {chunks.length} chunks • Click to highlight in document
                            {selectedChunkId && (
                                <span className="text-blue-600 ml-2">
                                    • Selected: Chunk {chunks.find(c => c.id === selectedChunkId)?.chunk_number}
                                </span>
                            )}
                        </p>
                    </div>

                    {/* Chunks List */}
                    <div className="flex-1 overflow-y-auto p-4">
                        {chunks.length === 0 ? (
                            <div className="text-center py-8">
                                <p className="text-gray-500">No chunks found for this document.</p>
                            </div>
                        ) : (
                            <div className="space-y-3">
                                {chunks.map((chunk, index) => (
                                    <div 
                                        key={chunk.id} 
                                        className={`border rounded-lg p-4 cursor-pointer transition-all ${
                                            selectedChunkId === chunk.id 
                                                ? 'bg-blue-50 border-blue-300 ring-2 ring-blue-200' 
                                                : 'bg-gray-50 hover:bg-gray-100 border-gray-200'
                                        }`}
                                        onClick={() => handleChunkClick(chunk.id)}
                                    >
                                        <div className="flex items-center justify-between mb-2">
                                            <div className="flex items-center space-x-3">
                                                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                                    selectedChunkId === chunk.id ? 'bg-blue-200 text-blue-900' : 'bg-blue-100 text-blue-800'
                                                }`}>
                                                    Chunk {chunk.chunk_number}
                                                </span>
                                                <span className="text-sm text-gray-600">
                                                    {chunk.word_count} words
                                                </span>
                                                {chunk.metadata?.page_number && (
                                                    <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded" title="Page number">
                                                        Page {chunk.metadata.page_number}
                                                    </span>
                                                )}
                                            </div>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    navigator.clipboard.writeText(chunk.content);
                                                }}
                                                className="text-gray-400 hover:text-gray-600 text-sm"
                                                title="Copy to clipboard"
                                            >
                                                📋
                                            </button>
                                        </div>
                                        
                                        <div className="text-sm text-gray-700 bg-white p-3 rounded border max-h-32 overflow-y-auto">
                                            <pre className="whitespace-pre-wrap font-mono text-xs">
                                                {chunk.content}
                                            </pre>
                                        </div>
                                        
                                        {chunk.metadata && Object.keys(chunk.metadata).length > 0 && (
                                            <div className="mt-2 text-xs text-gray-500">
                                                <details>
                                                    <summary className="cursor-pointer">Metadata</summary>
                                                    <pre className="mt-1 bg-gray-100 p-2 rounded text-xs">
                                                        {JSON.stringify(chunk.metadata, null, 2)}
                                                    </pre>
                                                </details>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Sidebar Footer */}
                    {selectedChunkId && (
                        <div className="p-3 bg-blue-50 border-t border-blue-200">
                            <div className="text-sm text-blue-700">
                                <span className="font-medium">Selected:</span> Chunk {chunks.find(c => c.id === selectedChunkId)?.chunk_number}
                                <span className="ml-2 text-blue-600">
                                    (highlighted in viewer)
                                </span>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Chunking Config Modal */}
            {/* Add modal here if needed */}
        </div>
    );
};

export default ChunksViewPage;
