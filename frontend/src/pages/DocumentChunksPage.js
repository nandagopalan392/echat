import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { api } from '../services/api';
import PDFViewer from '../components/PDFViewer';
import ImageViewer from '../components/ImageViewer';

const DocumentChunksPage = () => {
    const { docId } = useParams();
    const navigate = useNavigate();
    
    const [documentData, setDocumentData] = useState(null);
    const [documentPreview, setDocumentPreview] = useState(null);
    const [selectedChunkId, setSelectedChunkId] = useState(null);
    const [highlightMode, setHighlightMode] = useState('line');
    const [loading, setLoading] = useState(true);
    const [previewLoading, setPreviewLoading] = useState(false);
    const [error, setError] = useState(null);
    const [currentChunkingConfig, setCurrentChunkingConfig] = useState({
        method: 'general',
        chunk_token_num: 512,
        overlap: 50
    });

    useEffect(() => {
        loadDocumentData();
        loadDocumentPreview();
        loadChunkingConfig();
    }, [docId]);

    const loadDocumentData = async () => {
        try {
            setLoading(true);
            // First, get all documents to find the filename by ID
            const documentsResponse = await api.getDocuments();
            const documents = documentsResponse.documents || [];
            const document = documents.find(doc => doc.id === parseInt(docId));
            
            if (!document) {
                setError('Document not found');
                return;
            }

            // Now get chunks using the filename
            const chunksData = await api.getDocumentChunks(document.filename);
            setDocumentData({
                ...chunksData,
                document_info: { ...chunksData.document_info, ...document }
            });
        } catch (error) {
            console.error('Error loading document data:', error);
            setError('Failed to load document data');
        } finally {
            setLoading(false);
        }
    };

    const loadDocumentPreview = async () => {
        try {
            setPreviewLoading(true);
            const preview = await api.getDocumentPreview(docId);
            setDocumentPreview(preview);
        } catch (error) {
            console.error('Error loading document preview:', error);
        } finally {
            setPreviewLoading(false);
        }
    };

    const loadChunkingConfig = async () => {
        try {
            // Use default 'general' chunking method for all documents
            // This can be enhanced later to map file types to appropriate methods
            const response = await api.call(`/api/chunking/config/general`);
            setCurrentChunkingConfig(response);
        } catch (error) {
            console.error('Error loading chunking config:', error);
            // Set a default config if the API call fails
            setCurrentChunkingConfig({
                config: {
                    method: 'general',
                    chunk_token_num: 1000,
                    chunk_overlap: 200,
                    max_token: 8192
                }
            });
        }
    };

    const handleChunkClick = (chunkId) => {
        setSelectedChunkId(selectedChunkId === chunkId ? null : chunkId);
    };

    const handleChunkHighlight = (chunkId, pageNumber) => {
        setSelectedChunkId(chunkId);
        console.log(`Highlighting chunk ${chunkId} on page ${pageNumber}`);
    };

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
                        <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                    </div>
                    <h2 className="text-xl font-semibold text-gray-900 mb-2">{error}</h2>
                    <button
                        onClick={() => navigate('/knowledge-hub')}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                    >
                        Back to Knowledge Hub
                    </button>
                </div>
            </div>
        );
    }

    const isPDF = documentPreview?.type === 'pdf';
    const isImage = documentPreview?.type === 'image' || 
                   documentPreview?.content_type?.startsWith('image/') ||
                   documentData?.document_info?.content_type?.startsWith('image/') ||
                   /\.(jpg|jpeg|png|gif|bmp|webp|svg)$/i.test(documentData?.filename || '');

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header with Breadcrumb and Controls */}
            <div className="bg-white shadow-sm border-b">
                <div className="px-6 py-4">
                    {/* Breadcrumb */}
                    <nav className="flex items-center space-x-2 text-sm text-gray-600 mb-4">
                        <Link to="/" className="hover:text-blue-600">Home</Link>
                        <span>›</span>
                        <Link to="/knowledge-hub" className="hover:text-blue-600">Documents</Link>
                        <span>›</span>
                        <span className="text-gray-900 font-medium">
                            {documentData?.filename || 'Document'}
                        </span>
                        <span>›</span>
                        <span className="text-blue-600">Chunks</span>
                    </nav>

                    {/* Title and Controls */}
                    <div className="flex items-center justify-between">
                        <div>
                            <h1 className="text-2xl font-bold text-gray-900 flex items-center">
                                📄 Document Chunks
                                {isImage && (
                                    <span className="ml-3 inline-flex items-center px-2 py-1 text-xs font-medium rounded-md bg-indigo-100 text-indigo-700">
                                        📸 Image
                                    </span>
                                )}
                                {isPDF && (
                                    <span className="ml-3 inline-flex items-center px-2 py-1 text-xs font-medium rounded-md bg-red-100 text-red-700">
                                        📄 PDF
                                    </span>
                                )}
                            </h1>
                            <p className="mt-1 text-sm text-gray-500">
                                {documentData?.filename} • {documentData?.total_chunks || 0} chunks
                                {isImage && " • OCR extracted text"}
                            </p>
                        </div>

                        <div className="flex items-center space-x-4">
                            {/* Back Button */}
                            <button
                                onClick={() => navigate('/knowledge-hub')}
                                className="flex items-center px-3 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                            >
                                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                                </svg>
                                Back to Documents
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content Area */}
            <div className="flex h-[calc(100vh-140px)]">
                {/* Document Preview Panel */}
                <div className="w-1/2 bg-white border-r border-gray-200">
                    <div className="h-full flex flex-col">
                        <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
                            <h2 className="text-lg font-medium text-gray-900">Document Preview</h2>
                            <p className="text-sm text-gray-500">{documentData?.filename}</p>
                        </div>
                        
                        <div className="flex-1 overflow-hidden">
                            {previewLoading ? (
                                <div className="h-full flex items-center justify-center">
                                    <div className="flex items-center space-x-3">
                                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                                        <span className="text-gray-600">Loading preview...</span>
                                    </div>
                                </div>
                            ) : documentPreview ? (
                                <>
                                    {isPDF ? (
                                        <PDFViewer
                                            pdfUrl={documentPreview.pdf_url}
                                            selectedChunkId={selectedChunkId}
                                            chunks={documentData?.chunks || []}
                                            onChunkHighlight={handleChunkHighlight}
                                            highlightMode={highlightMode}
                                            className="h-full"
                                        />
                                    ) : isImage ? (
                                        <ImageViewer
                                            imageUrl={documentPreview.image_url}
                                            token={localStorage.getItem('token')}
                                            selectedChunkId={selectedChunkId}
                                            chunks={documentData?.chunks || []}
                                            onChunkHighlight={handleChunkHighlight}
                                            highlightMode={highlightMode}
                                            className="h-full"
                                        />
                                    ) : (
                                        <div className="h-full overflow-auto p-4">
                                            <div className="bg-gray-50 border rounded p-4">
                                                <div className="text-xs text-gray-500 mb-2">
                                                    Content Type: {documentPreview.content_type}
                                                    {documentPreview.truncated && (
                                                        <span className="ml-2 text-orange-600">(Preview truncated)</span>
                                                    )}
                                                </div>
                                                <pre className="whitespace-pre-wrap text-sm text-gray-800 font-mono leading-relaxed">
                                                    {documentPreview.content}
                                                </pre>
                                            </div>
                                        </div>
                                    )}
                                </>
                            ) : (
                                <div className="h-full flex items-center justify-center">
                                    <p className="text-gray-500">No preview available</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Chunks Panel */}
                <div className="w-1/2 bg-white">
                    <div className="h-full flex flex-col">
                        <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
                            <div className="flex items-center justify-between">
                                <h2 className="text-lg font-medium text-gray-900">Document Chunks</h2>
                                <span className="text-sm text-gray-500">
                                    {documentData?.chunks?.length || 0} chunks
                                </span>
                            </div>
                            {selectedChunkId && (
                                <div className="flex items-center justify-between mt-2">
                                    <p className="text-sm text-blue-600">
                                        Selected: Chunk {documentData?.chunks?.find(c => c.id === selectedChunkId)?.chunk_number}
                                    </p>
                                    {/* Highlight Mode Controls for Images */}
                                    {isImage && (
                                        <div className="flex items-center space-x-2">
                                            <span className="text-xs text-gray-700">Highlight:</span>
                                            <select
                                                value={highlightMode}
                                                onChange={(e) => setHighlightMode(e.target.value)}
                                                className="text-xs border border-gray-300 rounded px-2 py-1 bg-white"
                                            >
                                                <option value="area">Area</option>
                                                <option value="full">Full</option>
                                                <option value="outline">Outline</option>
                                            </select>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                        
                        <div className="flex-1 overflow-auto p-4">
                            {!documentData?.chunks?.length ? (
                                <div className="text-center py-8">
                                    <p className="text-gray-500">No chunks found for this document.</p>
                                </div>
                            ) : (
                                <div className="space-y-3">
                                    {documentData.chunks.map((chunk, index) => (
                                        <div 
                                            key={chunk.id} 
                                            className={`border rounded-lg p-4 cursor-pointer transition-all duration-200 ${
                                                selectedChunkId === chunk.id 
                                                    ? 'bg-blue-50 border-blue-300 ring-2 ring-blue-200 shadow-md' 
                                                    : 'bg-gray-50 hover:bg-gray-100 hover:border-gray-300'
                                            }`}
                                            onClick={() => handleChunkClick(chunk.id)}
                                        >
                                            <div className="flex items-center justify-between mb-3">
                                                <div className="flex items-center space-x-3">
                                                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                                        selectedChunkId === chunk.id ? 'bg-blue-200 text-blue-900' : 'bg-blue-100 text-blue-800'
                                                    }`}>
                                                        Chunk {chunk.chunk_number}
                                                    </span>
                                                    <span className="text-sm text-gray-600">
                                                        {chunk.word_count} words
                                                    </span>
                                                    {chunk.embedding_size > 0 && (
                                                        <span className="text-sm text-gray-600">
                                                            {chunk.embedding_size}D vector
                                                        </span>
                                                    )}
                                                    {/* Page number badge */}
                                                    {chunk.metadata?.page_number && (
                                                        <span className="text-xs text-gray-500 bg-gray-200 px-2 py-1 rounded" title="Page number">
                                                            Page {chunk.metadata.page_number}
                                                        </span>
                                                    )}
                                                    {isImage && (
                                                        <span className="inline-flex items-center px-2 py-1 text-xs font-medium rounded-md bg-indigo-100 text-indigo-700">
                                                            OCR Text
                                                        </span>
                                                    )}
                                                </div>
                                                <div className="flex items-center space-x-2">
                                                    {/* Highlight in PDF button */}
                                                    {isPDF && (
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                handleChunkHighlight(chunk.id, chunk.metadata?.page_number);
                                                            }}
                                                            className="text-yellow-600 hover:text-yellow-800 text-xs px-2 py-1 rounded bg-yellow-50 hover:bg-yellow-100 transition-colors"
                                                            title="Highlight in PDF"
                                                        >
                                                            🔍 Highlight
                                                        </button>
                                                    )}
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
                                            </div>
                                            
                                            <div className="text-sm text-gray-700 bg-white p-3 rounded border">
                                                <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed">
                                                    {chunk.content}
                                                </pre>
                                            </div>
                                            
                                            {chunk.metadata && Object.keys(chunk.metadata).length > 0 && (
                                                <div className="mt-2 text-xs text-gray-500">
                                                    <details>
                                                        <summary className="cursor-pointer hover:text-gray-700">
                                                            Metadata ({Object.keys(chunk.metadata).length} fields)
                                                        </summary>
                                                        <pre className="mt-1 bg-gray-100 p-2 rounded text-xs overflow-auto">
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
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DocumentChunksPage;
