import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';

// Set up PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`;

// CSS styles for highlighting
const highlightStyles = `
.pdf-highlight-line {
    background-color: #ffff99 !important;
    border-radius: 3px !important;
    padding: 1px 2px !important;
    margin: -1px -2px !important;
    transition: background-color 0.2s ease !important;
}

.pdf-highlight-line-selected {
    background-color: #ffed4e !important;
    box-shadow: 0 0 0 2px #fbbf24 !important;
}

.pdf-highlight-line-fade {
    background-color: #fef3c7 !important;
}
`;

// Custom file loader with authentication
const createAuthenticatedFileLoader = (url) => {
    const token = localStorage.getItem('token');
    if (!token) {
        throw new Error('No authentication token found');
    }

    return fetch(url, {
        headers: {
            'Authorization': `Bearer ${token}`,
        },
    }).then(response => {
        if (!response.ok) {
            throw new Error(`Failed to load PDF: ${response.status} ${response.statusText}`);
        }
        return response.arrayBuffer();
    });
};

const PDFViewer = ({ 
    pdfUrl, 
    selectedChunkId, 
    chunks = [], 
    onChunkHighlight,
    className = "",
    highlightMode = 'line' // 'line', 'block', or 'none'
}) => {
    const [numPages, setNumPages] = useState(null);
    const [pageNumber, setPageNumber] = useState(1);
    const [scale, setScale] = useState(1.0);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [pdfFile, setPdfFile] = useState(null);
    const [highlightedElements, setHighlightedElements] = useState(new Set());
    const pageRefs = useRef({});
    const textLayerRef = useRef(null);
    const highlightTimeoutRef = useRef(null);

    // Inject CSS styles for highlighting
    useEffect(() => {
        const styleId = 'pdf-highlight-styles';
        if (!document.getElementById(styleId)) {
            const style = document.createElement('style');
            style.id = styleId;
            style.textContent = highlightStyles;
            document.head.appendChild(style);
        }
        
        return () => {
            // Cleanup on unmount
            const style = document.getElementById(styleId);
            if (style) {
                style.remove();
            }
        };
    }, []);

    // Text matching and highlighting functions
    const cleanText = (text) => {
        return text.replace(/\s+/g, ' ').trim().toLowerCase();
    };

    const findTextMatches = (chunkText, textSpans) => {
        if (!chunkText || !textSpans || textSpans.length === 0) return [];
        
        const cleanChunk = cleanText(chunkText);
        const matches = [];
        
        // Split chunk into sentences and phrases for better matching
        const sentences = cleanChunk.split(/[.!?]+/).filter(s => s.trim().length > 10);
        const phrases = cleanChunk.split(/[,;:]+/).filter(p => p.trim().length > 5);
        const words = cleanChunk.split(/\s+/).filter(w => w.length > 3);
        
        // Create search patterns (in order of preference)
        const searchPatterns = [
            ...sentences,
            ...phrases,
            ...words.slice(0, 20) // Limit word matching to avoid too many false positives
        ];
        
        textSpans.forEach((span, index) => {
            const spanText = cleanText(span.textContent || '');
            if (spanText.length < 3) return; // Skip very short text
            
            // Check for matches with different patterns
            for (const pattern of searchPatterns) {
                if (pattern.length > 5 && spanText.includes(pattern)) {
                    matches.push({
                        element: span,
                        confidence: pattern.length / cleanChunk.length,
                        matchType: 'substring',
                        matchText: pattern
                    });
                    break;
                } else if (pattern.length > 10 && pattern.includes(spanText)) {
                    matches.push({
                        element: span,
                        confidence: spanText.length / cleanChunk.length,
                        matchType: 'partial',
                        matchText: spanText
                    });
                    break;
                }
            }
        });
        
        // Sort by confidence and remove duplicates
        return matches
            .sort((a, b) => b.confidence - a.confidence)
            .filter((match, index, arr) => 
                arr.findIndex(m => m.element === match.element) === index
            );
    };

    const clearHighlights = () => {
        highlightedElements.forEach(element => {
            if (element && element.classList) {
                element.classList.remove('pdf-highlight-line', 'pdf-highlight-line-selected', 'pdf-highlight-line-fade');
            }
        });
        setHighlightedElements(new Set());
    };

    const highlightTextInPage = (chunkText, isSelected = false) => {
        if (!chunkText || highlightMode === 'none') return;
        
        // Clear previous highlights
        clearHighlights();
        
        // Find text layer
        const textLayer = document.querySelector('.react-pdf__Page__textContent');
        if (!textLayer) {
            console.warn('Text layer not found for highlighting');
            return;
        }
        
        // Get all text spans
        const textSpans = Array.from(textLayer.querySelectorAll('span'));
        if (textSpans.length === 0) return;
        
        // Find matching text elements
        const matches = findTextMatches(chunkText, textSpans);
        
        if (matches.length === 0) {
            console.warn('No text matches found for chunk:', chunkText.substring(0, 100) + '...');
            return;
        }
        
        const newHighlightedElements = new Set();
        
        // Apply highlights based on mode
        matches.forEach((match, index) => {
            const { element, confidence } = match;
            
            if (highlightMode === 'line') {
                // Line-by-line highlighting
                if (isSelected && index < 3) { // Highlight top 3 matches as selected
                    element.classList.add('pdf-highlight-line-selected');
                } else if (confidence > 0.1) {
                    element.classList.add('pdf-highlight-line');
                } else {
                    element.classList.add('pdf-highlight-line-fade');
                }
            } else if (highlightMode === 'block') {
                // Block highlighting (fallback to your existing method)
                element.classList.add('pdf-highlight-line');
            }
            
            newHighlightedElements.add(element);
        });
        
        setHighlightedElements(newHighlightedElements);
        
        // Scroll to first highlighted element
        if (matches.length > 0 && isSelected) {
            const firstMatch = matches[0].element;
            setTimeout(() => {
                firstMatch.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'center',
                    inline: 'nearest'
                });
            }, 100);
        }
        
        console.log(`Highlighted ${matches.length} text elements for chunk`);
    };

    // Load PDF with authentication
    useEffect(() => {
        if (!pdfUrl) return;

        setLoading(true);
        setError(null);

        createAuthenticatedFileLoader(pdfUrl)
            .then(arrayBuffer => {
                setPdfFile(arrayBuffer);
            })
            .catch(error => {
                console.error('PDF loading error:', error);
                setError('Failed to load PDF document');
                setLoading(false);
            });
    }, [pdfUrl]);

    // Highlight selected chunk when it changes
    useEffect(() => {
        if (selectedChunkId && chunks.length > 0) {
            const selectedChunk = chunks.find(c => c.id === selectedChunkId);
            if (selectedChunk && selectedChunk.content) {
                // Debounce highlighting to avoid rapid updates
                if (highlightTimeoutRef.current) {
                    clearTimeout(highlightTimeoutRef.current);
                }
                
                highlightTimeoutRef.current = setTimeout(() => {
                    highlightTextInPage(selectedChunk.content, true);
                }, 300);
            }
        } else {
            clearHighlights();
        }
        
        return () => {
            if (highlightTimeoutRef.current) {
                clearTimeout(highlightTimeoutRef.current);
            }
        };
    }, [selectedChunkId, chunks, pageNumber, scale]);

    // Clear highlights when page changes
    useEffect(() => {
        clearHighlights();
    }, [pageNumber]);

    const onDocumentLoadSuccess = useCallback(({ numPages }) => {
        setNumPages(numPages);
        setLoading(false);
        setError(null);
    }, []);

    const onDocumentLoadError = useCallback((error) => {
        console.error('PDF loading error:', error);
        setError('Failed to load PDF document');
        setLoading(false);
    }, []);

    const goToPreviousPage = () => {
        setPageNumber(prevPageNumber => Math.max(prevPageNumber - 1, 1));
    };

    const goToNextPage = () => {
        setPageNumber(prevPageNumber => Math.min(prevPageNumber + 1, numPages));
    };

    const zoomIn = () => {
        setScale(prevScale => Math.min(prevScale + 0.2, 3.0));
    };

    const zoomOut = () => {
        setScale(prevScale => Math.max(prevScale - 0.2, 0.5));
    };

    const resetZoom = () => {
        setScale(1.0);
    };

    // Find which page a chunk belongs to (simplified logic)
    const getChunkPageNumber = (chunk) => {
        if (!chunk || !chunk.metadata) return null;
        
        // Look for page number in metadata
        if (chunk.metadata.page_number) {
            return chunk.metadata.page_number;
        }
        
        // Try to extract from chunk index (rough estimation)
        if (chunk.metadata.chunk_index !== undefined && chunks.length > 0) {
            // Rough estimation: distribute chunks evenly across pages
            const chunksPerPage = Math.ceil(chunks.length / (numPages || 1));
            return Math.ceil((chunk.metadata.chunk_index + 1) / chunksPerPage);
        }
        
        return null;
    };

    const highlightChunk = (chunkId) => {
        const chunk = chunks.find(c => c.id === chunkId);
        if (chunk) {
            const pageNum = getChunkPageNumber(chunk);
            if (pageNum && pageNum !== pageNumber) {
                setPageNumber(pageNum);
                // Highlighting will be triggered by the useEffect when page changes
            } else {
                // Already on correct page, highlight immediately
                highlightTextInPage(chunk.content, true);
            }
            
            if (onChunkHighlight) {
                onChunkHighlight(chunkId, pageNum);
            }
        }
    };

    const onPageLoadSuccess = useCallback(() => {
        // Re-highlight when page finishes loading
        if (selectedChunkId && chunks.length > 0) {
            const selectedChunk = chunks.find(c => c.id === selectedChunkId);
            if (selectedChunk && selectedChunk.content) {
                setTimeout(() => {
                    highlightTextInPage(selectedChunk.content, true);
                }, 500); // Give text layer time to render
            }
        }
    }, [selectedChunkId, chunks]);

    if (error) {
        return (
            <div className={`flex items-center justify-center p-8 ${className}`}>
                <div className="text-center">
                    <div className="text-red-500 mb-2">
                        <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                    </div>
                    <p className="text-gray-600">{error}</p>
                    <p className="text-sm text-gray-500 mt-2">Please check if the PDF file is accessible</p>
                </div>
            </div>
        );
    }

    return (
        <div className={`flex flex-col h-full ${className}`}>
            {/* PDF Controls */}
            <div className="flex items-center justify-between p-3 bg-gray-50 border-b border-gray-200 shrink-0">
                <div className="flex items-center space-x-2">
                    {/* Navigation */}
                    <button
                        onClick={goToPreviousPage}
                        disabled={pageNumber <= 1}
                        className="p-1 rounded hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Previous page"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                        </svg>
                    </button>
                    
                    <span className="text-sm text-gray-600 min-w-0 px-2">
                        {loading ? 'Loading...' : `${pageNumber} / ${numPages || 0}`}
                    </span>
                    
                    <button
                        onClick={goToNextPage}
                        disabled={pageNumber >= numPages}
                        className="p-1 rounded hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Next page"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                    </button>
                </div>

                <div className="flex items-center space-x-4">
                    {/* Highlight Mode Toggle */}
                    {selectedChunkId && (
                        <div className="flex items-center space-x-2">
                            <span className="text-xs text-gray-600">Highlight:</span>
                            <select
                                value={highlightMode}
                                onChange={(e) => {
                                    const newMode = e.target.value;
                                    if (newMode !== highlightMode && selectedChunkId) {
                                        clearHighlights();
                                        const selectedChunk = chunks.find(c => c.id === selectedChunkId);
                                        if (selectedChunk) {
                                            setTimeout(() => {
                                                highlightTextInPage(selectedChunk.content, true);
                                            }, 100);
                                        }
                                    }
                                }}
                                className="text-xs border border-gray-300 rounded px-2 py-1"
                            >
                                <option value="line">Line</option>
                                <option value="block">Block</option>
                                <option value="none">None</option>
                            </select>
                        </div>
                    )}

                    {/* Zoom Controls */}
                    <div className="flex items-center space-x-2">
                        <button
                            onClick={zoomOut}
                            className="p-1 rounded hover:bg-gray-200"
                            title="Zoom out"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                            </svg>
                        </button>
                        
                        <span className="text-xs text-gray-600 min-w-0 px-1">
                            {Math.round(scale * 100)}%
                        </span>
                        
                        <button
                            onClick={zoomIn}
                            className="p-1 rounded hover:bg-gray-200"
                            title="Zoom in"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                            </svg>
                        </button>
                        
                        <button
                            onClick={resetZoom}
                            className="p-1 rounded hover:bg-gray-200 text-xs"
                            title="Reset zoom"
                        >
                            Reset
                        </button>
                    </div>
                </div>
            </div>

            {/* PDF Content */}
            <div className="flex-1 overflow-auto bg-gray-100 p-4">
                <div className="flex justify-center">
                    <div className="relative inline-block bg-white shadow-lg">
                        {loading && (
                            <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 z-10">
                                <div className="flex items-center space-x-2">
                                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                                    <span className="text-sm text-gray-600">Loading PDF...</span>
                                </div>
                            </div>
                        )}
                        
                        {pdfFile && (
                            <Document
                                file={pdfFile}
                                onLoadSuccess={onDocumentLoadSuccess}
                                onLoadError={onDocumentLoadError}
                                loading={null}
                            >
                                <Page
                                    pageNumber={pageNumber}
                                    scale={scale}
                                    loading={null}
                                    renderTextLayer={true}
                                    renderAnnotationLayer={true}
                                    onLoadSuccess={onPageLoadSuccess}
                                />
                            </Document>
                        )}
                    </div>
                </div>
            </div>

            {/* Chunk Navigation Helper */}
            {selectedChunkId && (
                <div className="p-2 bg-blue-50 border-t border-blue-200 text-sm">
                    <div className="flex items-center justify-between">
                        <span className="text-blue-700">
                            Chunk highlighted: {highlightedElements.size} text elements found
                        </span>
                        <button
                            onClick={() => highlightChunk(selectedChunkId)}
                            className="text-blue-600 hover:text-blue-800 text-xs"
                        >
                            Go to chunk
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default PDFViewer;
