import React, { useState, useEffect, useRef, useMemo } from 'react';

const ImageViewer = ({
    documentId,
    token,
    imageUrl, // Support legacy imageUrl prop
    chunks = [],
    selectedChunkId = null,
    onChunkSelect = () => {},
    externalHighlightMode = 'area',
    debugMode = false // New prop for debug visualization
}) => {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [scale, setScale] = useState(1.0);
    const [highlightAreas, setHighlightAreas] = useState([]);
    const [internalHighlightMode, setInternalHighlightMode] = useState(externalHighlightMode);
    const [showDebugBoxes, setShowDebugBoxes] = useState(false);
    const imageRef = useRef(null);
    const containerRef = useRef(null);

    // Update internal highlight mode when external prop changes
    useEffect(() => {
        setInternalHighlightMode(externalHighlightMode);
    }, [externalHighlightMode]);

    // Show all OCR boxes when debug mode is enabled
    useEffect(() => {
        if (showDebugBoxes && chunks.length > 0) {
            // Create debug highlights for all chunks
            const allHighlights = [];
            const colors = [
                'rgb(255, 0, 0)', 'rgb(0, 255, 0)', 'rgb(0, 0, 255)', 'rgb(255, 255, 0)',
                'rgb(255, 0, 255)', 'rgb(0, 255, 255)', 'rgb(255, 128, 0)', 'rgb(128, 255, 0)',
                'rgb(255, 0, 128)', 'rgb(128, 0, 255)', 'rgb(0, 128, 255)', 'rgb(0, 255, 128)'
            ];
            
            chunks.forEach((chunk, index) => {
                const highlights = createChunkHighlight(chunk, colors[index % colors.length], true);
                allHighlights.push(...highlights);
            });
            
            setHighlightAreas(allHighlights);
        }
    }, [showDebugBoxes, chunks]);

    const createChunkHighlight = (chunk, debugColor = null, isDebugMode = false) => {
        if (!chunk || !imageRef.current) return [];

        const imageElement = imageRef.current;
        const imageRect = imageElement.getBoundingClientRect();
        const containerRect = containerRef.current?.getBoundingClientRect();
        
        if (!imageRect || !containerRect) return [];

        const highlights = [];

        // Try to use real OCR bounding boxes first - check multiple possible field names
        const ocrBoxes = chunk.metadata?.chunk_bounding_boxes || 
                        chunk.metadata?.ocr_boxes || 
                        chunk.metadata?.bounding_boxes ||
                        chunk.metadata?.ocr_bounding_boxes;
                        
        if (ocrBoxes && Array.isArray(ocrBoxes)) {
            const imageWidth = chunk.metadata.image_width || imageElement.naturalWidth;
            const imageHeight = chunk.metadata.image_height || imageElement.naturalHeight;
            
            if (imageWidth && imageHeight) {
                // Calculate scale factors based on the displayed image size vs original image size
                const scaleX = imageRect.width / imageWidth;
                const scaleY = imageRect.height / imageHeight;
                
                ocrBoxes.forEach((box, boxIndex) => {
                    if (box.confidence >= 10) { // Lower confidence threshold to match backend
                        // Convert OCR coordinates to display coordinates
                        const left = box.left * scaleX;
                        const top = box.top * scaleY;
                        const width = box.width * scaleX;
                        const height = box.height * scaleY;
                        
                        highlights.push({
                            type: isDebugMode ? 'debugBox' : 'ocrBox',
                            left: left,
                            top: top,
                            width: width,
                            height: height,
                            text: box.text,
                            confidence: box.confidence,
                            chunkNumber: chunk.chunk_number,
                            style: isDebugMode ? {
                                borderColor: debugColor,
                                backgroundColor: debugColor + '20'
                            } : {}
                        });
                    }
                });
                
                if (highlights.length > 0) {
                    return highlights;
                }
            }
        }

        // Fallback to simulated highlighting
        return createSimulatedHighlight(chunk, debugColor, isDebugMode);
    };

    const createSimulatedHighlight = (chunk, debugColor = null, isDebugMode = false) => {
        if (!chunk || !imageRef.current) return [];

        const imageElement = imageRef.current;
        const imageRect = imageElement.getBoundingClientRect();
        
        const highlights = [];
        const chunkLength = chunk.content?.length || 0;
        const wordsCount = chunk.content?.split(/\s+/).length || 1;
        
        // Create multiple highlight areas based on content length
        const numAreas = Math.min(Math.max(Math.ceil(wordsCount / 10), 1), 8);
        
        for (let i = 0; i < numAreas; i++) {
            const baseY = 0.2 + (i * 0.1) + (chunk.chunk_number - 1) * 0.05;
            const baseX = 0.1 + (i * 0.15);
            
            const left = Math.min(baseX, 0.8) * (imageRect.width * scale);
            const top = Math.min(baseY, 0.9) * (imageRect.height * scale);
            const width = Math.min(0.3 + (chunkLength / 1000), 0.6) * (imageRect.width * scale);
            const height = Math.max(20, Math.min(30, chunkLength / 20)) * scale;
            
            highlights.push({
                type: isDebugMode ? 'debugBox' : (internalHighlightMode === 'full' ? 'textLine' : 'area'),
                left: left,
                top: top,
                width: width,
                height: height,
                text: chunk.content?.substring(i * 50, (i + 1) * 50) || '',
                chunkNumber: chunk.chunk_number,
                style: isDebugMode ? {
                    borderColor: debugColor,
                    backgroundColor: debugColor + '20'
                } : {}
            });
        }
        
        return highlights;
    };

    useEffect(() => {
        if (!selectedChunkId || showDebugBoxes) return;
        
        const selectedChunk = chunks.find(chunk => chunk.id === selectedChunkId);
        if (selectedChunk && imageRef.current) {
            const highlights = createChunkHighlight(selectedChunk);
            setHighlightAreas(highlights);
        }
    }, [selectedChunkId, chunks, scale, internalHighlightMode]);

    const zoomIn = () => setScale(prev => Math.min(prev * 1.2, 5));
    const zoomOut = () => setScale(prev => Math.max(prev / 1.2, 0.5));
    const resetZoom = () => setScale(1);

    // Use provided imageUrl or construct from documentId and token
    const finalImageUrl = useMemo(() => {
        if (imageUrl) {
            // If imageUrl is provided but doesn't have a token, add it
            if (token && !imageUrl.includes('token=')) {
                const separator = imageUrl.includes('?') ? '&' : '?';
                return `${imageUrl}${separator}token=${encodeURIComponent(token)}`;
            }
            return imageUrl;
        } else if (documentId && token) {
            return `/api/documents/${documentId}/image?token=${encodeURIComponent(token)}`;
        }
        return null;
    }, [imageUrl, documentId, token]);

    // Debug logging
    useEffect(() => {
        console.log('ImageViewer Debug Info:', {
            documentId,
            token: token ? `${token.substring(0, 10)}...` : 'undefined',
            imageUrl,
            finalImageUrl,
            chunksCount: chunks.length,
            selectedChunk: selectedChunkId ? chunks.find(c => c.id === selectedChunkId) : null,
            firstChunkMetadata: chunks[0]?.metadata
        });
        
        // Debug OCR data for each chunk
        chunks.forEach((chunk, index) => {
            if (chunk.metadata?.chunk_bounding_boxes) {
                console.log(`Chunk ${index + 1} OCR data:`, {
                    chunkId: chunk.id,
                    chunkNumber: chunk.chunk_number,
                    boundingBoxes: chunk.metadata.chunk_bounding_boxes,
                    imageWidth: chunk.metadata.image_width,
                    imageHeight: chunk.metadata.image_height,
                    content: chunk.content?.substring(0, 100) + '...'
                });
            }
        });
    }, [documentId, token, imageUrl, finalImageUrl, chunks.length, selectedChunkId]);

    const getHighlightClassName = (area) => {
        const baseClass = "absolute border-2 pointer-events-auto transition-all duration-200 hover:z-30";
        
        if (area.type === 'debugBox') {
            return `${baseClass} hover:opacity-90`;
        } else if (area.type === 'ocrBox') {
            return `${baseClass} border-blue-500 bg-blue-200 bg-opacity-30 hover:bg-opacity-50 hover:border-blue-600`;
        } else if (area.type === 'textLine') {
            return `${baseClass} border-green-500 bg-green-200 bg-opacity-30 hover:bg-opacity-50`;
        } else {
            return `${baseClass} border-yellow-500 bg-yellow-200 bg-opacity-30 hover:bg-opacity-50`;
        }
    };

    const getTooltipContent = (area) => {
        if (area.type === 'debugBox') {
            return `Chunk ${area.chunkNumber}: ${area.text.substring(0, 50)}${area.text.length > 50 ? '...' : ''}`;
        } else if (area.type === 'ocrBox') {
            return `"${area.text}" (${area.confidence}% confidence)`;
        } else {
            return `Chunk ${area.chunkNumber}: ${area.text.substring(0, 30)}${area.text.length > 30 ? '...' : ''}`;
        }
    };

    return (
        <div className="h-full flex flex-col bg-gray-50">
            {/* Header with zoom controls and highlight mode selector */}
            <div className="flex-shrink-0 bg-white border-b border-gray-200 p-3">
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                        <h3 className="text-lg font-medium text-gray-900">Document Preview</h3>
                    </div>
                    <div className="flex items-center space-x-2">
                        <button onClick={zoomOut} className="p-1 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded">
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                            </svg>
                        </button>
                        <span className="text-sm text-gray-600 min-w-[3rem] text-center">
                            {Math.round(scale * 100)}%
                        </span>
                        <button onClick={zoomIn} className="p-1 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded">
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                            </svg>
                        </button>
                        <button onClick={resetZoom} className="px-2 py-1 text-xs text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded">
                            Reset
                        </button>
                    </div>
                </div>
            </div>

            {/* Image container */}
            <div className="flex-1 relative overflow-auto bg-gray-100">
                <div 
                    ref={containerRef}
                    className="relative w-full h-full flex items-center justify-center p-4"
                    style={{ minHeight: '400px' }}
                >
                    {loading && (
                        <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75 z-10">
                            <div className="text-center">
                                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2"></div>
                                <p className="text-sm text-gray-600">Loading image...</p>
                            </div>
                        </div>
                    )}
                    
                    {error && (
                        <div className="text-center text-red-600 bg-red-50 p-4 rounded-lg border border-red-200">
                            <p className="font-medium">Failed to load image</p>
                            <p className="text-sm mt-1">{error}</p>
                        </div>
                    )}

                    {!finalImageUrl && !loading && !error && (
                        <div className="text-center text-gray-600 bg-gray-50 p-4 rounded-lg border border-gray-200">
                            <p className="font-medium">No image URL provided</p>
                            <p className="text-sm mt-1">Please provide either imageUrl or documentId with token</p>
                        </div>
                    )}

                    {finalImageUrl && (
                        <div className="relative inline-block">
                        <img
                            ref={imageRef}
                            src={finalImageUrl}
                            alt="Document preview"
                            className="max-w-none border border-gray-300 shadow-lg"
                            style={{
                                transform: `scale(${scale})`,
                                transformOrigin: 'center center'
                            }}
                            onLoad={() => setLoading(false)}
                            onError={(e) => {
                                console.error('Image load error:', e);
                                console.error('Failed to load:', finalImageUrl);
                                setError(`Could not load document image. URL: ${finalImageUrl}`);
                                setLoading(false);
                            }}
                        />
                        
                        {/* Highlight overlays */}
                        <div className="absolute inset-0 pointer-events-none">
                            <div 
                                className="relative w-full h-full"
                                style={{
                                    transform: `scale(${scale})`,
                                    transformOrigin: 'center center'
                                }}
                            >
                                {highlightAreas.map((area, index) => (
                                    <div
                                        key={index}
                                        className={getHighlightClassName(area)}
                                        style={{
                                            left: `${area.left}px`,
                                            top: `${area.top}px`,
                                            width: `${area.width}px`,
                                            height: `${area.height}px`,
                                            ...area.style
                                        }}
                                        title={getTooltipContent(area)}
                                    >
                                        {/* Chunk number badge */}
                                        {area.type === 'area' && area.chunkNumber && (
                                            <div className="absolute -top-1 -left-1 bg-yellow-600 text-white text-xs px-1 py-0.5 rounded shadow-sm">
                                                {area.chunkNumber}
                                            </div>
                                        )}
                                        {area.type === 'textLine' && area.chunkNumber && (
                                            <div className="absolute -top-1 -left-1 bg-blue-600 text-white text-xs px-1 py-0.5 rounded shadow-sm">
                                                {area.chunkNumber}
                                            </div>
                                        )}
                                        {area.type === 'ocrBox' && area.chunkNumber && (
                                            <div className="absolute -top-2 -left-1 bg-blue-600 text-white text-xs px-1 py-0.5 rounded shadow-sm z-20">
                                                {area.chunkNumber}
                                            </div>
                                        )}
                                        {area.type === 'debugBox' && (
                                            <div className="absolute -top-2 -left-1 text-white text-xs px-1 py-0.5 rounded shadow-sm z-20"
                                                 style={{backgroundColor: area.style.borderColor}}>
                                                C{area.chunkNumber}
                                            </div>
                                        )}
                                        
                                        {/* OCR text overlay for full mode */}
                                        {area.type === 'ocrBox' && internalHighlightMode === 'full' && (
                                            <div className="absolute inset-0 flex items-center justify-center">
                                                <span className="text-xs text-blue-800 bg-white bg-opacity-75 px-1 rounded max-w-full overflow-hidden text-ellipsis">
                                                    {area.text}
                                                </span>
                                            </div>
                                        )}
                                        
                                        {/* Debug mode text overlay */}
                                        {area.type === 'debugBox' && (
                                            <div className="absolute inset-0 flex items-center justify-center">
                                                <span className="text-xs font-semibold opacity-75 max-w-full overflow-hidden text-ellipsis"
                                                      style={{color: area.style.borderColor}}>
                                                    {area.text}
                                                </span>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Chunk Navigation Helper */}
            {(selectedChunkId || showDebugBoxes) && (
                <div className="p-2 bg-blue-50 border-t border-blue-200 text-sm">
                    <div className="flex items-center justify-between">
                        <span className="text-blue-700">
                            {showDebugBoxes ? (
                                <>
                                    🔍 Debug Mode: Showing all OCR bounding boxes
                                    {highlightAreas.length > 0 && (
                                        <span className="ml-2 text-xs">
                                            • {highlightAreas.length} total words across {chunks.length} chunks
                                        </span>
                                    )}
                                </>
                            ) : (
                                <>
                                    {highlightAreas.some(area => area.type === 'ocrBox') ? '🔍' : '📝'} 
                                    {highlightAreas.some(area => area.type === 'ocrBox') ? 'OCR' : 'Simulated'} Chunk {chunks.find(c => c.id === selectedChunkId)?.chunk_number} highlighted
                                    {chunks.length > 1 && ` (${chunks.length} total chunks)`}
                                    {highlightAreas.length > 0 && (
                                        <span className="ml-2 text-xs">
                                            • {highlightAreas.length} {highlightAreas.some(area => area.type === 'ocrBox') ? 'words' : 'areas'}
                                        </span>
                                    )}
                                </>
                            )}
                        </span>
                        <div className="text-xs text-blue-600">
                            {showDebugBoxes 
                                ? '💡 Each color represents a different chunk. Hover for details.'
                                : highlightAreas.some(area => area.type === 'ocrBox') 
                                    ? '💡 Hover over highlights to see OCR text and confidence'
                                    : '💡 Highlighted areas show approximate text positions'
                            }
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ImageViewer;
