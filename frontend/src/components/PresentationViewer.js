import React, { useState, useEffect } from 'react';

const PresentationViewer = ({
    document,
    documentPreview,
    selectedChunkId,
    chunks = [],
    onChunkHighlight,
    highlightMode = 'line',
    className = ''
}) => {
    const [currentSlide, setCurrentSlide] = useState(0);
    const [slideView, setSlideView] = useState('content'); // 'content' or 'overview'

    const slides = documentPreview?.slides || [];
    const totalSlides = documentPreview?.total_slides || slides.length;

    // Debug logging for presentation data
    useEffect(() => {
        console.log('DEBUG: PresentationViewer received data:');
        console.log('DEBUG: - Document:', document);
        console.log('DEBUG: - DocumentPreview:', documentPreview);
        console.log('DEBUG: - Has HTML:', documentPreview?.has_html);
        console.log('DEBUG: - Slides count:', slides.length);
        console.log('DEBUG: - Total slides:', totalSlides);
        
        if (slides.length > 0) {
            console.log('DEBUG: First slide structure:', Object.keys(slides[0]));
            console.log('DEBUG: First slide data:', slides[0]);
            
            const hasHtml = slides.some(slide => slide.html_content && slide.format === 'html');
            console.log('DEBUG: Has HTML slides:', hasHtml);
        }
    }, [document, documentPreview, slides]);

    // Find which slide contains the selected chunk
    useEffect(() => {
        if (selectedChunkId && slides.length > 0) {
            const selectedChunk = chunks.find(chunk => chunk.id === selectedChunkId);
            if (selectedChunk && selectedChunk.metadata?.slide_number) {
                const slideNumber = selectedChunk.metadata.slide_number - 1; // Convert to 0-based index
                if (slideNumber >= 0 && slideNumber < slides.length) {
                    setCurrentSlide(slideNumber);
                }
            }
        }
    }, [selectedChunkId, slides, chunks]);

    const handleSlideNavigation = (direction) => {
        if (direction === 'prev' && currentSlide > 0) {
            setCurrentSlide(currentSlide - 1);
        } else if (direction === 'next' && currentSlide < slides.length - 1) {
            setCurrentSlide(currentSlide + 1);
        }
    };

    const handleSlideClick = (slideIndex) => {
        setCurrentSlide(slideIndex);
        setSlideView('content');
    };

    const getSlideChunks = (slideNumber) => {
        return chunks.filter(chunk => 
            chunk.metadata?.slide_number === slideNumber + 1 || 
            chunk.content?.includes(`Slide ${slideNumber + 1}`)
        );
    };

    const handleChunkClick = (chunkId) => {
        if (onChunkHighlight) {
            onChunkHighlight(chunkId);
        }
    };

    const renderSlideContent = (slide, slideIndex) => {
        const slideChunks = getSlideChunks(slideIndex);
        const isCurrentSlide = slideIndex === currentSlide;
        const hasSelectedChunk = slideChunks.some(chunk => chunk.id === selectedChunkId);
        const hasImage = slide.image_data && slide.image_format;
        const hasHtml = slide.html_content && slide.format === 'html';
        
        // Force HTML rendering if html_content exists, regardless of format field
        const forceHtml = slide.html_content && slide.html_content.length > 0;

        // Debug logging for each slide render
        console.log(`DEBUG: Rendering slide ${slideIndex}:`);
        console.log(`DEBUG: - Has HTML: ${hasHtml}`);
        console.log(`DEBUG: - Force HTML: ${forceHtml}`);
        console.log(`DEBUG: - Has Image: ${hasImage}`);
        console.log(`DEBUG: - Format: ${slide.format}`);
        console.log(`DEBUG: - HTML content length: ${slide.html_content ? slide.html_content.length : 0}`);
        console.log(`DEBUG: - Text content length: ${slide.content ? slide.content.length : 0}`);
        console.log(`DEBUG: - Slide object keys:`, Object.keys(slide));

        return (
            <div 
                key={slideIndex}
                className={`${
                    slideView === 'overview' 
                        ? 'cursor-pointer hover:shadow-lg transition-shadow' 
                        : ''
                } ${hasSelectedChunk ? 'ring-2 ring-blue-500' : ''}`}
                onClick={() => slideView === 'overview' && handleSlideClick(slideIndex)}
            >
                <div className={`
                    bg-white rounded-lg border ${hasSelectedChunk ? 'border-blue-300' : 'border-gray-200'}
                    ${slideView === 'overview' ? 'p-4' : 'p-6'}
                `}>
                    {/* Slide header */}
                    <div className="flex items-center justify-between mb-4">
                        <h3 className={`font-semibold text-gray-900 ${
                            slideView === 'overview' ? 'text-sm' : 'text-lg'
                        }`}>
                            Slide {slide.slide_number}
                        </h3>
                        <div className="flex items-center space-x-2">
                            {(hasHtml || forceHtml) && (
                                <span className="inline-flex items-center px-2 py-1 text-xs font-medium rounded-md bg-purple-100 text-purple-700">
                                    HTML {forceHtml && !hasHtml ? '(Forced)' : ''}
                                </span>
                            )}
                            {hasImage && (
                                <span className="inline-flex items-center px-2 py-1 text-xs font-medium rounded-md bg-green-100 text-green-700">
                                    {slide.is_placeholder ? 'Placeholder' : 'Image'}
                                </span>
                            )}
                            {slideChunks.length > 0 && (
                                <span className="inline-flex items-center px-2 py-1 text-xs font-medium rounded-md bg-blue-100 text-blue-700">
                                    {slideChunks.length} chunk{slideChunks.length !== 1 ? 's' : ''}
                                </span>
                            )}
                            {hasSelectedChunk && (
                                <span className="inline-flex items-center px-2 py-1 text-xs font-medium rounded-md bg-green-100 text-green-700">
                                    Selected
                                </span>
                            )}
                        </div>
                    </div>

                    {/* Debug info indicator - temporary */}
                    <div className="mb-2 p-2 bg-yellow-100 border border-yellow-300 rounded text-xs">
                        Debug: hasHtml={hasHtml ? 'true' : 'false'}, 
                        forceHtml={forceHtml ? 'true' : 'false'},
                        format={slide.format || 'none'}, 
                        htmlLength={slide.html_content ? slide.html_content.length : 0}
                    </div>

                    {/* HTML slide content (preferred method) */}
                    {(hasHtml || forceHtml) && (
                        <div className={`
                            slide-html-container mb-4 
                            ${slideView === 'overview' ? 'transform scale-50 origin-top-left overflow-hidden h-32' : 'w-full'} 
                            rounded-lg bg-gray-50 flex justify-center items-center p-4
                        `}>
                            <div 
                                className="slide-html-content w-full"
                                dangerouslySetInnerHTML={{ __html: slide.html_content }}
                                style={{
                                    overflow: slideView === 'overview' ? 'hidden' : 'visible'
                                }}
                                onError={(e) => {
                                    console.error('DEBUG: Error rendering HTML content:', e);
                                }}
                            />
                        </div>
                    )}

                    {/* Debug: Show what would be rendered if no HTML */}
                    {!hasHtml && !forceHtml && (
                        <div className="mb-2 p-2 bg-red-100 border border-red-300 rounded text-xs">
                            No HTML content - showing fallback text
                        </div>
                    )}

                    {/* Slide image (fallback if no HTML) */}
                    {!hasHtml && !forceHtml && hasImage && (
                        <div className={`mb-4 ${slideView === 'overview' ? 'h-32' : 'h-96'} overflow-hidden rounded-lg bg-gray-50 flex items-center justify-center`}>
                            <img
                                src={`data:image/${slide.image_format};base64,${slide.image_data}`}
                                alt={`Slide ${slide.slide_number}`}
                                className="max-w-full max-h-full object-contain"
                                onError={(e) => {
                                    e.target.style.display = 'none';
                                    e.target.nextElementSibling.style.display = 'block';
                                }}
                            />
                            <div className="text-center text-gray-500 hidden">
                                <svg className="w-12 h-12 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                                <p className="text-sm">Image not available</p>
                            </div>
                        </div>
                    )}

                    {/* Slide content (text fallback) */}
                    {!hasHtml && !forceHtml && (slide.content || !hasImage) && (
                        <div className={`
                            text-gray-700 whitespace-pre-wrap
                            ${slideView === 'overview' ? 'text-xs line-clamp-4' : 'text-sm'}
                            ${hasImage ? 'mt-2 p-3 bg-gray-50 rounded-lg' : ''}
                        `}>
                            {slide.content || '[No content available]'}
                        </div>
                    )}

                    {/* Show chunks for this slide in content view */}
                    {slideView === 'content' && slideChunks.length > 0 && (
                        <div className="mt-4 pt-4 border-t border-gray-200">
                            <h4 className="text-sm font-medium text-gray-900 mb-2">
                                Chunks in this slide:
                            </h4>
                            <div className="space-y-2">
                                {slideChunks.map(chunk => (
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
                                        <div className="flex items-center justify-between mb-1">
                                            <span className="text-xs font-medium text-gray-500">
                                                Chunk {chunk.chunk_number}
                                            </span>
                                            <span className="text-xs text-gray-400">
                                                {chunk.content?.length || 0} chars
                                            </span>
                                        </div>
                                        <div className="text-sm text-gray-700 line-clamp-2">
                                            {chunk.content}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
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
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                        </svg>
                    </div>
                    <p className="text-gray-600">Loading presentation...</p>
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
                            {document?.filename || 'Presentation'}
                        </h3>
                        <span className="text-sm text-gray-500">
                            {totalSlides} slide{totalSlides !== 1 ? 's' : ''}
                        </span>
                        {documentPreview?.has_images && (
                            <span className="inline-flex items-center px-2 py-1 text-xs font-medium rounded-md bg-green-100 text-green-700">
                                {documentPreview.conversion_method === 'libreoffice' ? '🖼️ High Quality' : '📄 Preview'}
                            </span>
                        )}
                        {documentPreview?.has_images === false && (
                            <span className="inline-flex items-center px-2 py-1 text-xs font-medium rounded-md bg-yellow-100 text-yellow-700">
                                📝 Text Only
                            </span>
                        )}
                    </div>

                    <div className="flex items-center space-x-3">
                        {/* View toggle */}
                        <div className="flex rounded-lg border border-gray-300 overflow-hidden">
                            <button
                                onClick={() => setSlideView('content')}
                                className={`px-3 py-1 text-sm ${
                                    slideView === 'content'
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-white text-gray-700 hover:bg-gray-50'
                                }`}
                            >
                                Content
                            </button>
                            <button
                                onClick={() => setSlideView('overview')}
                                className={`px-3 py-1 text-sm ${
                                    slideView === 'overview'
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-white text-gray-700 hover:bg-gray-50'
                                }`}
                            >
                                Overview
                            </button>
                        </div>

                        {/* Navigation controls (only show in content view) */}
                        {slideView === 'content' && (
                            <div className="flex items-center space-x-2">
                                <button
                                    onClick={() => handleSlideNavigation('prev')}
                                    disabled={currentSlide === 0}
                                    className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                                    </svg>
                                </button>
                                
                                <span className="text-sm text-gray-600 min-w-[4rem] text-center">
                                    {currentSlide + 1} / {slides.length}
                                </span>
                                
                                <button
                                    onClick={() => handleSlideNavigation('next')}
                                    disabled={currentSlide === slides.length - 1}
                                    className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                    </svg>
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Main content area */}
            <div className="flex-1 overflow-auto bg-gray-50 p-4">
                {slideView === 'content' ? (
                    /* Single slide view */
                    <div className="max-w-4xl mx-auto">
                        {slides[currentSlide] && renderSlideContent(slides[currentSlide], currentSlide)}
                    </div>
                ) : (
                    /* Overview grid */
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                        {slides.map((slide, index) => renderSlideContent(slide, index))}
                    </div>
                )}
            </div>

            {/* Footer with slide indicator */}
            {slideView === 'content' && slides.length > 1 && (
                <div className="flex-shrink-0 bg-white border-t border-gray-200 p-3">
                    <div className="flex justify-center space-x-1">
                        {slides.map((_, index) => (
                            <button
                                key={index}
                                onClick={() => setCurrentSlide(index)}
                                className={`w-3 h-3 rounded-full transition-colors ${
                                    index === currentSlide 
                                        ? 'bg-blue-600' 
                                        : 'bg-gray-300 hover:bg-gray-400'
                                }`}
                                title={`Go to slide ${index + 1}`}
                            />
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default PresentationViewer;
