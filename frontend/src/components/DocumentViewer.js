import React, { useState, useEffect } from 'react';
import PDFViewer from './PDFViewer';
import ImageViewer from './ImageViewer';
import PresentationViewer from './PresentationViewer';
import SpreadsheetViewer from './SpreadsheetViewer';
import TextViewer from './TextViewer';
import EmailViewer from './EmailViewer';
import HTMLViewer from './HTMLViewer';

const DocumentViewer = ({
    document,
    documentPreview,
    selectedChunkId,
    chunks = [],
    onChunkHighlight,
    highlightMode = 'line',
    className = ''
}) => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Debug logging for DocumentViewer
    useEffect(() => {
        console.log('DEBUG: DocumentViewer received data:');
        console.log('DEBUG: - Document:', document);
        console.log('DEBUG: - DocumentPreview:', documentPreview);
        console.log('DEBUG: - DocumentPreview type:', documentPreview?.type);
        console.log('DEBUG: - Has HTML:', documentPreview?.has_html);
        console.log('DEBUG: - Slides:', documentPreview?.slides);
    }, [document, documentPreview]);

    // Determine document type and appropriate viewer
    const getDocumentType = () => {
        if (!document) return 'unsupported';

        // Check preview type first
        if (documentPreview?.type) {
            // Handle presentations with or without images
            if (documentPreview.type === 'presentation' || documentPreview.type === 'presentation_images') {
                return 'presentation';
            }
            return documentPreview.type;
        }

        // Fallback to content type and filename
        const contentType = document.content_type?.toLowerCase() || '';
        const filename = document.filename?.toLowerCase() || '';

        if (contentType === 'application/pdf' || filename.endsWith('.pdf')) {
            return 'pdf';
        }

        if (contentType.startsWith('image/') || 
            /\.(jpg|jpeg|png|gif|bmp|webp|svg|tif|tiff)$/i.test(filename)) {
            return 'image';
        }

        if (contentType.includes('presentation') || filename.match(/\.(pptx|ppt)$/i)) {
            return 'presentation';
        }

        if (contentType.includes('spreadsheet') || contentType.includes('excel') || 
            contentType === 'text/csv' || filename.match(/\.(xlsx|xls|csv)$/i)) {
            return 'spreadsheet';
        }

        if (contentType === 'message/rfc822' || filename.endsWith('.eml')) {
            return 'email';
        }

        if (contentType === 'text/html' || filename.match(/\.(html|htm)$/i)) {
            return 'html';
        }

        if (contentType.startsWith('text/') || 
            filename.match(/\.(txt|md|json|log)$/i)) {
            return 'text';
        }

        if (contentType.includes('word') || filename.match(/\.(docx|doc)$/i)) {
            return 'text'; // Word documents are handled as text in preview
        }

        return 'unsupported';
    };

    const documentType = getDocumentType();
    
    // Debug document type detection
    useEffect(() => {
        console.log('DEBUG: DocumentViewer document type detection:');
        console.log('DEBUG: - Detected type:', documentType);
        console.log('DEBUG: - Document filename:', document?.filename);
        console.log('DEBUG: - Document content_type:', document?.content_type);
        console.log('DEBUG: - Preview type:', documentPreview?.type);
    }, [documentType, document, documentPreview]);

    const renderViewer = () => {
        const commonProps = {
            selectedChunkId,
            chunks,
            onChunkHighlight,
            highlightMode,
            className: "h-full"
        };

        switch (documentType) {
            case 'pdf':
                return (
                    <PDFViewer
                        pdfUrl={documentPreview?.pdf_url || `/api/documents/${document.id}/raw`}
                        documentPreview={documentPreview}
                        {...commonProps}
                    />
                );

            case 'image':
                return (
                    <ImageViewer
                        documentId={document.id}
                        imageUrl={documentPreview?.image_url || `/api/documents/${document.id}/image`}
                        token={localStorage.getItem('token')}
                        {...commonProps}
                    />
                );

            case 'presentation':
                console.log('DEBUG: DocumentViewer rendering PresentationViewer');
                console.log('DEBUG: - Document:', document);
                console.log('DEBUG: - DocumentPreview:', documentPreview);
                return (
                    <PresentationViewer
                        document={document}
                        documentPreview={documentPreview}
                        {...commonProps}
                    />
                );

            case 'spreadsheet':
            case 'csv':
                return (
                    <SpreadsheetViewer
                        document={document}
                        documentPreview={documentPreview}
                        {...commonProps}
                    />
                );

            case 'email':
                return (
                    <EmailViewer
                        document={document}
                        documentPreview={documentPreview}
                        {...commonProps}
                    />
                );

            case 'html':
                return (
                    <HTMLViewer
                        document={document}
                        documentPreview={documentPreview}
                        {...commonProps}
                    />
                );

            case 'text':
                return (
                    <TextViewer
                        document={document}
                        documentPreview={documentPreview}
                        {...commonProps}
                    />
                );

            case 'unsupported':
            default:
                return (
                    <div className="h-full flex items-center justify-center p-8">
                        <div className="text-center">
                            <div className="text-gray-400 mb-4">
                                <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                            </div>
                            <p className="text-gray-600 font-medium">Viewer not available</p>
                            <p className="text-sm text-gray-500 mt-1">
                                {document?.filename && `Format: ${document.filename.split('.').pop()?.toUpperCase()}`}
                            </p>
                            <p className="text-sm text-gray-500">
                                {document?.content_type && `Type: ${document.content_type}`}
                            </p>
                            <div className="mt-4">
                                <a
                                    href={`/api/documents/${document?.id}/raw`}
                                    download={document?.filename}
                                    className="inline-flex items-center px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors"
                                >
                                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                    </svg>
                                    Download File
                                </a>
                            </div>
                        </div>
                    </div>
                );
        }
    };

    return (
        <div className={`relative ${className}`}>
            {/* Debug overlay */}
            <div className="absolute top-2 right-2 z-50 bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">
                Type: {documentType}
            </div>
            
            {loading && (
                <div className="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center z-10">
                    <div className="flex items-center space-x-3">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                        <span className="text-gray-600">Loading viewer...</span>
                    </div>
                </div>
            )}

            {error && (
                <div className="absolute inset-0 bg-red-50 flex items-center justify-center z-10">
                    <div className="text-center">
                        <div className="text-red-500 mb-2">
                            <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                        </div>
                        <p className="text-red-600 font-medium">Viewer Error</p>
                        <p className="text-sm text-red-500 mt-1">{error}</p>
                    </div>
                </div>
            )}

            {renderViewer()}
        </div>
    );
};

export default DocumentViewer;
