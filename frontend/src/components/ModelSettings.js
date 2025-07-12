import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

const ModelSettings = ({ isOpen, onClose, onSave }) => {
    const [availableModels, setAvailableModels] = useState([]);
    const [selectedModels, setSelectedModels] = useState({
        llm: 'deepseek-r1:latest',
        embedding: 'mxbai-embed-large'
    });
    const [selectedVariants, setSelectedVariants] = useState({});
    const [isLoading, setIsLoading] = useState(false);
    const [isLoadingModels, setIsLoadingModels] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [error, setError] = useState('');
    
    // Vector Store and Ingestion Management State
    const [showVectorStore, setShowVectorStore] = useState(false);
    const [ingestionStatus, setIngestionStatus] = useState(null);
    const [isLoadingIngestion, setIsLoadingIngestion] = useState(false);
    const [isReingesting, setIsReingesting] = useState(false);
    const [vectorStoreStats, setVectorStoreStats] = useState(null);
    const [selectedDocuments, setSelectedDocuments] = useState(new Set());
    const [gpuInfo, setGpuInfo] = useState(null);
    const [isLoadingGpuInfo, setIsLoadingGpuInfo] = useState(false);
    


    // Load available models when dialog opens
    useEffect(() => {
        if (isOpen) {
            loadAvailableModels();
            loadCurrentSettings();
            loadIngestionStatus();
            loadGpuInfo();
        }
    }, [isOpen]);

    const loadGpuInfo = async () => {
        setIsLoadingGpuInfo(true);
        try {
            const response = await api.get('/api/gpu/memory-info');
            if (response && response.success) {
                setGpuInfo(response.gpu_memory);
            }
        } catch (err) {
            console.error('Error loading GPU info:', err);
            // Set default values if GPU info is not available
            setGpuInfo({
                total: 8192,
                used: 2048,
                free: 6144,
                available: 6144
            });
        } finally {
            setIsLoadingGpuInfo(false);
        }
    };

    const loadAvailableModels = async () => {
        setIsLoadingModels(true);
        setError('');
        
        try {
            console.log('Loading available models from Ollama...');
            const response = await api.get('/api/models/available');
            console.log('Available models response:', response);
            
            if (response && response.models) {
                setAvailableModels(response.models);
            } else if (response && (response.llm_models || response.embedding_models)) {
                // Handle legacy response format
                const allModels = [
                    ...(response.llm_models || []).map(m => ({...m, category: 'llm'})),
                    ...(response.embedding_models || []).map(m => ({...m, category: 'embedding'}))
                ];
                setAvailableModels(allModels);
            } else {
                setError('Failed to load available models');
            }
        } catch (err) {
            console.error('Error loading models:', err);
            setError('Failed to connect to model server');
        } finally {
            setIsLoadingModels(false);
        }
    };

    const loadCurrentSettings = async () => {
        try {
            console.log('Loading current model settings...');
            const response = await api.get('/api/models/current');
            console.log('Current settings response:', response);
            
            if (response) {
                const llmModel = response.llm || 'deepseek-r1:latest';
                const embeddingModel = response.embedding || 'mxbai-embed-large';
                
                // Extract base names and variants
                const llmParts = llmModel.split(':');
                const embeddingParts = embeddingModel.split(':');
                
                const llmBase = llmParts[0];
                const llmVariant = llmParts[1] || 'latest';
                
                const embeddingBase = embeddingParts[0];
                const embeddingVariant = embeddingParts[1] || 'latest';
                
                setSelectedModels({
                    llm: llmModel,
                    embedding: embeddingModel
                });
                
                // Set the variant selections
                setSelectedVariants({
                    [llmBase]: llmVariant,
                    [embeddingBase]: embeddingVariant
                });
            }
        } catch (err) {
            console.error('Error loading current settings:', err);
            // Use defaults if loading fails
        }
    };

    const loadIngestionStatus = async () => {
        setIsLoadingIngestion(true);
        try {
            const response = await api.get('/api/admin/ingestion-status');
            setIngestionStatus(response);
            
            // Also load vector store stats
            const statsResponse = await api.get('/api/vector-store/stats');
            setVectorStoreStats(statsResponse.stats);
        } catch (err) {
            console.error('Error loading ingestion status:', err);
        } finally {
            setIsLoadingIngestion(false);
        }
    };



    const handleReingestionAll = async () => {
        setIsReingesting(true);
        setError('');
        
        try {
            const response = await api.post('/api/admin/reingest-all');
            if (response && response.success) {
                // Reload ingestion status
                await loadIngestionStatus();
            } else {
                setError('Failed to trigger re-ingestion');
            }
        } catch (err) {
            console.error('Error triggering re-ingestion:', err);
            setError('Failed to trigger re-ingestion');
        } finally {
            setIsReingesting(false);
        }
    };

    const handleCleanupOrphanedDocuments = async () => {
        setError('');
        
        try {
            const response = await api.post('/api/admin/cleanup-orphaned-documents');
            if (response && response.success) {
                // Reload ingestion status
                await loadIngestionStatus();
                setError(`✓ ${response.message}`);
                // Clear success message after 3 seconds
                setTimeout(() => setError(''), 3000);
            } else {
                setError('Failed to cleanup orphaned documents');
            }
        } catch (err) {
            console.error('Error cleaning up orphaned documents:', err);
            setError('Failed to cleanup orphaned documents');
        }
    };

    const handleDocumentSelection = (docId, isSelected) => {
        const newSelection = new Set(selectedDocuments);
        if (isSelected) {
            newSelection.add(docId);
        } else {
            newSelection.delete(docId);
        }
        setSelectedDocuments(newSelection);
    };

    const handleSelectAllDocuments = (documents, isSelectAll) => {
        if (isSelectAll) {
            setSelectedDocuments(new Set(documents.map(doc => doc.id)));
        } else {
            setSelectedDocuments(new Set());
        }
    };

    const handleModelChange = (modelType, modelName) => {
        // Set the base model name (without variant)
        setSelectedModels(prev => ({
            ...prev,
            [modelType]: modelName
        }));
        
        // Initialize variant selection to 'latest' for new model selection
        setSelectedVariants(prev => ({
            ...prev,
            [modelName]: 'latest'
        }));
    };

    const handleVariantChange = (modelName, variant) => {
        setSelectedVariants(prev => ({
            ...prev,
            [modelName]: variant
        }));
        
        // Update the selected model with the variant (only if not 'latest')
        const fullModelName = variant === 'latest' ? modelName : `${modelName}:${variant}`;
        const modelType = Object.keys(selectedModels).find(type => 
            selectedModels[type] === modelName || selectedModels[type].startsWith(`${modelName}:`)
        );
        
        if (modelType) {
            setSelectedModels(prev => ({
                ...prev,
                [modelType]: fullModelName
            }));
        }
    };

    const handleSave = async () => {
        setIsSaving(true);
        setError('');
        
        try {
            console.log('Checking GPU compatibility...');
            
            // First check GPU compatibility
            const selectedLlmModel = filterModelsByType('llm').find(model => 
                selectedModels.llm === model.name || selectedModels.llm.startsWith(`${model.name}:`)
            );
            const selectedEmbeddingModel = filterModelsByType('embedding').find(model => 
                selectedModels.embedding === model.name || selectedModels.embedding.startsWith(`${model.name}:`)
            );
            
            // Get selected variants for size estimation
            const llmVariant = selectedVariants[selectedLlmModel?.name] || 'latest';
            const embeddingVariant = selectedVariants[selectedEmbeddingModel?.name] || 'latest';
            
            // Prepare compatibility check payload
            const compatibilityPayload = {
                llm: selectedModels.llm,
                embedding: selectedModels.embedding,
                llm_size: getModelSizeInGB(selectedLlmModel, llmVariant),
                embedding_size: getModelSizeInGB(selectedEmbeddingModel, embeddingVariant)
            };
            
            console.log('Compatibility check payload:', compatibilityPayload);
            
            try {
                const compatibilityResponse = await api.post('/api/models/check-compatibility', compatibilityPayload);
                console.log('Compatibility response:', compatibilityResponse);
                
                if (!compatibilityResponse.compatible) {
                    // Show detailed error with recommendations
                    const errorMsg = `GPU Memory Insufficient:\n\n` +
                        `${compatibilityResponse.llm_check.message}\n` +
                        `${compatibilityResponse.embedding_check.message}\n\n` +
                        `${compatibilityResponse.combined_check.message}\n\n` +
                        `Recommendations:\n${compatibilityResponse.recommendations.join('\n')}`;
                    
                    setError(errorMsg);
                    return;
                }
            } catch (compatError) {
                console.warn('Could not check GPU compatibility:', compatError);
                // Continue with saving but warn the user
                const continueAnyway = window.confirm(
                    'Could not verify GPU compatibility. Continue anyway?\n\n' +
                    'Warning: Large models may cause out-of-memory errors.'
                );
                
                if (!continueAnyway) {
                    return;
                }
            }
            
            console.log('Saving model settings:', selectedModels);
            const response = await api.post('/api/models/settings', {
                ...selectedModels,
                llm_size: compatibilityPayload.llm_size,
                embedding_size: compatibilityPayload.embedding_size
            });
            console.log('Save response:', response);
            
            if (response && response.success) {
                // Reload ingestion status if embedding model changed
                if (response.embedding_changed) {
                    await loadIngestionStatus();
                }
                onSave?.(selectedModels);
                onClose();
            } else {
                setError('Failed to save model settings');
            }
        } catch (err) {
            console.error('Error saving settings:', err);
            
            // Check if it's a GPU memory error
            if (err.response?.data?.detail?.error === 'GPU_MEMORY_INSUFFICIENT') {
                const errorData = err.response.data.detail;
                const errorMsg = `${errorData.message}\n\n` +
                    `LLM: ${errorData.llm_check.message}\n` +
                    `Embedding: ${errorData.embedding_check.message}\n\n` +
                    `Combined: ${errorData.combined_check.message}\n\n` +
                    `Recommendations:\n${errorData.recommendations.join('\n')}\n\n` +
                    `Available GPU Memory: ${errorData.combined_check.available_mb}MB\n` +
                    `Required Memory: ${errorData.combined_check.required_mb}MB\n` +
                    `Shortage: ${errorData.combined_check.shortage_mb}MB`;
                
                setError(errorMsg);
            } else {
                setError(`Failed to save model settings: ${err.response?.data?.detail || err.message}`);
            }
        } finally {
            setIsSaving(false);
        }
    };

    const normalizeVariant = (variant) => {
        // Normalize variant names to handle both 7b and 7B
        if (!variant) return variant;
        
        // Convert to lowercase and handle common patterns
        const normalized = variant.toLowerCase();
        
        // Handle billion parameters (b/B)
        if (normalized.match(/^\d+\.?\d*b$/)) {
            return normalized;
        }
        
        // Handle million parameters (m/M)
        if (normalized.match(/^\d+\.?\d*m$/)) {
            return normalized;
        }
        
        // Handle other patterns like x7b, e2b, etc.
        return normalized;
    };

    const filterModelsByType = (type) => {
        if (!availableModels || !Array.isArray(availableModels)) return [];
        
        // Use category-based filtering if available, fallback to name-based
        const categoryFiltered = availableModels.filter(model => {
            // If model has category, use it strictly
            if (model.category) {
                return model.category === type;
            }
            
            // Fallback to name-based filtering for backward compatibility
            const name = model.name.toLowerCase();
            
            // First, check if it's an embedding model (exclude from LLM)
            const isEmbeddingModel = name.includes('embed') || 
                                   name.includes('bge') ||
                                   name.includes('minilm') ||
                                   name.includes('all-minilm') ||
                                   name.includes('nomic') ||
                                   name.includes('e5-') ||
                                   name.includes('sentence') ||
                                   name.includes('text-embedding') ||
                                   name.includes('instructor') ||
                                   name.includes('gte-') ||
                                   name.includes('multilingual-e5') ||
                                   name.includes('arctic-embed');
            
            switch (type) {
                case 'llm':
                    // Exclude embedding and reranking models, include everything else
                    return !isEmbeddingModel && !name.includes('rerank');
                case 'embedding':
                    return isEmbeddingModel;
                default:
                    return true;
            }
        });
        
        // Group models by base name to handle variants
        const groupedModels = {};
        
        categoryFiltered.forEach(model => {
            // Extract base model name (remove variant tag)
            const baseName = model.name.split(':')[0];
            
            if (!groupedModels[baseName]) {
                groupedModels[baseName] = {
                    ...model,
                    name: baseName,
                    variants: []
                };
            }
            
            // If this is a variant (has colon), add to variants
            if (model.name.includes(':')) {
                const variant = model.name.split(':')[1];
                const normalizedVariant = normalizeVariant(variant);
                
                // Skip 'latest' variants as they're the default
                if (normalizedVariant !== 'latest') {
                    // Check if this normalized variant already exists
                    const existingVariant = groupedModels[baseName].variants.find(v => 
                        normalizeVariant(v.tag) === normalizedVariant
                    );
                    
                    if (!existingVariant) {
                        groupedModels[baseName].variants.push({
                            tag: normalizedVariant,
                            size: model.size,
                            fullName: model.name,
                            originalTag: variant
                        });
                    }
                }
            }
        });
        
        // Sort variants by size/name
        Object.values(groupedModels).forEach(model => {
            if (model.variants) {
                model.variants.sort((a, b) => {
                    // Extract numeric value for sorting
                    const getNumericValue = (tag) => {
                        const match = tag.match(/(\d+\.?\d*)/);
                        return match ? parseFloat(match[1]) : 0;
                    };
                    
                    const aNum = getNumericValue(a.tag);
                    const bNum = getNumericValue(b.tag);
                    
                    return aNum - bNum;
                });
            }
        });
        
        return Object.values(groupedModels);
    };

    const formatModelSize = (size) => {
        if (!size || size === 'Unknown') return '';
        if (typeof size === 'string') {
            // Handle string sizes like "7b", "13B", "1.5b", etc.
            const lowerSize = size.toLowerCase();
            if (lowerSize.includes('b') && !lowerSize.includes('gb')) {
                // Convert billion parameters to approximate GB
                const match = lowerSize.match(/(\d+\.?\d*)/);
                if (match) {
                    const params = parseFloat(match[1]);
                    // Rough approximation: 1B parameters ≈ 2GB (for FP16)
                    const gbSize = (params * 2).toFixed(1);
                    return `${gbSize}GB`;
                }
            }
            return size;
        }
        // Handle numeric sizes (bytes)
        if (size < 1024 * 1024 * 1024) {
            return `${Math.round(size / (1024 * 1024))} MB`;
        }
        return `${(size / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    };

    const getModelSizeInGB = (model, variant = null) => {
        // If variant is provided, try to get size from variant
        if (variant && model.variants) {
            const variantObj = model.variants.find(v => v.tag === variant);
            if (variantObj && variantObj.size && variantObj.size !== 'Unknown') {
                return formatModelSize(variantObj.size);
            }
        }
        
        // Fallback to model size or estimate from variant name
        if (model.size && model.size !== 'Unknown') {
            return formatModelSize(model.size);
        }
        
        // If no size info, try to estimate from variant name
        if (variant && variant !== 'latest') {
            return formatModelSize(variant);
        }
        
        return '';
    };

    const formatFileSize = (bytes) => {
        if (!bytes) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg shadow-xl max-w-6xl w-full mx-4 max-h-[95vh] overflow-hidden relative">
                {/* Loading Overlay for Models */}
                {isLoadingModels && (
                    <div className="absolute inset-0 bg-white bg-opacity-95 flex items-center justify-center z-10 rounded-lg">
                        <div className="text-center">
                            <div className="relative">
                                {/* Animated circles */}
                                <div className="w-20 h-20 relative mx-auto mb-4">
                                    <div className="absolute inset-0 border-4 border-blue-200 rounded-full"></div>
                                    <div className="absolute inset-0 border-4 border-blue-600 rounded-full border-t-transparent animate-spin"></div>
                                    <div className="absolute inset-2 border-2 border-blue-400 rounded-full border-r-transparent animate-spin" style={{animationDirection: 'reverse', animationDuration: '1.5s'}}></div>
                                </div>
                            </div>
                            <div className="text-lg font-semibold text-gray-800 mb-2">
                                Loading Models...
                            </div>
                            <div className="text-sm text-gray-600 mb-4">
                                Fetching available models from Ollama
                            </div>
                            {/* Progress dots animation */}
                            <div className="flex justify-center space-x-1">
                                <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></div>
                                <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{animationDelay: '0.1s'}}></div>
                                <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                                <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{animationDelay: '0.3s'}}></div>
                                <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Header */}
                <div className="flex justify-between items-center p-6 border-b border-gray-200">                <div className="flex items-center gap-4">
                    <h2 className="text-xl font-semibold text-gray-900">Model Settings</h2>
                    
                    {/* GPU Memory Indicator */}
                    {gpuInfo && (
                        <div className="flex items-center gap-2 px-3 py-1 bg-gray-100 rounded-lg text-sm">
                            <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                            </svg>
                            <span className="text-gray-700">
                                GPU: {((gpuInfo.available / 1024).toFixed(1))}GB / {((gpuInfo.total / 1024).toFixed(1))}GB available
                            </span>
                            <div className="w-16 h-2 bg-gray-300 rounded-full overflow-hidden">
                                <div 
                                    className={`h-full rounded-full transition-all duration-300 ${
                                        (gpuInfo.used / gpuInfo.total) > 0.8 ? 'bg-red-500' :
                                        (gpuInfo.used / gpuInfo.total) > 0.6 ? 'bg-yellow-500' : 'bg-green-500'
                                    }`}
                                    style={{ width: `${(gpuInfo.used / gpuInfo.total) * 100}%` }}
                                ></div>
                            </div>
                        </div>
                    )}
                    
                    <button
                        onClick={loadAvailableModels}
                        disabled={isLoadingModels || isSaving}
                        className="text-blue-600 hover:text-blue-800 disabled:text-gray-400 transition-colors flex items-center gap-1 text-sm"
                        title="Refresh models"
                    >
                        <svg className={`w-4 h-4 ${isLoadingModels ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                        Refresh
                    </button>
                        
                        {/* Vector Store Toggle */}
                        <button
                            onClick={() => setShowVectorStore(!showVectorStore)}
                            className="text-purple-600 hover:text-purple-800 disabled:text-gray-400 transition-colors flex items-center gap-1 text-sm"
                            title="Toggle vector store management"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                            </svg>
                            {showVectorStore ? 'Hide' : 'Show'} Vector Store
                        </button>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-gray-400 hover:text-gray-600 transition-colors"
                        disabled={isLoadingModels || isSaving}
                    >
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {/* Content */}
                <div className="flex">
                    {/* Left Panel - Model Settings */}
                    <div className={`${showVectorStore ? 'w-1/2' : 'w-full'} p-6 overflow-y-auto max-h-[70vh] border-r border-gray-200`}>
                        {error && (
                            <div className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
                                <div className="flex items-start">
                                    <svg className="w-5 h-5 text-red-500 mr-2 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                                    </svg>
                                    <div className="flex-1">
                                        <div className="font-medium mb-1">Model Configuration Error</div>
                                        <div className="text-sm whitespace-pre-line">{error}</div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {!isLoadingModels && availableModels.length > 0 && (
                            <div className="mb-4 p-3 bg-green-50 border border-green-200 text-green-700 rounded text-sm">
                                ✓ Successfully loaded {availableModels.length} models from Ollama
                            </div>
                        )}

                        {isSaving && (
                            <div className="mb-4 p-3 bg-blue-50 border border-blue-200 text-blue-700 rounded text-sm">
                                <div className="flex items-center">
                                    <svg className="animate-spin h-4 w-4 mr-2 text-blue-600" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                    </svg>
                                    Downloading models and updating settings... This may take a few minutes for large models.
                                </div>
                            </div>
                        )}

                        {!isLoadingModels && (
                            <div className="space-y-6">
                                {/* LLM Model Selection */}
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Large Language Model (LLM)
                                    </label>
                                    <p className="text-xs text-gray-500 mb-3">
                                        The main AI model used for generating responses and reasoning.
                                    </p>
                                    <div className="border border-gray-300 rounded-lg max-h-60 overflow-y-auto">
                                        {filterModelsByType('llm').map((model, index) => {
                                            // Check if this base model is selected (with or without variant)
                                            const baseModelSelected = selectedModels.llm === model.name || selectedModels.llm.startsWith(`${model.name}:`);
                                            const selectedVariant = selectedVariants[model.name] || 'latest';
                                            
                                            return (
                                                <div
                                                    key={`llm-${model.name}-${index}`}
                                                    className={`p-3 cursor-pointer hover:bg-gray-50 ${
                                                        index > 0 ? 'border-t border-gray-200' : ''
                                                    }`}
                                                >
                                                    <label className="flex items-start cursor-pointer">
                                                        <input
                                                            type="radio"
                                                            name="llm"
                                                            value={model.name}
                                                            checked={baseModelSelected}
                                                            onChange={(e) => handleModelChange('llm', e.target.value)}
                                                            className="mr-3 mt-1 text-blue-600"
                                                        />
                                                        <div className="flex-1">
                                                            <div className="flex items-center gap-2 mb-2">
                                                                <div className="font-medium text-gray-900">
                                                                    {model.name}
                                                                    {(() => {
                                                                        const sizeGB = getModelSizeInGB(model, baseModelSelected ? selectedVariant : 'latest');
                                                                        return sizeGB ? ` (${sizeGB})` : '';
                                                                    })()}
                                                                </div>
                                                                {model.source && (
                                                                    <span className={`px-2 py-1 text-xs rounded-full ${
                                                                        model.source === 'local' ? 'bg-green-100 text-green-800' :
                                                                        model.source === 'library' ? 'bg-blue-100 text-blue-800' :
                                                                        'bg-gray-100 text-gray-800'
                                                                    }`}>
                                                                        {model.source}
                                                                    </span>
                                                                )}
                                                            </div>
                                                            
                                                            {/* Variant Selection - only show if model is selected and has variants */}
                                                            {baseModelSelected && model.variants && model.variants.length > 0 && (
                                                                <div className="mb-2">
                                                                    <label className="block text-xs font-medium text-gray-600 mb-1">
                                                                        Select Variant:
                                                                    </label>
                                                                    <select
                                                                        value={selectedVariant}
                                                                        onChange={(e) => handleVariantChange(model.name, e.target.value)}
                                                                        className="text-sm border border-gray-300 rounded px-2 py-1 bg-white w-full max-w-xs"
                                                                    >
                                                                        <option value="latest">
                                                                            latest (default){(() => {
                                                                                const latestSize = getModelSizeInGB(model, 'latest');
                                                                                return latestSize ? ` - ${latestSize}` : '';
                                                                            })()}
                                                                        </option>
                                                                        {model.variants.map((variant, vIndex) => (
                                                                            <option key={vIndex} value={variant.tag}>
                                                                                {variant.tag}{(() => {
                                                                                    const variantSize = getModelSizeInGB(model, variant.tag);
                                                                                    return variantSize ? ` (${variantSize})` : '';
                                                                                })()}
                                                                            </option>
                                                                        ))}
                                                                    </select>
                                                                </div>
                                                            )}
                                                            
                                                            {model.description && (
                                                                <div className="text-sm text-gray-600 mb-1">
                                                                    {model.description}
                                                                </div>
                                                            )}
                                                        </div>
                                                    </label>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>

                                {/* Embedding Model Selection */}
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-2">
                                        Embedding Model
                                    </label>
                                    <p className="text-xs text-gray-500 mb-3">
                                        Used to convert text into numerical vectors for similarity search and retrieval.
                                    </p>
                                    <div className="border border-gray-300 rounded-lg max-h-40 overflow-y-auto">
                                        {filterModelsByType('embedding').map((model, index) => {
                                            // Check if this base model is selected (with or without variant)
                                            const baseModelSelected = selectedModels.embedding === model.name || selectedModels.embedding.startsWith(`${model.name}:`);
                                            const selectedVariant = selectedVariants[model.name] || 'latest';
                                            
                                            return (
                                                <div
                                                    key={`embedding-${model.name}-${index}`}
                                                    className={`p-3 cursor-pointer hover:bg-gray-50 ${
                                                        index > 0 ? 'border-t border-gray-200' : ''
                                                    }`}
                                                >
                                                    <label className="flex items-start cursor-pointer">
                                                        <input
                                                            type="radio"
                                                            name="embedding"
                                                            value={model.name}
                                                            checked={baseModelSelected}
                                                            onChange={(e) => handleModelChange('embedding', e.target.value)}
                                                            className="mr-3 mt-1 text-blue-600"
                                                        />
                                                        <div className="flex-1">
                                                            <div className="font-medium text-gray-900 mb-2">
                                                                {model.name}
                                                                {(() => {
                                                                    const sizeGB = getModelSizeInGB(model, baseModelSelected ? selectedVariant : 'latest');
                                                                    return sizeGB ? ` (${sizeGB})` : '';
                                                                })()}
                                                            </div>
                                                            
                                                            {/* Variant Selection - only show if model is selected and has variants */}
                                                            {baseModelSelected && model.variants && model.variants.length > 0 && (
                                                                <div className="mb-2">
                                                                    <label className="block text-xs font-medium text-gray-600 mb-1">
                                                                        Select Variant:
                                                                    </label>
                                                                    <select
                                                                        value={selectedVariant}
                                                                        onChange={(e) => handleVariantChange(model.name, e.target.value)}
                                                                        className="text-sm border border-gray-300 rounded px-2 py-1 bg-white w-full max-w-xs"
                                                                    >
                                                                        <option value="latest">
                                                                            latest (default){(() => {
                                                                                const latestSize = getModelSizeInGB(model, 'latest');
                                                                                return latestSize ? ` - ${latestSize}` : '';
                                                                            })()}
                                                                        </option>
                                                                        {model.variants.map((variant, vIndex) => (
                                                                            <option key={vIndex} value={variant.tag}>
                                                                                {variant.tag}{(() => {
                                                                                    const variantSize = getModelSizeInGB(model, variant.tag);
                                                                                    return variantSize ? ` (${variantSize})` : '';
                                                                                })()}
                                                                            </option>
                                                                        ))}
                                                                    </select>
                                                                </div>
                                                            )}
                                                        </div>
                                                    </label>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Right Panel - Vector Store Management */}
                    {showVectorStore && (
                        <div className="w-1/2 p-6 overflow-y-auto max-h-[70vh] bg-gray-50">
                            <div className="mb-4">
                                <h3 className="text-lg font-semibold text-gray-900 mb-2">Vector Store & Document Ingestion</h3>
                                <p className="text-sm text-gray-600">
                                    Manage your knowledge base and document ingestion for the current embedding model.
                                </p>
                            </div>

                            {isLoadingIngestion ? (
                                <div className="flex items-center justify-center py-8">
                                    <div className="text-center">
                                        <svg className="animate-spin h-8 w-8 text-blue-600 mx-auto mb-2" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                        </svg>
                                        <div className="text-sm text-gray-600">Loading ingestion status...</div>
                                    </div>
                                </div>
                            ) : (
                                <div className="space-y-4">
                                    {/* Current Status Summary */}
                                    {ingestionStatus && (
                                        <div className="bg-white p-4 rounded-lg border border-gray-200">
                                            <h4 className="font-medium text-gray-900 mb-2">Current Status</h4>
                                            <div className="grid grid-cols-2 gap-4 text-sm">
                                                <div>
                                                    <span className="text-gray-600">Embedding Model:</span>
                                                    <div className="font-medium text-blue-600">{ingestionStatus.current_embedding_model}</div>
                                                </div>
                                                <div>
                                                    <span className="text-gray-600">Total Documents:</span>
                                                    <div className="font-medium">{ingestionStatus.total_documents}</div>
                                                </div>
                                                <div>
                                                    <span className="text-gray-600">Ingested:</span>
                                                    <div className="font-medium text-green-600">{ingestionStatus.ingested_documents.count}</div>
                                                </div>
                                                <div>
                                                    <span className="text-gray-600">Pending:</span>
                                                    <div className="font-medium text-orange-600">{ingestionStatus.pending_documents.count}</div>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {/* Vector Store Stats */}
                                    {vectorStoreStats && (
                                        <div className="bg-white p-4 rounded-lg border border-gray-200">
                                            <h4 className="font-medium text-gray-900 mb-2">Vector Store Statistics</h4>
                                            <div className="text-sm space-y-2">
                                                <div className="flex justify-between">
                                                    <span className="text-gray-600">Total Collections:</span>
                                                    <span className="font-medium">{vectorStoreStats.total_collections}</span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-gray-600">Current Collection:</span>
                                                    <span className="font-medium text-blue-600">{vectorStoreStats.current_collection}</span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-gray-600">Vector Documents:</span>
                                                    <span className="font-medium">{vectorStoreStats.total_documents}</span>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {/* Actions */}
                                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                                        <h4 className="font-medium text-gray-900 mb-3">Actions</h4>
                                        <div className="space-y-3">
                                            <button
                                                onClick={handleReingestionAll}
                                                disabled={isReingesting}
                                                className="w-full flex items-center justify-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                                            >
                                                {isReingesting ? (
                                                    <>
                                                        <svg className="animate-spin h-4 w-4 mr-2" viewBox="0 0 24 24">
                                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                                        </svg>
                                                        Re-ingesting...
                                                    </>
                                                ) : (
                                                    <>
                                                        <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                                        </svg>
                                                        Re-ingest All Documents
                                                    </>
                                                )}
                                            </button>
                                            
                                            <button
                                                onClick={loadIngestionStatus}
                                                className="w-full flex items-center justify-center px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
                                            >
                                                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                                </svg>
                                                Refresh Status
                                            </button>
                                            
                                            <button
                                                onClick={handleCleanupOrphanedDocuments}
                                                className="w-full flex items-center justify-center px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700"
                                            >
                                                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                                </svg>
                                                Cleanup Orphaned Documents
                                            </button>
                                        </div>
                                    </div>

                                    {/* Document Lists */}
                                    {ingestionStatus && (
                                        <div className="space-y-4">
                                            {/* Ingested Documents */}
                                            {ingestionStatus.ingested_documents.count > 0 && (
                                                <div className="bg-white p-4 rounded-lg border border-gray-200">
                                                    <h4 className="font-medium text-green-700 mb-2">
                                                        Ingested Documents ({ingestionStatus.ingested_documents.count})
                                                    </h4>
                                                    <div className="max-h-40 overflow-y-auto space-y-2">
                                                        {ingestionStatus.ingested_documents.files.map((doc, index) => (
                                                            <div key={doc.id} className="flex items-center justify-between p-2 bg-green-50 rounded text-sm">
                                                                <div className="flex-1 min-w-0">
                                                                    <div className="font-medium text-gray-900 truncate">{doc.filename}</div>
                                                                    <div className="text-gray-500 text-xs">{formatFileSize(doc.size)}</div>
                                                                </div>
                                                                <span className="text-green-600 text-xs">✓ Ingested</span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}

                                            {/* Pending Documents */}
                                            {ingestionStatus.pending_documents.count > 0 && (
                                                <div className="bg-white p-4 rounded-lg border border-gray-200">
                                                    <h4 className="font-medium text-orange-700 mb-2">
                                                        Pending Documents ({ingestionStatus.pending_documents.count})
                                                    </h4>
                                                    <div className="max-h-40 overflow-y-auto space-y-2">
                                                        {ingestionStatus.pending_documents.files.map((doc, index) => (
                                                            <div key={doc.id} className="flex items-center justify-between p-2 bg-orange-50 rounded text-sm">
                                                                <div className="flex-1 min-w-0">
                                                                    <div className="font-medium text-gray-900 truncate">{doc.filename}</div>
                                                                    <div className="text-gray-500 text-xs">{formatFileSize(doc.size)}</div>
                                                                </div>
                                                                <span className="text-orange-600 text-xs">⏳ Pending</span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="flex justify-end space-x-4 p-6 border-t border-gray-200 bg-gray-50">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-gray-700 bg-gray-200 rounded-lg hover:bg-gray-300 transition-colors disabled:opacity-50"
                        disabled={isLoadingModels || isSaving}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSave}
                        disabled={isLoadingModels || isSaving}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center"
                    >
                        {isSaving && (
                            <svg className="animate-spin h-4 w-4 mr-2 text-white" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                            </svg>
                        )}
                        {isSaving ? 'Downloading & Saving...' : 'Save Settings'}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ModelSettings;
