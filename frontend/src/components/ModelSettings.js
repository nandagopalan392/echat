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

    // Load available models when dialog opens
    useEffect(() => {
        if (isOpen) {
            loadAvailableModels();
            loadCurrentSettings();
        }
    }, [isOpen]);

    const loadAvailableModels = async () => {
        setIsLoadingModels(true);
        setError('');
        
        try {
            console.log('Loading available models from Ollama...');
            const response = await api.get('/api/models/available');
            console.log('Available models response:', response);
            
            if (response && response.models) {
                setAvailableModels(response.models);
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
            console.log('Saving model settings:', selectedModels);
            const response = await api.post('/api/models/settings', selectedModels);
            console.log('Save response:', response);
            
            if (response && response.success) {
                onSave?.(selectedModels);
                onClose();
            } else {
                setError('Failed to save model settings');
            }
        } catch (err) {
            console.error('Error saving settings:', err);
            setError('Failed to save model settings');
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
            // If model has category, use it
            if (model.category) {
                return model.category === type;
            }
            
            // Fallback to name-based filtering for backward compatibility
            const name = model.name.toLowerCase();
            switch (type) {
                case 'llm':
                    return !name.includes('embed') && !name.includes('rerank') && (
                        name.includes('deepseek') || 
                        name.includes('llama') || 
                        name.includes('mistral') ||
                        name.includes('codellama') ||
                        name.includes('qwen') ||
                        name.includes('phi') ||
                        name.includes('gemma')
                    );
                case 'embedding':
                    return name.includes('embed') || 
                           name.includes('bge') ||
                           name.includes('nomic');
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

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-hidden relative">
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
                <div className="flex justify-between items-center p-6 border-b border-gray-200">
                    <div className="flex items-center gap-4">
                        <h2 className="text-xl font-semibold text-gray-900">Model Settings</h2>
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
                <div className="p-6 overflow-y-auto max-h-[70vh]">
                    {error && (
                        <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
                            {error}
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
