import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

const ModelSettings = ({ isOpen, onClose, onSave }) => {
    const [availableModels, setAvailableModels] = useState([]);
    const [selectedModels, setSelectedModels] = useState({
        llm: 'deepseek-r1:latest',
        embedding: 'mxbai-embed-large',
        reranker: 'BAAI/bge-reranker-large'
    });
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    // Load available models when dialog opens
    useEffect(() => {
        if (isOpen) {
            loadAvailableModels();
            loadCurrentSettings();
        }
    }, [isOpen]);

    const loadAvailableModels = async () => {
        setIsLoading(true);
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
            setIsLoading(false);
        }
    };

    const loadCurrentSettings = async () => {
        try {
            console.log('Loading current model settings...');
            const response = await api.get('/api/models/current');
            console.log('Current settings response:', response);
            
            if (response) {
                setSelectedModels({
                    llm: response.llm || 'deepseek-r1:latest',
                    embedding: response.embedding || 'mxbai-embed-large',
                    reranker: response.reranker || 'BAAI/bge-reranker-large'
                });
            }
        } catch (err) {
            console.error('Error loading current settings:', err);
            // Use defaults if loading fails
        }
    };

    const handleModelChange = (modelType, modelName) => {
        setSelectedModels(prev => ({
            ...prev,
            [modelType]: modelName
        }));
    };

    const handleSave = async () => {
        setIsLoading(true);
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
            setIsLoading(false);
        }
    };

    const filterModelsByType = (type) => {
        if (!availableModels || !Array.isArray(availableModels)) return [];
        
        switch (type) {
            case 'llm':
                return availableModels.filter(model => 
                    model.name.includes('deepseek') || 
                    model.name.includes('llama') || 
                    model.name.includes('mistral') ||
                    model.name.includes('codellama') ||
                    model.name.includes('qwen') ||
                    !model.name.includes('embed')
                );
            case 'embedding':
                return availableModels.filter(model => 
                    model.name.includes('embed') || 
                    model.name.includes('bge') ||
                    model.name.includes('nomic')
                );
            case 'reranker':
                // For rerankers, we'll show both Ollama models and predefined options
                const ollamaRerankers = availableModels.filter(model => 
                    model.name.includes('rerank') || 
                    model.name.includes('bge')
                );
                const predefinedRerankers = [
                    { name: 'BAAI/bge-reranker-large', size: 'Built-in' },
                    { name: 'BAAI/bge-reranker-base', size: 'Built-in' },
                    { name: 'cross-encoder/ms-marco-MiniLM-L-6-v2', size: 'Built-in' }
                ];
                return [...ollamaRerankers, ...predefinedRerankers];
            default:
                return availableModels;
        }
    };

    const formatModelSize = (size) => {
        if (!size) return '';
        if (typeof size === 'string') return size;
        if (size < 1024 * 1024 * 1024) {
            return `${Math.round(size / (1024 * 1024))} MB`;
        }
        return `${(size / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-hidden">
                {/* Header */}
                <div className="flex justify-between items-center p-6 border-b border-gray-200">
                    <h2 className="text-xl font-semibold text-gray-900">Model Settings</h2>
                    <button
                        onClick={onClose}
                        className="text-gray-400 hover:text-gray-600 transition-colors"
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

                    {isLoading && (
                        <div className="mb-4 text-center">
                            <div className="inline-flex items-center">
                                <svg className="animate-spin h-5 w-5 mr-3 text-blue-600" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                </svg>
                                Loading models...
                            </div>
                        </div>
                    )}

                    <div className="space-y-6">
                        {/* LLM Model Selection */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Large Language Model (LLM)
                            </label>
                            <p className="text-xs text-gray-500 mb-3">
                                The main AI model used for generating responses and reasoning.
                            </p>
                            <div className="border border-gray-300 rounded-lg max-h-40 overflow-y-auto">
                                {filterModelsByType('llm').map((model, index) => (
                                    <label
                                        key={`llm-${model.name}-${index}`}
                                        className={`flex items-center p-3 cursor-pointer hover:bg-gray-50 ${
                                            index > 0 ? 'border-t border-gray-200' : ''
                                        }`}
                                    >
                                        <input
                                            type="radio"
                                            name="llm"
                                            value={model.name}
                                            checked={selectedModels.llm === model.name}
                                            onChange={(e) => handleModelChange('llm', e.target.value)}
                                            className="mr-3 text-blue-600"
                                        />
                                        <div className="flex-1">
                                            <div className="font-medium text-gray-900">{model.name}</div>
                                            {model.size && (
                                                <div className="text-sm text-gray-500">
                                                    Size: {formatModelSize(model.size)}
                                                </div>
                                            )}
                                        </div>
                                    </label>
                                ))}
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
                                {filterModelsByType('embedding').map((model, index) => (
                                    <label
                                        key={`embedding-${model.name}-${index}`}
                                        className={`flex items-center p-3 cursor-pointer hover:bg-gray-50 ${
                                            index > 0 ? 'border-t border-gray-200' : ''
                                        }`}
                                    >
                                        <input
                                            type="radio"
                                            name="embedding"
                                            value={model.name}
                                            checked={selectedModels.embedding === model.name}
                                            onChange={(e) => handleModelChange('embedding', e.target.value)}
                                            className="mr-3 text-blue-600"
                                        />
                                        <div className="flex-1">
                                            <div className="font-medium text-gray-900">{model.name}</div>
                                            {model.size && (
                                                <div className="text-sm text-gray-500">
                                                    Size: {formatModelSize(model.size)}
                                                </div>
                                            )}
                                        </div>
                                    </label>
                                ))}
                            </div>
                        </div>

                        {/* Reranker Model Selection */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Reranker Model
                            </label>
                            <p className="text-xs text-gray-500 mb-3">
                                Improves search result relevance by reordering retrieved documents.
                            </p>
                            <div className="border border-gray-300 rounded-lg max-h-40 overflow-y-auto">
                                {filterModelsByType('reranker').map((model, index) => (
                                    <label
                                        key={`reranker-${model.name}-${index}`}
                                        className={`flex items-center p-3 cursor-pointer hover:bg-gray-50 ${
                                            index > 0 ? 'border-t border-gray-200' : ''
                                        }`}
                                    >
                                        <input
                                            type="radio"
                                            name="reranker"
                                            value={model.name}
                                            checked={selectedModels.reranker === model.name}
                                            onChange={(e) => handleModelChange('reranker', e.target.value)}
                                            className="mr-3 text-blue-600"
                                        />
                                        <div className="flex-1">
                                            <div className="font-medium text-gray-900">{model.name}</div>
                                            {model.size && (
                                                <div className="text-sm text-gray-500">
                                                    Size: {formatModelSize(model.size)}
                                                </div>
                                            )}
                                        </div>
                                    </label>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div className="flex justify-end space-x-4 p-6 border-t border-gray-200 bg-gray-50">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-gray-700 bg-gray-200 rounded-lg hover:bg-gray-300 transition-colors"
                        disabled={isLoading}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSave}
                        disabled={isLoading}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
                    >
                        {isLoading ? 'Saving...' : 'Save Settings'}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ModelSettings;
