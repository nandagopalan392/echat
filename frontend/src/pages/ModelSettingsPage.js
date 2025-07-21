import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';

// Safe utility: handles model as string or object
const getModelName = (model) => {
    if (!model) return '';
    if (typeof model === 'string') return model;
    if (typeof model === 'object' && model.name) return model.name;
    return '';
};

// Safe formatting of display name
const formatModelDisplayName = (model) => {
    const displayName = getModelName(model);

    // Parameter info
    let parameterInfo = '';
    const nameMatch = displayName.match(/(\d+\.?\d*)[bB]/i);
    if (nameMatch) {
        parameterInfo = ` (${nameMatch[1]}B params)`;
    } else if (displayName.includes(':')) {
        const colonMatch = displayName.match(/:(\d+\.?\d*)([bB])?/i);
        if (colonMatch) {
            parameterInfo = ` (${colonMatch[1]}B params)`;
        }
    }

    // Size info
    let sizeInfo = '';
    if (model && typeof model === 'object' && model.size && model.size !== 'Unknown') {
        if (model.size.match(/\d+(\.\d+)?\s*(GB|MB|KB)/i)) {
            sizeInfo = ` - ${model.size}`;
            parameterInfo = ''; // override paramInfo for actual file size
        } else if (model.size.toLowerCase().includes('various')) {
            sizeInfo = ` - ${model.size}`;
            parameterInfo = '';
        } else if (model.size.match(/^\d+\.?\d*[bB]$/i)) {
            if (!parameterInfo) {
                const sizeParam = model.size.match(/^(\d+\.?\d*)[bB]$/i);
                if (sizeParam) {
                    parameterInfo = ` (${sizeParam[1]}B params)`;
                }
            }
        } else {
            sizeInfo = ` - ${model.size}`;
        }
    }

    return `${displayName}${parameterInfo}${sizeInfo}`;
};


const ModelSettingsPage = () => {
    const navigate = useNavigate();
    
    // Loading states
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [downloading, setDownloading] = useState(false);
    const [downloadProgress, setDownloadProgress] = useState('');
    const [isChangingEmbedding, setIsChangingEmbedding] = useState(false);
    
    // UI state
    const [activeTab, setActiveTab] = useState('llm');
    const [showWarningDialog, setShowWarningDialog] = useState(false);
    const [warningData, setWarningData] = useState(null);
    const [showEmbeddingWarning, setShowEmbeddingWarning] = useState(false);
    const [embeddingWarningData, setEmbeddingWarningData] = useState(null);
    
    // Model data
    const [availableModels, setAvailableModels] = useState([]);
    const [embeddingModels, setEmbeddingModels] = useState([]);
    const [currentEmbeddingModel, setCurrentEmbeddingModel] = useState('');
    const [currentLLMModel, setCurrentLLMModel] = useState('');
    
    // Settings state
    const [settings, setSettings] = useState({
        model: '',
        temperature: 0.7,
        max_tokens: 2048,
        top_p: 0.9,
        frequency_penalty: 0,
        presence_penalty: 0,
        system_prompt: ''
    });

    useEffect(() => {
        loadModelSettings();
        loadAvailableModels();
        loadEmbeddingModels();
    }, []);

    const loadModelSettings = async () => {
        try {
            const response = await api.get('/api/models/current');
            if (response) {
                setCurrentEmbeddingModel(response.embedding);
                setCurrentLLMModel(response.llm);
            }
        } catch (error) {
            console.error('Error loading model settings:', error);
        }
    };

    const loadAvailableModels = async () => {
        try {
            const response = await api.get('/api/models/available');
            setAvailableModels(response.models || []);
            
            // Also load current LLM model
            const currentResponse = await api.get('/api/models/current');
            if (currentResponse) {
                setCurrentLLMModel(currentResponse.llm || '');
                // Set the current model in settings if not already set
                if (!settings.model && currentResponse.llm) {
                    setSettings(prev => ({
                        ...prev,
                        model: currentResponse.llm
                    }));
                }
            }
        } catch (error) {
            console.error('Error loading available models:', error);
        } finally {
            setLoading(false);
        }
    };

    const loadEmbeddingModels = async () => {
        try {
            const response = await api.get('/api/models/available');
            console.log('Full models response:', response);
            
            // Filter embedding models from the unified response
            const embeddingModels = (response.models || []).filter(model => 
                model.category === 'embedding'
            );
            
            console.log('Filtered embedding models:', embeddingModels);
            setEmbeddingModels(embeddingModels);
            
            // Also get current settings
            const currentResponse = await api.get('/api/models/current');
            if (currentResponse) {
                setCurrentEmbeddingModel(currentResponse.embedding || '');
            }
        } catch (error) {
            console.error('Error loading embedding models:', error);
        }
    };

    const handleSaveSettings = async () => {
        try {
            setSaving(true);
            setDownloading(false);
            setDownloadProgress('');
            
            // Prepare data in the format expected by the backend
            const payload = {
                llm: settings.model,
                embedding: currentEmbeddingModel
            };
            
            console.log('Saving settings payload:', payload);
            
            if (!payload.llm) {
                alert('Please select an LLM model first');
                return;
            }
            
            if (!payload.embedding) {
                alert('Please select an embedding model first');
                return;
            }
            
            // Check GPU compatibility first and show warning to user
            try {
                setDownloadProgress('Checking GPU compatibility...');
                const compatibilityResponse = await api.post('/api/models/check-gpu', payload);
                
                console.log('GPU Compatibility Response:', compatibilityResponse);
                
                // Force warning dialog for large models (for testing and safety)
                const isLargeModel = payload.llm.includes('70B') || payload.llm.includes('405B') || 
                                   payload.llm.includes('70b') || payload.llm.includes('405b') ||
                                   payload.llm.includes('33B') || payload.llm.includes('34B') ||
                                   payload.llm.includes('13B') || payload.llm.includes('13b') ||
                                   payload.llm.includes('7B') || payload.llm.includes('7b') ||
                                   payload.llm.includes('8B') || payload.llm.includes('8b');
                
                // Show dialog for incompatible models OR large models (basically most models for testing)
                const shouldShowWarning = (compatibilityResponse && !compatibilityResponse.compatible) || 
                                        isLargeModel || 
                                        (compatibilityResponse && compatibilityResponse.combined_check && 
                                         compatibilityResponse.combined_check.required_mb > 8000); // Show warning for models > 8GB
                
                if (shouldShowWarning) {
                    console.log('Showing GPU warning dialog for model:', payload.llm);
                    // Show warning dialog instead of basic confirm
                    setWarningData({
                        llmModel: payload.llm,
                        embeddingModel: payload.embedding,
                        compatibility: compatibilityResponse,
                        payload: payload,
                        isLargeModel: isLargeModel
                    });
                    setShowWarningDialog(true);
                    setDownloading(false);
                    setDownloadProgress('');
                    return; // Don't proceed until user confirms
                } else {
                    console.log('Models are compatible, proceeding with download');
                }
            } catch (compatError) {
                console.warn('GPU compatibility check failed, proceeding:', compatError);
                // Continue if GPU check fails - don't block the user
            }
            
            setDownloading(true);
            setDownloadProgress('Downloading models if needed...');
            
            // Use the simpler endpoint that doesn't require complex validation
            const response = await api.post('/api/models/simple-settings', payload);
            
            // Update current model states
            setCurrentLLMModel(payload.llm);
            setCurrentEmbeddingModel(payload.embedding);
            
            setDownloading(false);
            setDownloadProgress('');
            
            let message = 'Settings saved successfully!';
            if (response.downloaded_models && response.downloaded_models.length > 0) {
                message += ` Downloaded models: ${response.downloaded_models.join(', ')}`;
            }
            if (response.gpu_warnings && response.gpu_warnings.length > 0) {
                message += `\n\n⚠️ GPU Compatibility Warnings:\n${response.gpu_warnings.join('\n')}`;
            }
            if (response.reingest_suggested) {
                message += '\n\nNote: Embedding model changed - consider re-ingesting documents.';
            }
            
            alert(message);
        } catch (error) {
            console.error('Error saving settings:', error);
            setDownloading(false);
            setDownloadProgress('');
            
            if (error.response?.data?.detail) {
                const errorDetail = error.response.data.detail;
                let errorMessage = '';
                
                if (typeof errorDetail === 'string') {
                    errorMessage = errorDetail;
                } else {
                    errorMessage = JSON.stringify(errorDetail);
                }
                
                // Check if it's a GPU compatibility error
                if (errorMessage.includes('GPU') || errorMessage.includes('compatible') || errorMessage.includes('memory')) {
                    alert(`⚠️ GPU Compatibility Issue:\n\n${errorMessage}\n\nPlease select smaller models or upgrade your GPU.`);
                } else {
                    alert(`Error saving settings: ${errorMessage}`);
                }
            } else {
                alert('Error saving settings. Please try again.');
            }
        } finally {
            setSaving(false);
        }
    };

    const proceedWithDownload = async (payload) => {
        try {
            setDownloading(true);
            setDownloadProgress('Downloading models if needed...');
            setShowWarningDialog(false);
            
            const response = await api.post('/api/models/simple-settings', payload);
            
            if (response && response.success) {
                setCurrentLLMModel(payload.llm);
                setCurrentEmbeddingModel(payload.embedding);
                
                let message = 'Model settings saved successfully!';
                
                if (response.llm_changed) {
                    message += '\n\nLLM model has been changed.';
                }
                
                if (response.embedding_changed) {
                    message += '\n\nNote: Embedding model changed - consider re-ingesting documents.';
                }
                
                alert(message);
            }
        } catch (error) {
            console.error('Error saving settings:', error);
            
            if (error.response?.data?.detail) {
                const errorDetail = error.response.data.detail;
                let errorMessage = '';
                
                if (typeof errorDetail === 'string') {
                    errorMessage = errorDetail;
                } else {
                    errorMessage = JSON.stringify(errorDetail);
                }
                
                alert(`Error saving settings: ${errorMessage}`);
            } else {
                alert('Error saving settings. Please try again.');
            }
        } finally {
            setDownloading(false);
            setDownloadProgress('');
        }
    };

    const handleInputChange = (field, value) => {
        setSettings(prev => ({
            ...prev,
            [field]: value
        }));
    };

    const resetToDefaults = () => {
        setSettings({
            model: availableModels[0]?.name || '',
            temperature: 0.7,
            max_tokens: 2048,
            top_p: 0.9,
            frequency_penalty: 0,
            presence_penalty: 0,
            system_prompt: ''
        });
    };

    const handleEmbeddingModelChange = async (modelName) => {
        // Show warning dialog instead of window.confirm
        setEmbeddingWarningData({
            modelName: modelName,
            currentEmbedding: currentEmbeddingModel
        });
        setShowEmbeddingWarning(true);
    };

    const proceedWithEmbeddingChange = async (modelName) => {
        try {
            setIsChangingEmbedding(true);
            setShowEmbeddingWarning(false);
            
            // Get current LLM model first
            const currentResponse = await api.get('/api/models/current');
            const currentLLM = currentResponse?.llm || 'deepseek-r1:latest';
            
            // Update both models using the unified API
            await api.post('/api/models/settings', {
                llm: currentLLM,
                embedding: modelName
            });
            
            setCurrentEmbeddingModel(modelName);
            alert('Embedding model switched successfully! Documents will be re-processed automatically.');
        } catch (error) {
            console.error('Error switching embedding model:', error);
            alert('Error switching embedding model. Please try again.');
            await loadEmbeddingModels(); // Reload to reset the current model
        } finally {
            setIsChangingEmbedding(false);
        }
    };

    // Helper function to format model display name with size and parameters
    const formatModelDisplayName = (model) => {
        let displayName = model.name;
        
        // Extract parameter count from model name (this shows model complexity)
        let parameterInfo = '';
        const nameMatch = model.name.match(/(\d+\.?\d*)[bB]/i);
        if (nameMatch) {
            parameterInfo = ` (${nameMatch[1]}B params)`;
        } else if (model.name.includes(':')) {
            // For models like "llama3:8b", extract the parameter info
            const colonMatch = model.name.match(/:(\d+\.?\d*)([bB])?/i);
            if (colonMatch) {
                parameterInfo = ` (${colonMatch[1]}B params)`;
            }
        }
        
        // Add size information 
        let sizeInfo = '';
        if (model.size && model.size !== 'Unknown') {
            // Check if this is a file size (contains GB, MB, KB) or parameter/variant info
            if (model.size.match(/\d+(\.\d+)?\s*(GB|MB|KB)/i)) {
                // This is an actual file size - show it
                sizeInfo = ` - ${model.size}`;
                // If we already have parameter info from the name, don't duplicate
                // Remove parameter info if size shows actual file size
                if (parameterInfo) {
                    parameterInfo = '';
                }
            } else if (model.size.toLowerCase().includes('various')) {
                // For models with multiple variants, show as-is instead of params from name
                sizeInfo = ` - ${model.size}`;
                parameterInfo = ''; // Don't show param info from name since size has variant info
            } else if (model.size.match(/^\d+\.?\d*[bB]$/i)) {
                // This looks like parameter count from library (e.g., "70b", "3.8b")
                // Only show if we don't already have param info from name
                if (!parameterInfo) {
                    const sizeParam = model.size.match(/^(\d+\.?\d*)[bB]$/i);
                    if (sizeParam) {
                        parameterInfo = ` (${sizeParam[1]}B params)`;
                    }
                }
            } else {
                // Other size info, show as-is
                sizeInfo = ` - ${model.size}`;
            }
        }
        
        return `${displayName}${parameterInfo}${sizeInfo}`;
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="text-center">
                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
                    <p className="mt-2 text-gray-500">Loading model settings...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50 flex">
            {/* Sidebar */}
            <div className="w-64 bg-white shadow-lg">
                <div className="p-6 border-b border-gray-200">
                    <div className="flex items-center">
                        <button
                            onClick={() => navigate('/chat')}
                            className="mr-3 p-2 text-gray-400 hover:text-gray-600"
                        >
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                            </svg>
                        </button>
                        <h1 className="text-xl font-bold text-gray-900">Model Settings</h1>
                    </div>
                </div>
                
                <nav className="mt-6">
                    <div className="px-3 space-y-1">
                        <button
                            onClick={() => setActiveTab('llm')}
                            className={`${
                                activeTab === 'llm'
                                    ? 'bg-indigo-100 border-indigo-500 text-indigo-700'
                                    : 'border-transparent text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                            } group flex items-center w-full pl-2 pr-2 py-2 border-l-4 text-sm font-medium`}
                        >
                            <svg className="mr-3 h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                            </svg>
                            Language Model
                        </button>
                        
                        <button
                            onClick={() => setActiveTab('embedding')}
                            className={`${
                                activeTab === 'embedding'
                                    ? 'bg-indigo-100 border-indigo-500 text-indigo-700'
                                    : 'border-transparent text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                            } group flex items-center w-full pl-2 pr-2 py-2 border-l-4 text-sm font-medium`}
                        >
                            <svg className="mr-3 h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14-7H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V6a2 2 0 00-2-2z" />
                            </svg>
                            Embedding Model
                        </button>
                    </div>
                </nav>
            </div>

            {/* Main Content */}
            <div className="flex-1 flex flex-col">
                {/* Header */}
                <div className="bg-white shadow-sm border-b">
                    <div className="px-6 py-4">
                        <div className="flex justify-between items-center">
                            <div>
                                <h2 className="text-2xl font-bold text-gray-900">
                                    {activeTab === 'llm' ? 'Language Model Settings' : 'Embedding Model Settings'}
                                </h2>
                                <p className="mt-1 text-sm text-gray-500">
                                    {activeTab === 'llm' ? 'Configure AI model parameters and behavior' : 'Manage embedding models for document processing'}
                                </p>
                            </div>
                            <div className="flex items-center space-x-4">
                                {activeTab === 'llm' && (
                                    <>
                                        <button
                                            onClick={resetToDefaults}
                                            className="px-4 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                                        >
                                            Reset to Defaults
                                        </button>
                                        <button
                                            onClick={handleSaveSettings}
                                            disabled={saving || downloading}
                                            className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg transition-colors disabled:opacity-50"
                                        >
                                            {downloading ? 'Downloading...' : saving ? 'Saving...' : 'Save Settings'}
                                        </button>
                                    </>
                                )}
                                
                                {/* Progress indicator */}
                                {(downloading || downloadProgress) && (
                                    <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                                        <div className="flex items-center space-x-3">
                                            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                                            <div>
                                                <p className="text-sm font-medium text-blue-900">Model Download in Progress</p>
                                                <p className="text-sm text-blue-700">{downloadProgress}</p>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 p-6">
                    {/* LLM Settings Tab */}
                    {activeTab === 'llm' && (
                        <div className="bg-white rounded-lg shadow">
                            <div className="p-6">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    {/* Model Selection */}
                                    <div className="md:col-span-2">
                                        <label className="block text-sm font-medium text-gray-700 mb-2">
                                            AI Model
                                        </label>
                                        
                                        {/* Current Model Display */}
                                        {currentLLMModel && (
                                            <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                                                <h4 className="text-sm font-medium text-blue-900 mb-2">Currently Selected Model</h4>
                                                <div className="flex items-center">
                                                    <div className="flex-grow">
                                                        <p className="text-lg font-semibold text-blue-800">{currentLLMModel}</p>
                                                    </div>
                                                </div>
                                            </div>
                                        )}
                                        
                                        <select
                                            value={settings.model || ''}
                                            onChange={(e) => handleInputChange('model', e.target.value)}
                                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                                        >
                                            <option value="">Select a model</option>
                                            {Array.isArray(availableModels) && availableModels
                                                .filter(model => {
                                                    if (typeof model === 'string') return true;
                                                    return model && typeof model === 'object' && (!model.category || model.category !== 'embedding');
                                                })
                                                .map((model, index) => {
                                                    const modelName = getModelName(model);
                                                    const key = modelName || `model-${index}`;
                                                    return (
                                                        <option key={key} value={modelName}>
                                                            {formatModelDisplayName(model)} {currentLLMModel === modelName ? ' (Current)' : ''}
                                                        </option>
                                                    );
                                                })}
                                        </select>
                                        <p className="mt-1 text-sm text-gray-500">
                                            Choose the AI model for generating responses
                                        </p>
                                    </div>

                                    {/* Temperature */}
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 mb-2">
                                            Temperature: {settings.temperature}
                                        </label>
                                        <input
                                            type="range"
                                            min="0"
                                            max="2"
                                            step="0.1"
                                            value={settings.temperature || 0.7}
                                            onChange={(e) => handleInputChange('temperature', parseFloat(e.target.value))}
                                            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                        />
                                        <div className="flex justify-between text-xs text-gray-500 mt-1">
                                            <span>More Focused</span>
                                            <span>More Creative</span>
                                        </div>
                                    </div>

                                    {/* Max Tokens */}
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 mb-2">
                                            Max Tokens
                                        </label>
                                        <input
                                            type="number"
                                            min="100"
                                            max="8192"
                                            value={settings.max_tokens || 2048}
                                            onChange={(e) => handleInputChange('max_tokens', parseInt(e.target.value))}
                                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                                        />
                                        <p className="mt-1 text-sm text-gray-500">
                                            Maximum length of the response
                                        </p>
                                    </div>

                                    {/* Top P */}
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 mb-2">
                                            Top P: {settings.top_p}
                                        </label>
                                        <input
                                            type="range"
                                            min="0"
                                            max="1"
                                            step="0.05"
                                            value={settings.top_p || 0.9}
                                            onChange={(e) => handleInputChange('top_p', parseFloat(e.target.value))}
                                            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                        />
                                        <p className="mt-1 text-sm text-gray-500">
                                            Controls diversity via nucleus sampling
                                        </p>
                                    </div>

                                    {/* Frequency Penalty */}
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 mb-2">
                                            Frequency Penalty: {settings.frequency_penalty}
                                        </label>
                                        <input
                                            type="range"
                                            min="-2"
                                            max="2"
                                            step="0.1"
                                            value={settings.frequency_penalty || 0}
                                            onChange={(e) => handleInputChange('frequency_penalty', parseFloat(e.target.value))}
                                            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                        />
                                        <p className="mt-1 text-sm text-gray-500">
                                            Reduces repetition of tokens
                                        </p>
                                    </div>

                                    {/* Presence Penalty */}
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 mb-2">
                                            Presence Penalty: {settings.presence_penalty}
                                        </label>
                                        <input
                                            type="range"
                                            min="-2"
                                            max="2"
                                            step="0.1"
                                            value={settings.presence_penalty || 0}
                                            onChange={(e) => handleInputChange('presence_penalty', parseFloat(e.target.value))}
                                            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                                        />
                                        <p className="mt-1 text-sm text-gray-500">
                                            Encourages talking about new topics
                                        </p>
                                    </div>
                                </div>

                                {/* Model Information */}
                                {settings.model && (
                                    <div className="mt-8 p-4 bg-gray-50 rounded-lg">
                                        <h3 className="text-lg font-medium text-gray-900 mb-2">Model Information</h3>
                                        {(() => {
                                            const selectedModel = Array.isArray(availableModels) ? 
                                                availableModels.find(m => getModelName(m) === settings.model) : null;
                                            if (selectedModel) {
                                                return (
                                                    <div className="text-sm text-gray-600">
                                                        <p><strong>Name:</strong> {settings.model}</p>
                                                        {selectedModel.size && (
                                                            <p><strong>Size:</strong> {selectedModel.size}</p>
                                                        )}
                                                        {selectedModel.description && (
                                                            <p><strong>Description:</strong> {selectedModel.description}</p>
                                                        )}
                                                    </div>
                                                );
                                            }
                                            return (
                                                <div className="text-sm text-gray-600">
                                                    <p><strong>Name:</strong> {settings.model}</p>
                                                </div>
                                            );
                                        })()}
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Embedding Settings Tab */}
                    {activeTab === 'embedding' && (
                        <div className="bg-white rounded-lg shadow">
                            <div className="px-6 py-4 border-b border-gray-200">
                                <h3 className="text-lg font-medium text-gray-900">Embedding Model Configuration</h3>
                                <p className="mt-1 text-sm text-gray-500">Choose and configure the embedding model for document processing</p>
                            </div>
                            <div className="p-6">
                                <div className="space-y-6">
                                    {/* Current Model Display */}
                                    {currentEmbeddingModel && (
                                        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                                            <h4 className="text-sm font-medium text-blue-900 mb-2">Currently Selected Model</h4>
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <p className="text-lg font-semibold text-blue-800">{currentEmbeddingModel}</p>
                                                    <p className="text-sm text-blue-600">Active embedding model for document processing</p>
                                                </div>
                                                <span className="inline-flex px-3 py-1 text-sm font-semibold rounded-full bg-green-100 text-green-800">
                                                    Active
                                                </span>
                                            </div>
                                        </div>
                                    )}
                                    
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 mb-3">
                                            Embedding Model Selection
                                        </label>
                                        <select
                                            value={currentEmbeddingModel || ''}
                                            onChange={(e) => handleEmbeddingModelChange(e.target.value)}
                                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                                        >
                                            <option value="">Select an embedding model</option>
                                            {Array.isArray(embeddingModels) && embeddingModels.map((model, index) => {
                                                const modelName = getModelName(model);
                                                const key = modelName || `embedding-${index}`;
                                                return (
                                                    <option key={key} value={modelName}>
                                                        {formatModelDisplayName(model)}
                                                    </option>
                                                );
                                            })}
                                        </select>
                                        <p className="mt-1 text-sm text-gray-500">
                                            Choose the embedding model for document processing and semantic search
                                        </p>

                                        {/* Model Information */}
                                        {currentEmbeddingModel && (
                                            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                                                <h4 className="text-sm font-medium text-gray-900 mb-2">Current Model Info</h4>
                                                {(() => {
                                                    const selectedModel = Array.isArray(embeddingModels) ? 
                                                        embeddingModels.find(m => getModelName(m) === currentEmbeddingModel) : null;
                                                    return (
                                                        <div className="text-sm text-gray-600">
                                                            <p><strong>Name:</strong> {currentEmbeddingModel}</p>
                                                            {selectedModel && selectedModel.size && (
                                                                <p><strong>Size:</strong> {selectedModel.size}</p>
                                                            )}
                                                            <p><strong>Description:</strong> {
                                                                (selectedModel && selectedModel.description) || 
                                                                (currentEmbeddingModel.includes('bge') ? 'BGE Embedding Model' : 
                                                                 currentEmbeddingModel.includes('e5') ? 'E5 Embedding Model' :
                                                                 currentEmbeddingModel.includes('nomic') ? 'Nomic Embedding Model' :
                                                                 currentEmbeddingModel.includes('mxbai') ? 'MxBai Embedding Model' :
                                                                 currentEmbeddingModel.includes('arctic') ? 'Snowflake Arctic Embedding Model' :
                                                                 'Embedding Model for document processing')
                                                            }</p>
                                                            <p className="mt-1 text-amber-600 font-medium">Note: Changing the embedding model will require re-ingesting all documents.</p>
                                                        </div>
                                                    );
                                                })()}
                                            </div>
                                        )}
                                    </div>
                                    
                                    {/* Progress indicator for downloads/processing */}
                                    {(downloadProgress || isChangingEmbedding) && (
                                        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                                            <div className="flex items-center space-x-3">
                                                <div className="flex-shrink-0">
                                                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                                                </div>
                                                <div className="flex-1">
                                                    <div className="text-sm font-medium text-blue-900">
                                                        {isChangingEmbedding ? 'Switching embedding model...' : 'Processing...'}
                                                    </div>
                                                    {downloadProgress && (
                                                        <div className="text-sm text-blue-700 mt-1">
                                                            {downloadProgress}
                                                        </div>
                                                    )}
                                                    <div className="mt-2">
                                                        <div className="w-full bg-blue-200 rounded-full h-2">
                                                            <div className="bg-blue-600 h-2 rounded-full animate-pulse" style={{width: '45%'}}></div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* GPU Compatibility Warning Dialog */}
            {showWarningDialog && warningData && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-white rounded-lg p-6 m-4 max-w-2xl w-full max-h-[80vh] overflow-y-auto">
                        <div className="flex items-center mb-4">
                            <div className="flex-shrink-0">
                                <svg className="w-8 h-8 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                                </svg>
                            </div>
                            <div className="ml-3">
                                <h3 className="text-lg font-medium text-gray-900">⚠️ GPU Memory Warning</h3>
                            </div>
                        </div>
                        
                        <div className="mb-6">
                            <p className="text-sm text-gray-600 mb-4">
                                {warningData.isLargeModel 
                                    ? "You've selected a large language model that may require significant GPU memory."
                                    : "The selected models may not fit in your GPU memory and could cause system issues."
                                }
                            </p>
                            
                            <div className="bg-gray-50 p-4 rounded-md space-y-3">
                                <div>
                                    <h4 className="font-medium text-gray-900">Selected Models:</h4>
                                    <p className="text-sm text-gray-600">LLM: <span className="font-mono">{warningData.llmModel}</span></p>
                                    <p className="text-sm text-gray-600">Embedding: <span className="font-mono">{warningData.embeddingModel}</span></p>
                                </div>
                                
                                {warningData.compatibility && (
                                    <div>
                                        <h4 className="font-medium text-gray-900">Memory Analysis:</h4>
                                        <p className="text-sm text-gray-600">{warningData.compatibility.llm_check?.message}</p>
                                        <p className="text-sm text-gray-600">{warningData.compatibility.embedding_check?.message}</p>
                                        <p className="text-sm font-medium text-gray-800">{warningData.compatibility.combined_check?.message}</p>
                                    </div>
                                )}
                            </div>
                        </div>
                        
                        <div className="flex justify-end space-x-3">
                            <button
                                onClick={() => setShowWarningDialog(false)}
                                className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={() => {
                                    setShowWarningDialog(false);
                                    // Actually apply the settings that triggered the warning
                                    if (warningData.action === 'save') {
                                        handleSaveSettings();
                                    }
                                }}
                                className="px-4 py-2 border border-transparent rounded-md text-sm font-medium text-white bg-orange-600 hover:bg-orange-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-orange-500"
                            >
                                Proceed Anyway
                            </button>
                        </div>
                    </div>
                </div>
            )}
            
            {/* Embedding Model Change Warning Dialog */}
            {showEmbeddingWarning && embeddingWarningData && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-white rounded-lg p-6 m-4 max-w-lg w-full">
                        <div className="flex items-center mb-4">
                            <div className="flex-shrink-0">
                                <svg className="w-8 h-8 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                                </svg>
                            </div>
                            <div className="ml-3">
                                <h3 className="text-lg font-medium text-gray-900">⚠️ Change Embedding Model</h3>
                            </div>
                        </div>
                        
                        <div className="mb-6">
                            <p className="text-sm text-gray-600 mb-4">
                                Changing the embedding model will require re-ingesting all documents. This process may take some time.
                            </p>
                            
                            <div className="bg-yellow-50 p-4 rounded-md">
                                <div className="flex">
                                    <div className="flex-shrink-0">
                                        <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                                            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                                        </svg>
                                    </div>
                                    <div className="ml-3">
                                        <h3 className="text-sm font-medium text-yellow-800">Important</h3>
                                        <div className="mt-2 text-sm text-yellow-700">
                                            <p>• All documents will be re-processed with the new embedding model</p>
                                            <p>• Search results may differ from the previous model</p>
                                            <p>• This operation cannot be undone</p>
                                            <p><strong>Note:</strong> All existing documents will need to be re-processed with the new embedding model. This process may take several minutes depending on the number of documents.</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div className="flex justify-end space-x-3">
                            <button
                                onClick={() => {
                                    setShowEmbeddingWarning(false);
                                    setEmbeddingWarningData(null);
                                }}
                                className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={() => proceedWithEmbeddingChange(embeddingWarningData.modelName)}
                                className="px-4 py-2 border border-transparent rounded-md text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                            >
                                Switch Model
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ModelSettingsPage;
