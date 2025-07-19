import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';

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
                
                <div className="p-4">
                    <nav className="space-y-2">
                        <button
                            onClick={() => setActiveTab('llm')}
                            className={`w-full flex items-center px-3 py-2 rounded-lg transition-colors ${
                                activeTab === 'llm' 
                                    ? 'bg-indigo-50 text-indigo-700' 
                                    : 'text-gray-700 hover:bg-gray-100'
                            }`}
                        >
                            <svg className="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                            </svg>
                            LLM Models
                        </button>

                        <button
                            onClick={() => setActiveTab('embedding')}
                            className={`w-full flex items-center px-3 py-2 rounded-lg transition-colors ${
                                activeTab === 'embedding' 
                                    ? 'bg-purple-50 text-purple-700' 
                                    : 'text-gray-700 hover:bg-gray-100'
                            }`}
                        >
                            <svg className="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                            </svg>
                            Embedding Models
                        </button>
                    </nav>
                    
                    {/* Stats */}
                    <div className="mt-8 p-4 bg-gray-50 rounded-lg">
                        <h3 className="text-sm font-medium text-gray-900 mb-2">Quick Info</h3>
                        <div className="text-sm text-gray-600 space-y-1">
                            {activeTab === 'llm' && (
                                <>
                                    <div className="flex justify-between">
                                        <span>Available:</span>
                                        <span className="font-medium">{availableModels.length}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Current:</span>
                                        <span className="font-medium text-xs">{settings.model || 'None'}</span>
                                    </div>
                                </>
                            )}
                            {activeTab === 'embedding' && (
                                <>
                                    <div className="flex justify-between">
                                        <span>Available:</span>
                                        <span className="font-medium">{embeddingModels.length}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span>Current:</span>
                                        <span className="font-medium text-xs">{currentEmbeddingModel || 'None'}</span>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 flex flex-col">
                {/* Header */}
                <div className="bg-white shadow-sm border-b">
                    <div className="px-6 py-4">
                        <div className="flex justify-between items-center">
                            <div>
                                <h2 className="text-2xl font-bold text-gray-900">
                                    {activeTab === 'llm' && 'Language Model Settings'}
                                    {activeTab === 'embedding' && 'Embedding Model Settings'}
                                </h2>
                                <p className="mt-1 text-sm text-gray-500">
                                    {activeTab === 'llm' && 'Configure AI model parameters and behavior'}
                                    {activeTab === 'embedding' && 'Manage embedding models for document processing'}
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
                                    value={settings.model}
                                    onChange={(e) => handleInputChange('model', e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                                >
                                    <option value="">Select a model</option>
                                    {availableModels.filter(model => model.category !== 'embedding').map((model) => (
                                        <option key={model.name} value={model.name}>
                                            {formatModelDisplayName(model)} {currentLLMModel === model.name && ' (Current)'}
                                        </option>
                                    ))}
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
                                    value={settings.temperature}
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
                                    value={settings.max_tokens}
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
                                    value={settings.top_p}
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
                                    value={settings.frequency_penalty}
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
                                    value={settings.presence_penalty}
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
                                {availableModels.find(m => m.name === settings.model) && (
                                    <div className="text-sm text-gray-600">
                                        <p><strong>Name:</strong> {settings.model}</p>
                                        {availableModels.find(m => m.name === settings.model)?.size && (
                                            <p><strong>Size:</strong> {availableModels.find(m => m.name === settings.model).size}</p>
                                        )}
                                        {availableModels.find(m => m.name === settings.model)?.description && (
                                            <p><strong>Description:</strong> {availableModels.find(m => m.name === settings.model).description}</p>
                                        )}
                                    </div>
                                )}
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
                                            value={currentEmbeddingModel}
                                            onChange={(e) => handleEmbeddingModelChange(e.target.value)}
                                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                                        >
                                            <option value="">Select an embedding model</option>
                                            {embeddingModels.map((model) => (
                                                <option key={model.name} value={model.name}>
                                                    {formatModelDisplayName(model)}
                                                </option>
                                            ))}
                                        </select>
                                        <p className="mt-1 text-sm text-gray-500">
                                            Choose the embedding model for document processing and semantic search
                                        </p>

                                        {/* Model Information */}
                                        {currentEmbeddingModel && (
                                            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                                                <h4 className="text-sm font-medium text-gray-900 mb-2">Current Model Info</h4>
                                                {embeddingModels.find(m => m.name === currentEmbeddingModel) && (
                                                    <div className="text-sm text-gray-600">
                                                        <p><strong>Name:</strong> {currentEmbeddingModel}</p>
                                                        {embeddingModels.find(m => m.name === currentEmbeddingModel)?.size && (
                                                            <p><strong>Size:</strong> {embeddingModels.find(m => m.name === currentEmbeddingModel).size}</p>
                                                        )}
                                                        <p><strong>Description:</strong> {
                                                            embeddingModels.find(m => m.name === currentEmbeddingModel)?.description || 
                                                            (currentEmbeddingModel.includes('bge') ? 'BGE Embedding Model' : 
                                                             currentEmbeddingModel.includes('e5') ? 'E5 Embedding Model' :
                                                             currentEmbeddingModel.includes('nomic') ? 'Nomic Embedding Model' :
                                                             currentEmbeddingModel.includes('mxbai') ? 'MxBai Embedding Model' :
                                                             currentEmbeddingModel.includes('arctic') ? 'Snowflake Arctic Embedding Model' :
                                                             'Embedding Model for document processing')
                                                        }</p>
                                                        <p className="mt-1 text-amber-600 font-medium">Note: Changing the embedding model will require re-ingesting all documents.</p>
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </div>
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
                                        {warningData.compatibility.recommendation && (
                                            <p className="text-sm text-yellow-700 mt-2">💡 {warningData.compatibility.recommendation}</p>
                                        )}
                                    </div>
                                )}
                                
                                {warningData.isLargeModel && (
                                    <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-md">
                                        <p className="text-sm text-red-800">
                                            <strong>Large Model Warning:</strong> This model requires significant computational resources. 
                                            Ensure you have adequate GPU memory (typically 24GB+ for 70B models).
                                        </p>
                                    </div>
                                )}
                            </div>
                            
                            <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
                                <p className="text-sm text-yellow-800">
                                    <strong>Recommendation:</strong> Consider selecting smaller models (7B-13B) for better performance 
                                    on consumer hardware, or ensure you have sufficient GPU memory before proceeding.
                                </p>
                            </div>
                        </div>
                        
                        <div className="flex justify-end space-x-3">
                            <button
                                onClick={() => {
                                    setShowWarningDialog(false);
                                    setWarningData(null);
                                    setDownloading(false);
                                    setDownloadProgress('');
                                }}
                                className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                            >
                                Choose Different Models
                            </button>
                            <button
                                onClick={() => proceedWithDownload(warningData.payload)}
                                className="px-4 py-2 border border-transparent rounded-md text-sm font-medium text-white bg-yellow-600 hover:bg-yellow-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500"
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
                    <div className="bg-white rounded-lg p-6 m-4 max-w-md w-full">
                        <div className="flex items-center mb-4">
                            <div className="flex-shrink-0">
                                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                            </div>
                            <div className="ml-3">
                                <h3 className="text-lg font-medium text-gray-900">Switch Embedding Model</h3>
                            </div>
                        </div>
                        
                        <div className="mb-6">
                            <p className="text-sm text-gray-600 mb-4">
                                You are about to switch the embedding model. This will require re-ingesting all documents and may take some time.
                            </p>
                            
                            <div className="bg-blue-50 p-4 rounded-md space-y-2">
                                <p className="text-sm text-gray-700">
                                    <strong>Current model:</strong> {embeddingWarningData.currentEmbedding || 'None'}
                                </p>
                                <p className="text-sm text-gray-700">
                                    <strong>New model:</strong> {embeddingWarningData.modelName}
                                </p>
                            </div>
                            
                            <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
                                <p className="text-sm text-yellow-800">
                                    <strong>Note:</strong> All existing documents will need to be re-processed with the new embedding model. 
                                    This process may take several minutes depending on the number of documents.
                                </p>
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
