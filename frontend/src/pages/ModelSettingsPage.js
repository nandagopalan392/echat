import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import GatedModelDialog from '../components/GatedModelDialog';
import {
    Box,
    Drawer,
    List,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Typography,
    IconButton,
    Divider,
    ThemeProvider,
    Card,
    CardContent,
    Grid,
    Chip,
    CircularProgress,
    Avatar,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    FormControlLabel,
    Switch,
    TextField,
    Slider,
    Button,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Alert,
    AlertTitle
} from '@mui/material';
import {
    ArrowBack,
    Computer as ComputerIcon,
    Memory as MemoryIcon,
    Settings as SettingsIcon,
    CloudDownload as CloudDownloadIcon,
    Hub as HubIcon,
    CheckCircle as CheckCircleIcon,
    EmojiObjects as EmojiObjectsIcon
} from '@mui/icons-material';
import { theme } from '../theme';
import ollamaIcon from '../assets/ollama.svg';
import huggingfaceIcon from '../assets/huggingface.svg';

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
    
    // Gated model dialog state
    const [showGatedModelDialog, setShowGatedModelDialog] = useState(false);
    const [gatedModelInfo, setGatedModelInfo] = useState(null);
    
    // Model data
    const [availableModels, setAvailableModels] = useState([]);
    const [embeddingModels, setEmbeddingModels] = useState([]);
    const [currentEmbeddingModel, setCurrentEmbeddingModel] = useState('');
    const [currentLLMModel, setCurrentLLMModel] = useState('');
    
    // Provider state
    const [providers, setProviders] = useState({});
    const [selectedProvider, setSelectedProvider] = useState('ollama');
    const [selectedEmbeddingProvider, setSelectedEmbeddingProvider] = useState('ollama');
    const [isLoadingProviders, setIsLoadingProviders] = useState(false);
    
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
        console.log('🔄 ModelSettingsPage useEffect triggered');
        loadProviders();
        loadModelSettings();
        loadAvailableModels();
        loadEmbeddingModels();
    }, []);

    // Reload models when provider changes
    useEffect(() => {
        if (selectedProvider) {
            console.log(`🔄 Provider changed to: ${selectedProvider}, reloading models...`);
            // Clear current selections when switching providers
            setSettings(prev => ({ ...prev, model: '' }));
            setCurrentLLMModel('');
            setCurrentEmbeddingModel('');
            // Reload embedding models with new provider filter
            loadEmbeddingModels();
        }
    }, [selectedProvider]);

    // Reload embedding models when embedding provider changes
    useEffect(() => {
        if (selectedEmbeddingProvider) {
            console.log(`🔄 Embedding provider changed to: ${selectedEmbeddingProvider}, reloading embedding models...`);
            // Clear current embedding model selection when switching providers
            setCurrentEmbeddingModel('');
            // Reload embedding models with new provider filter
            loadEmbeddingModels();
        }
    }, [selectedEmbeddingProvider]);

    const loadProviders = async () => {
        setIsLoadingProviders(true);
        try {
            console.log('🔄 Loading providers...');
            const response = await api.get('/api/models/providers');
            console.log('📡 Providers API response:', response);
            
            if (response && response.providers) {
                console.log('✅ Setting providers:', response.providers);
                setProviders(response.providers);
                
                // Only set default provider if none selected (let loadModelSettings handle DB provider)
                if (!selectedProvider && Object.keys(response.providers).length > 0) {
                    console.log('🔄 Setting default provider to first available');
                    setSelectedProvider(Object.keys(response.providers)[0]);
                }
            } else {
                console.log('❌ No providers in response:', response);
            }
        } catch (err) {
            console.error('❌ Error loading providers:', err);
            console.log('❌ Provider loading error details:', err.response || err);
        } finally {
            setIsLoadingProviders(false);
        }
    };

    const loadModelSettings = async () => {
        try {
            const response = await api.get('/api/models/current');
            if (response) {
                setCurrentEmbeddingModel(response.embedding);
                setCurrentLLMModel(response.llm);
                
                // Set provider from the database response
                if (response.provider) {
                    console.log(`🔄 Setting provider from model settings: ${response.provider}`);
                    setSelectedProvider(response.provider);
                }
                
                // Set embedding provider from the database response or detect from embedding model
                if (response.embedding_provider) {
                    console.log(`🔄 Setting embedding provider from model settings: ${response.embedding_provider}`);
                    setSelectedEmbeddingProvider(response.embedding_provider);
                } else if (response.embedding) {
                    // Auto-detect embedding provider from model name
                    const detectedProvider = response.embedding.includes('/') ? 'huggingface' : 'ollama';
                    console.log(`🔄 Auto-detected embedding provider from model name: ${detectedProvider}`);
                    setSelectedEmbeddingProvider(detectedProvider);
                }
                
                // Load parameters if available
                if (response.parameters) {
                    setSettings(prev => ({
                        ...prev,
                        model: response.llm,
                        temperature: response.parameters.temperature || 0.7,
                        max_tokens: response.parameters.max_tokens || 2048,
                        top_p: response.parameters.top_p || 0.9,
                        frequency_penalty: response.parameters.frequency_penalty || 0,
                        presence_penalty: response.parameters.presence_penalty || 0
                    }));
                }
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

    // Filter models by provider and type
    const getFilteredModels = (type = 'llm', providerKey = null) => {
        // Use providerKey parameter, or default to selected provider for LLM or selectedEmbeddingProvider for embedding
        const targetProvider = providerKey || (type === 'embedding' ? selectedEmbeddingProvider : selectedProvider);
        
        if (targetProvider && providers[targetProvider] && providers[targetProvider].models) {
            console.log(`🔍 Filtering ${type} models for provider ${targetProvider}:`, providers[targetProvider].models);
            
            const providerModels = providers[targetProvider].models;
            
            // Add defensive check for providerModels
            if (!Array.isArray(providerModels)) {
                console.error('⚠️ Provider models is not an array:', providerModels);
                return [];
            }
            
            return providerModels.filter(model => {
                // Add defensive check for model
                if (!model) {
                    console.warn('⚠️ Found null/undefined model in provider models');
                    return false;
                }
                
                // Handle both string format (old) and object format (new)
                const modelName = typeof model === 'string' ? model : model.name;
                const modelType = typeof model === 'object' && model.type ? model.type : null;
                
                if (!modelName) {
                    console.warn('⚠️ Model has no name:', model);
                    return false;
                }
                
                // If model has explicit type metadata, use it
                if (modelType) {
                    return modelType === type;
                }
                
                // Fallback to name-based detection for backward compatibility
                const name = modelName.toLowerCase();
                
                // Enhanced embedding model detection
                const isEmbeddingModel = name.includes('embed') || 
                                       name.includes('bge') ||
                                       name.includes('minilm') ||
                                       name.includes('all-minilm') ||
                                       name.includes('nomic') ||
                                       name.includes('e5-') ||
                                       name.includes('sentence') ||
                                       name.includes('text-embedding') ||
                                       name.includes('rerank');
                
                if (type === 'llm') {
                    return !isEmbeddingModel;
                } else if (type === 'embedding') {
                    return isEmbeddingModel;
                }
                return true;
            }).map(model => {
                // Handle both string and object formats
                const modelName = typeof model === 'string' ? model : model.name;
                return {
                    name: modelName,
                    category: type,
                    provider: targetProvider,
                    size: typeof model === 'object' ? model.size || 'Unknown' : 'Unknown',
                    source: 'provider',
                    downloads: typeof model === 'object' ? model.downloads : undefined
                };
            });
        }
        
        // Fallback to original logic
        return (availableModels || []).filter(model => {
            if (model.category) {
                return model.category === type;
            }
            
            const name = model.name.toLowerCase();
            const isEmbeddingModel = name.includes('embed') || 
                                   name.includes('bge') ||
                                   name.includes('minilm') ||
                                   name.includes('all-minilm') ||
                                   name.includes('nomic') ||
                                   name.includes('e5-') ||
                                   name.includes('sentence') ||
                                   name.includes('text-embedding');
            
            if (type === 'llm') {
                return !isEmbeddingModel && !name.includes('rerank');
            } else if (type === 'embedding') {
                return isEmbeddingModel;
            }
            return true;
        });
    };

    const loadEmbeddingModels = async () => {
        try {
            // Use filtered models based on provider selection
            const embeddingModels = getFilteredModels('embedding');
            
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
                embedding: currentEmbeddingModel,
                provider: selectedProvider, // Include provider information
                // Include language model parameters
                parameters: {
                    temperature: settings.temperature,
                    max_tokens: settings.max_tokens,
                    top_p: settings.top_p,
                    frequency_penalty: settings.frequency_penalty,
                    presence_penalty: settings.presence_penalty
                }
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

            // Check if models are the same - if so, skip GPU compatibility check
            const modelsUnchanged = (payload.llm === currentLLMModel && payload.embedding === currentEmbeddingModel);
            
            if (modelsUnchanged) {
                console.log('Models unchanged, skipping GPU compatibility check and proceeding directly');
                // Skip GPU check and proceed directly with parameter update
                proceedWithDownload(payload);
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
                        action: 'save',
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
            
            console.log('🔍 SAVE SETTINGS RESPONSE:', response);
            
            // Check if the response contains a gated model error (even in success response)
            if (response && typeof response === 'object') {
                // Check all string values in the response for gated model errors
                const responseStr = JSON.stringify(response);
                console.log('🔍 RESPONSE STRING:', responseStr);
                if (responseStr.includes('GATED_MODEL_ERROR:')) {
                    try {
                        // Find the start of the JSON after "GATED_MODEL_ERROR:"
                        const startMarker = 'GATED_MODEL_ERROR:';
                        const startIndex = responseStr.indexOf(startMarker) + startMarker.length;
                        
                        // Find the matching closing brace by counting braces
                        let braceCount = 0;
                        let endIndex = startIndex;
                        let foundStart = false;
                        
                        for (let i = startIndex; i < responseStr.length; i++) {
                            if (responseStr[i] === '{') {
                                if (!foundStart) foundStart = true;
                                braceCount++;
                            } else if (responseStr[i] === '}') {
                                braceCount--;
                                if (foundStart && braceCount === 0) {
                                    endIndex = i + 1;
                                    break;
                                }
                            }
                        }
                        
                        if (foundStart && braceCount === 0) {
                            let jsonString = responseStr.substring(startIndex, endIndex);
                            console.log('🔍 EXTRACTED JSON STRING:', jsonString);
                            
                            // Replace escaped quotes with regular quotes - handle double escaping
                            jsonString = jsonString.replace(/\\\\"/g, '"').replace(/\\"/g, '"');
                            console.log('🔍 UNESCAPED JSON STRING:', jsonString);
                            
                            const gatedModelData = JSON.parse(jsonString);
                            
                            console.log('🚨 Gated model error detected in success response:', gatedModelData);
                            
                            // Show gated model dialog
                            setGatedModelInfo(gatedModelData);
                            setShowGatedModelDialog(true);
                            setDownloading(false);
                            setDownloadProgress('');
                            return;
                        }
                    } catch (parseError) {
                        console.error('Failed to parse gated model error from success response:', parseError);
                        // Fall through to regular success handling
                    }
                }
            }
            
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
                    
                    // Check for gated model error
                    if (errorMessage.includes('GATED_MODEL_ERROR:')) {
                        try {
                            const gatedErrorJson = errorMessage.split('GATED_MODEL_ERROR:')[1];
                            const gatedModelData = JSON.parse(gatedErrorJson);
                            
                            console.log('Gated model error detected:', gatedModelData);
                            
                            // Show gated model dialog
                            setGatedModelInfo(gatedModelData);
                            setShowGatedModelDialog(true);
                            return;
                        } catch (parseError) {
                            console.error('Failed to parse gated model error:', parseError);
                            // Fall through to regular error handling
                        }
                    }
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
            console.log('proceedWithDownload called with payload:', payload);
            setDownloading(true);
            setDownloadProgress('Downloading models if needed...');
            setShowWarningDialog(false);
            
            console.log('Calling /api/models/simple-settings with payload:', payload);
            const response = await api.post('/api/models/simple-settings', payload);
            
            // Check if the response contains a gated model error (even in success response)
            if (response && typeof response === 'object') {
                // Check all string values in the response for gated model errors
                const responseStr = JSON.stringify(response);
                if (responseStr.includes('GATED_MODEL_ERROR:')) {
                    try {
                        // Find the start of the JSON after GATED_MODEL_ERROR:
                        const startIndex = responseStr.indexOf('GATED_MODEL_ERROR:') + 'GATED_MODEL_ERROR:'.length;
                        const jsonStart = responseStr.indexOf('{', startIndex);
                        
                        if (jsonStart !== -1) {
                            // Parse JSON by counting braces to find the end
                            let braceCount = 0;
                            let jsonEnd = jsonStart;
                            let inString = false;
                            let escaped = false;
                            
                            for (let i = jsonStart; i < responseStr.length; i++) {
                                const char = responseStr[i];
                                
                                if (escaped) {
                                    escaped = false;
                                    continue;
                                }
                                
                                if (char === '\\') {
                                    escaped = true;
                                    continue;
                                }
                                
                                if (char === '"') {
                                    inString = !inString;
                                    continue;
                                }
                                
                                if (!inString) {
                                    if (char === '{') {
                                        braceCount++;
                                    } else if (char === '}') {
                                        braceCount--;
                                        if (braceCount === 0) {
                                            jsonEnd = i + 1;
                                            break;
                                        }
                                    }
                                }
                            }
                            
                            let jsonString = responseStr.substring(jsonStart, jsonEnd);
                            console.log('🔍 EXTRACTED JSON STRING (proceedWithDownload):', jsonString);
                            
                            // Handle double-escaped JSON strings
                            jsonString = jsonString.replace(/\\"/g, '"').replace(/\\\\"/g, '\\"');
                            console.log('🔍 UNESCAPED JSON STRING (proceedWithDownload):', jsonString);
                            
                            const gatedModelData = JSON.parse(jsonString);
                            
                            console.log('Gated model error detected in proceedWithDownload success response:', gatedModelData);
                            
                            // Show gated model dialog
                            setGatedModelInfo(gatedModelData);
                            setShowGatedModelDialog(true);
                            setDownloading(false);
                            setDownloadProgress('');
                            return;
                        }
                    } catch (parseError) {
                        console.error('Failed to parse gated model error from proceedWithDownload success response:', parseError);
                        // Fall through to regular success handling
                    }
                }
            }
            
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
                    
                    // Check for gated model error
                    if (errorMessage.includes('GATED_MODEL_ERROR:')) {
                        try {
                            const gatedErrorJson = errorMessage.split('GATED_MODEL_ERROR:')[1];
                            const gatedModelData = JSON.parse(gatedErrorJson);
                            
                            console.log('Gated model error detected in proceedWithDownload:', gatedModelData);
                            
                            // Show gated model dialog
                            setGatedModelInfo(gatedModelData);
                            setShowGatedModelDialog(true);
                            return;
                        } catch (parseError) {
                            console.error('Failed to parse gated model error:', parseError);
                            // Fall through to regular error handling
                        }
                    }
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
                embedding: modelName,
                provider: selectedProvider, // Include provider for LLM
                embedding_provider: selectedEmbeddingProvider // Include provider for embedding
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

    // Helper function to extract model name from both string and object formats
    const getModelName = (model) => {
        if (typeof model === 'string') return model;
        if (typeof model === 'object' && model && model.name) return String(model.name);
        return 'unknown';
    };

    // Helper function to format model display name with size and parameters
    const formatModelDisplayName = (model) => {
        let displayName = getModelName(model);
        
        // Extract parameter count from model name (this shows model complexity)
        let parameterInfo = '';
        const nameMatch = displayName.match(/(\d+\.?\d*)[bB]/i);
        if (nameMatch) {
            parameterInfo = ` (${nameMatch[1]}B params)`;
        } else if (displayName.includes(':')) {
            // For models like "llama3:8b", extract the parameter info
            const colonMatch = displayName.match(/:(\d+\.?\d*)([bB])?/i);
            if (colonMatch) {
                parameterInfo = ` (${colonMatch[1]}B params)`;
            }
        }
        
        // Add size information 
        let sizeInfo = '';
        if (model && typeof model === 'object' && model.size && model.size !== 'Unknown') {
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
        
        return `${displayName}${parameterInfo}${sizeInfo}` || 'Unknown Model';
    };

    const tabs = [
        { id: 'llm', name: 'Language Model', icon: ComputerIcon },
        { id: 'embedding', name: 'Embedding Model', icon: MemoryIcon }
    ];

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
        <ThemeProvider theme={theme}>
            <Box sx={{ minHeight: '100vh', bgcolor: '#f8fafc', display: 'flex' }}>
                {/* Sidebar */}
                <Drawer
                    variant="permanent"
                    sx={{
                        width: 280,
                        flexShrink: 0,
                        '& .MuiDrawer-paper': {
                            width: 280,
                            boxSizing: 'border-box',
                            bgcolor: '#ffffff',
                            borderRight: '1px solid #f1f5f9',
                            boxShadow: '0px 4px 6px rgba(0, 0, 0, 0.07), 0px 2px 4px rgba(0, 0, 0, 0.06)',
                            background: 'linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)',
                        },
                    }}
                >
                    <Box sx={{ p: 3 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                            <IconButton
                                edge="start"
                                onClick={() => navigate('/chat')}
                                sx={{ 
                                    mr: 2,
                                    color: '#64748b',
                                    borderRadius: '10px',
                                    transition: 'all 0.2s ease-in-out',
                                    '&:hover': {
                                        backgroundColor: 'rgba(37, 99, 235, 0.08)',
                                        color: '#2563eb',
                                        transform: 'scale(1.05)',
                                    },
                                }}
                            >
                                <ArrowBack />
                            </IconButton>
                            <SettingsIcon sx={{ 
                                mr: 1, 
                                color: '#2563eb',
                                fontSize: '1.5rem',
                            }} />
                            <Typography variant="h6" sx={{ 
                                fontWeight: 700,
                                color: '#0f172a',
                                fontSize: '1.125rem',
                            }}>
                                Model Settings
                            </Typography>
                        </Box>
                        <Divider sx={{ 
                            mb: 3,
                            borderColor: 'rgba(148, 163, 184, 0.2)',
                        }} />
                        
                        {/* Navigation Items */}
                        <List sx={{ p: 0 }}>
                            {tabs.map((tab) => {
                                const IconComponent = tab.icon;
                                return (
                                    <ListItemButton
                                        key={tab.id}
                                        selected={activeTab === tab.id}
                                        onClick={() => setActiveTab(tab.id)}
                                        sx={{ 
                                            borderRadius: '12px', 
                                            mb: 1,
                                            margin: '4px 0',
                                            padding: '12px 16px',
                                            transition: 'all 0.2s ease-in-out',
                                            '&:hover': {
                                                backgroundColor: 'rgba(37, 99, 235, 0.08)',
                                                transform: 'translateX(4px)',
                                            },
                                            '&.Mui-selected': {
                                                backgroundColor: 'rgba(37, 99, 235, 0.12)',
                                                color: '#2563eb',
                                                '&:hover': {
                                                    backgroundColor: 'rgba(37, 99, 235, 0.16)',
                                                },
                                                '& .MuiListItemIcon-root': {
                                                    color: '#2563eb',
                                                },
                                                '& .MuiListItemText-primary': {
                                                    color: '#2563eb',
                                                    fontWeight: 600,
                                                },
                                            },
                                        }}
                                    >
                                        <ListItemIcon sx={{ 
                                            color: activeTab === tab.id ? '#2563eb' : '#64748b',
                                            minWidth: '40px',
                                        }}>
                                            <IconComponent />
                                        </ListItemIcon>
                                        <ListItemText 
                                            primary={tab.name} 
                                            primaryTypographyProps={{
                                                fontSize: '0.875rem',
                                                fontWeight: activeTab === tab.id ? 600 : 500,
                                                color: activeTab === tab.id ? '#2563eb' : '#475569',
                                            }}
                                        />
                                    </ListItemButton>
                                );
                            })}
                        </List>
                    </Box>
                </Drawer>

                {/* Main Content */}
                <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                    {/* Header */}
                    <Box sx={{ bgcolor: 'white', borderBottom: '1px solid #f1f5f9', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)' }}>
                        <Box sx={{ px: 3, py: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Box>
                                    <Typography variant="h4" sx={{ 
                                        fontWeight: 700, 
                                        color: '#0f172a',
                                        fontSize: '1.5rem',
                                        mb: 0.5,
                                    }}>
                                        {activeTab === 'llm' ? 'Language Model Settings' : 'Embedding Model Settings'}
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: '#64748b' }}>
                                        {activeTab === 'llm' ? 'Configure AI model parameters and behavior' : 'Manage embedding models for document processing'}
                                    </Typography>
                                </Box>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
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
                                </Box>
                            </Box>
                        </Box>
                    </Box>

                    {/* Content */}
                    <Box sx={{ flexGrow: 1, p: 3 }}>
                        {/* LLM Settings Tab */}
                        {activeTab === 'llm' && (
                        <div className="bg-white rounded-lg shadow">
                            <div className="px-6 py-4 border-b border-gray-200">
                                <h3 className="text-lg font-medium text-gray-900">Language Model Configuration</h3>
                                <p className="mt-1 text-sm text-gray-500">Choose and configure the language model for chat responses</p>
                            </div>
                            <div className="p-6">
                                <div className="space-y-6">
                                    {/* Current Model Display */}
                                    {currentLLMModel && (
                                        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                                            <h4 className="text-sm font-medium text-blue-900 mb-2">Currently Selected Model</h4>
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <p className="text-lg font-semibold text-blue-800">{currentLLMModel}</p>
                                                    <p className="text-sm text-blue-600">Active language model for chat responses</p>
                                                </div>
                                                <span className="inline-flex px-3 py-1 text-sm font-semibold rounded-full bg-green-100 text-green-800">
                                                    Active
                                                </span>
                                            </div>
                                        </div>
                                    )}
                                    
                                    {/* Provider Selection for LLM Models */}
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 mb-3">
                                            Model Provider
                                        </label>
                                        <div className="grid grid-cols-2 gap-4">
                                            {/* Ollama Provider */}
                                            <Card 
                                                className={`cursor-pointer transition-all duration-200 ${
                                                    selectedProvider === 'ollama' 
                                                        ? 'ring-2 ring-indigo-500 bg-indigo-50' 
                                                        : 'hover:shadow-md'
                                                }`}
                                                onClick={() => setSelectedProvider('ollama')}
                                            >
                                                <CardContent className="p-4">
                                                    <div className="flex items-center space-x-3">
                                                        <Avatar className="h-8 w-8">
                                                            <img 
                                                                src={ollamaIcon} 
                                                                alt="Ollama" 
                                                                className="h-8 w-8"
                                                            />
                                                        </Avatar>
                                                        <div className="flex-1">
                                                            <h3 className="text-sm font-medium text-gray-900">Ollama</h3>
                                                            <p className="text-xs text-gray-500">Local models</p>
                                                        </div>
                                                        {selectedProvider === 'ollama' && (
                                                            <CheckCircleIcon className="h-5 w-5 text-indigo-600" />
                                                        )}
                                                    </div>
                                                </CardContent>
                                            </Card>

                                            {/* HuggingFace Provider */}
                                            <Card 
                                                className={`cursor-pointer transition-all duration-200 ${
                                                    selectedProvider === 'huggingface' 
                                                        ? 'ring-2 ring-indigo-500 bg-indigo-50' 
                                                        : 'hover:shadow-md'
                                                }`}
                                                onClick={() => setSelectedProvider('huggingface')}
                                            >
                                                <CardContent className="p-4">
                                                    <div className="flex items-center space-x-3">
                                                        <Avatar className="h-8 w-8">
                                                            <img 
                                                                src={huggingfaceIcon} 
                                                                alt="HuggingFace" 
                                                                className="h-8 w-8"
                                                            />
                                                        </Avatar>
                                                        <div className="flex-1">
                                                            <h3 className="text-sm font-medium text-gray-900">HuggingFace</h3>
                                                            <p className="text-xs text-gray-500">Cloud models</p>
                                                        </div>
                                                        {selectedProvider === 'huggingface' && (
                                                            <CheckCircleIcon className="h-5 w-5 text-indigo-600" />
                                                        )}
                                                    </div>
                                                </CardContent>
                                            </Card>
                                        </div>
                                        <p className="mt-2 text-sm text-gray-500">
                                            Select the provider for language models
                                        </p>
                                    </div>
                                    
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 mb-3">
                                            Language Model Selection
                                        </label>
                                        <FormControl fullWidth variant="outlined">
                                            <Select
                                                value={settings.model || ''}
                                                onChange={(e) => handleInputChange('model', e.target.value)}
                                                displayEmpty
                                                className="bg-white"
                                            >
                                                <MenuItem value="">Select a model</MenuItem>
                                                {getFilteredModels('llm', selectedProvider).map((model, index) => {
                                                    const modelName = getModelName(model);
                                                    const key = modelName || `model-${index}`;
                                                    return (
                                                        <MenuItem key={key} value={modelName}>
                                                            {String(formatModelDisplayName(model))} {currentLLMModel === modelName ? ' (Current)' : ''}
                                                            {model.provider && model.provider !== 'ollama' ? ` [${model.provider}]` : ''}
                                                        </MenuItem>
                                                    );
                                                })}
                                            </Select>
                                        </FormControl>
                                        <p className="mt-1 text-sm text-gray-500">
                                            Choose the language model for generating responses
                                        </p>
                                    </div>

                                    {/* Model Parameters */}
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
                                            min="0"
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
                                            min="0"
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
                                    
                                    {/* Provider Selection for Embedding Models */}
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 mb-3">
                                            Model Provider
                                        </label>
                                        <div className="grid grid-cols-2 gap-4">
                                            {/* Ollama Provider */}
                                            <Card 
                                                className={`cursor-pointer transition-all duration-200 ${
                                                    selectedEmbeddingProvider === 'ollama' 
                                                        ? 'ring-2 ring-indigo-500 bg-indigo-50' 
                                                        : 'hover:shadow-md'
                                                }`}
                                                onClick={() => setSelectedEmbeddingProvider('ollama')}
                                            >
                                                <CardContent className="p-4">
                                                    <div className="flex items-center space-x-3">
                                                        <Avatar className="h-8 w-8">
                                                            <img 
                                                                src={ollamaIcon} 
                                                                alt="Ollama" 
                                                                className="h-8 w-8"
                                                            />
                                                        </Avatar>
                                                        <div className="flex-1">
                                                            <h3 className="text-sm font-medium text-gray-900">Ollama</h3>
                                                            <p className="text-xs text-gray-500">Local models</p>
                                                        </div>
                                                        {selectedEmbeddingProvider === 'ollama' && (
                                                            <CheckCircleIcon className="h-5 w-5 text-indigo-600" />
                                                        )}
                                                    </div>
                                                </CardContent>
                                            </Card>

                                            {/* HuggingFace Provider */}
                                            <Card 
                                                className={`cursor-pointer transition-all duration-200 ${
                                                    selectedEmbeddingProvider === 'huggingface' 
                                                        ? 'ring-2 ring-indigo-500 bg-indigo-50' 
                                                        : 'hover:shadow-md'
                                                }`}
                                                onClick={() => setSelectedEmbeddingProvider('huggingface')}
                                            >
                                                <CardContent className="p-4">
                                                    <div className="flex items-center space-x-3">
                                                        <Avatar className="h-8 w-8">
                                                            <img 
                                                                src={huggingfaceIcon} 
                                                                alt="HuggingFace" 
                                                                className="h-8 w-8"
                                                            />
                                                        </Avatar>
                                                        <div className="flex-1">
                                                            <h3 className="text-sm font-medium text-gray-900">HuggingFace</h3>
                                                            <p className="text-xs text-gray-500">Cloud models</p>
                                                        </div>
                                                        {selectedEmbeddingProvider === 'huggingface' && (
                                                            <CheckCircleIcon className="h-5 w-5 text-indigo-600" />
                                                        )}
                                                    </div>
                                                </CardContent>
                                            </Card>
                                        </div>
                                        <p className="mt-2 text-sm text-gray-500">
                                            Select the provider for embedding models
                                        </p>
                                    </div>
                                    
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 mb-3">
                                            Embedding Model Selection
                                        </label>
                                        <FormControl fullWidth variant="outlined">
                                            <Select
                                                value={currentEmbeddingModel || ''}
                                                onChange={(e) => handleEmbeddingModelChange(e.target.value)}
                                                displayEmpty
                                                className="bg-white"
                                            >
                                                <MenuItem value="">Select an embedding model</MenuItem>
                                                {getFilteredModels('embedding', selectedEmbeddingProvider).map((model, index) => {
                                                    const modelName = getModelName(model);
                                                    const key = modelName || `embedding-${index}`;
                                                    return (
                                                        <MenuItem key={key} value={modelName}>
                                                            {String(formatModelDisplayName(model))}
                                                            {model.provider && model.provider !== 'ollama' ? ` [${model.provider}]` : ''}
                                                        </MenuItem>
                                                    );
                                                })}
                                            </Select>
                                        </FormControl>
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
                    </Box>
                </Box>

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
                                    console.log('User clicked "Proceed Anyway" - calling proceedWithDownload directly');
                                    // Directly proceed with the settings that triggered the warning
                                    if (warningData.action === 'save' && warningData.payload) {
                                        proceedWithDownload(warningData.payload);
                                    } else {
                                        setShowWarningDialog(false);
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

            {/* Gated Model Dialog */}
            <GatedModelDialog
                isOpen={showGatedModelDialog}
                onClose={() => {
                    setShowGatedModelDialog(false);
                    setGatedModelInfo(null);
                }}
                gatedModelInfo={gatedModelInfo}
            />
            </Box>
        </ThemeProvider>
    );
};

export default ModelSettingsPage;
