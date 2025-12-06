import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    Box,
    Card,
    CardContent,
    Typography,
    Grid,
    Button,
    TextField,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Chip,
    Alert,
    LinearProgress,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    IconButton,
    Tooltip,
    Snackbar,
    CircularProgress,
    FormControlLabel,
    Checkbox,
    Divider,
    Drawer,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    ListItemSecondaryAction,
    Container,
    Tabs,
    Tab
} from '@mui/material';
import {
    PlayArrow as PlayIcon,
    Stop as StopIcon,
    Delete as DeleteIcon,
    CloudUpload as UploadIcon,
    Info as InfoIcon,
    Visibility as ViewIcon,
    Download as DownloadIcon,
    Add as AddIcon,
    Settings as SettingsIcon,
    Timeline as TimelineIcon,
    Assessment as AssessmentIcon,
    DataObject as DataIcon,
    Memory as MemoryIcon,
    Speed as SpeedIcon,
    Timer as TimerIcon,
    TrendingUp as TrendingUpIcon,
    Close as CloseIcon,
    Refresh as RefreshIcon
} from '@mui/icons-material';

import api from '../services/api';
import { evaluationApi } from '../services/api';
import webSocketService from '../services/websocketService';
import TrainingDashboard from '../components/TrainingDashboard';
import ExperimentComparison from '../components/ExperimentComparison';

const SIDEBAR_WIDTH = 280;

const FinetuningPage = () => {
    const navigate = useNavigate();
    const [activeTab, setActiveTab] = useState(0);
    
    // Training Dashboard State
    const [selectedTrainingExperiment, setSelectedTrainingExperiment] = useState(null);
    const [showTrainingDashboard, setShowTrainingDashboard] = useState(false);
    const [trainingUpdates, setTrainingUpdates] = useState(null); // Store training updates for dashboard
    
    // Core State
    const [experiments, setExperiments] = useState([]);
    const [availableModels, setAvailableModels] = useState([]);
    const [datasets, setDatasets] = useState([]);
    const [evaluationDatasets, setEvaluationDatasets] = useState([]);
    const [availableDocuments, setAvailableDocuments] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);
    
    // Form State
    const [experimentName, setExperimentName] = useState('');
    const [experimentDescription, setExperimentDescription] = useState('');
    const [selectedModelName, setSelectedModelName] = useState('');
    const [selectedDatasetId, setSelectedDatasetId] = useState('');
    const [learningRate, setLearningRate] = useState(0.0001);
    const [numEpochs, setNumEpochs] = useState(3);
    const [batchSize, setBatchSize] = useState(4);
    const [loraR, setLoraR] = useState(16);
    const [loraAlpha, setLoraAlpha] = useState(32);
    const [loraDropout, setLoraDropout] = useState(0.1);
    const [targetModules, setTargetModules] = useState('q_proj,v_proj');
    
    // Dataset Dialog State
    const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
    const [createDatasetDialogOpen, setCreateDatasetDialogOpen] = useState(false);
    const [viewDatasetDialogOpen, setViewDatasetDialogOpen] = useState(false);
    const [datasetDetails, setDatasetDetails] = useState(null);
    
    // Create Dataset State
    const [newDatasetName, setNewDatasetName] = useState('');
    const [newDatasetDescription, setNewDatasetDescription] = useState('');
    const [selectedDocuments, setSelectedDocuments] = useState([]);
    const [questionsPerDoc, setQuestionsPerDoc] = useState(5);
    
    // WebSocket State
    const [webSocket, setWebSocket] = useState(null);
    
    // Q-C-A Dataset WebSocket State (following EvaluationPage pattern)
    const [qcaActiveConnections, setQcaActiveConnections] = useState(new Set());
    const [qcaProgress, setQcaProgress] = useState(new Map()); // taskId -> progress info
    const [qcaConnectionStatuses, setQcaConnectionStatuses] = useState(new Map());

    // Load initial data
    useEffect(() => {
        loadInitialData();
    }, []);

    const loadInitialData = async () => {
        setLoading(true);
        try {
            await Promise.all([
                loadExperiments(),
                loadAvailableModels(),
                loadDatasets(),
                loadEvaluationDatasets(),
                loadAvailableDocuments()
            ]);
        } catch (err) {
            setError('Failed to load initial data: ' + (err.message || 'Unknown error'));
        } finally {
            setLoading(false);
        }
    };

    const loadExperiments = async () => {
        try {
            const data = await api.getFineTuningExperiments();
            setExperiments(data.experiments || []);
        } catch (err) {
            console.error('Failed to load experiments:', err);
            setError('Failed to load experiments: ' + (err.message || 'Unknown error'));
        }
    };

    const loadAvailableModels = async () => {
        try {
            const data = await api.getAvailableHFModels();
            setAvailableModels(data.models || []);
        } catch (err) {
            console.error('Failed to load models:', err);
            setError('Failed to load models: ' + (err.message || 'Unknown error'));
        }
    };

    const loadDatasets = async () => {
        try {
            const data = await api.getFineTuningDatasets();
            setDatasets(data.datasets || []);
        } catch (err) {
            console.error('Failed to load datasets:', err);
            setError('Failed to load datasets: ' + (err.message || 'Unknown error'));
        }
    };

    const loadEvaluationDatasets = async () => {
        try {
            const data = await evaluationApi.getDatasets();
            setEvaluationDatasets(data.datasets || []);
        } catch (err) {
            console.error('Failed to load evaluation datasets:', err);
            setError('Failed to load evaluation datasets: ' + (err.message || 'Unknown error'));
        }
    };

    const loadAvailableDocuments = async () => {
        try {
            const data = await api.getDocuments();
            setAvailableDocuments(data.documents || []);
        } catch (err) {
            console.error('Failed to load documents:', err);
            setError('Failed to load documents: ' + (err.message || 'Unknown error'));
        }
    };

    const handleCreateExperiment = async () => {
        if (!experimentName || !selectedModelName || !selectedDatasetId) {
            setError('Please fill in all required fields and select a dataset');
            return;
        }

        setLoading(true);
        try {
            const formData = new FormData();
            
            // Add basic fields
            formData.append('name', experimentName);
            formData.append('description', experimentDescription);
            formData.append('model_name', selectedModelName);
            formData.append('dataset_id', selectedDatasetId);
            
            // Add training parameters
            formData.append('learning_rate', learningRate.toString());
            formData.append('num_epochs', numEpochs.toString());
            formData.append('batch_size', batchSize.toString());
            
            // Add LoRA parameters
            formData.append('lora_r', loraR.toString());
            formData.append('lora_alpha', loraAlpha.toString());
            formData.append('lora_dropout', loraDropout.toString());
            formData.append('target_modules', targetModules);

            console.log('🧪 [Experiment] Creating experiment...');
            const result = await api.createFineTuningExperiment(formData);
            console.log('🧪 [Experiment] Create result:', result);
            
            const experimentId = result?.experiment?.id;
            setSuccess(`Experiment "${result?.experiment?.name || 'New experiment'}" created successfully!`);
            
            // Set up WebSocket connection for training progress (training auto-starts)
            if (experimentId) {
                console.log('🔌 [Experiment] Setting up WebSocket for auto-started training:', experimentId);
                const ws = webSocketService.connect(experimentId, {
                    endpointType: 'finetuning',
                    onMessage: (data) => {
                        console.log('📨 [Experiment] Training update:', data);
                        // Store updates for TrainingDashboard
                        setTrainingUpdates(data);
                        if (data.status === 'completed') {
                            setSuccess('Training completed successfully!');
                            webSocketService.disconnect(experimentId);
                        } else if (data.status === 'failed') {
                            setError('Training failed: ' + (data.error_message || 'Unknown error'));
                            webSocketService.disconnect(experimentId);
                        }
                        // Update experiment status
                        setExperiments(prev => prev.map(exp => 
                            exp.id === experimentId ? { ...exp, status: data.status } : exp
                        ));
                    },
                    onError: (error) => {
                        console.error('❌ [Experiment] WebSocket error:', error);
                    },
                    onClose: () => {
                        console.log('🔒 [Experiment] WebSocket closed');
                    },
                    onStatusChange: (status, oldStatus) => {
                        console.log(`🔄 [Experiment] Connection: ${oldStatus} → ${status}`);
                    },
                    enablePolling: true,
                    pollCallback: async (expId) => {
                        try {
                            const exp = await api.getFineTuningExperiment(expId);
                            if (exp.status === 'completed' || exp.status === 'failed') {
                                setExperiments(prev => prev.map(e => 
                                    e.id === expId ? { ...e, status: exp.status } : e
                                ));
                            }
                        } catch (error) {
                            console.error('[Experiment] Polling error:', error);
                        }
                    }
                });
                setWebSocket(ws);
            }
            
            // Reset form
            setExperimentName('');
            setExperimentDescription('');
            setSelectedModelName('');
            setSelectedDatasetId('');
            
            await loadExperiments();
        } catch (err) {
            console.error('❌ [Experiment] Failed to create:', err);
            setError('Failed to create experiment: ' + (err.message || 'Unknown error'));
        } finally {
            setLoading(false);
        }
    };

    const handleStartTraining = async (experimentId) => {
        try {
            // Set up WebSocket for training updates - use 'finetuning' endpoint type
            const ws = webSocketService.connect(experimentId, {
                endpointType: 'finetuning', // Use finetuning WebSocket endpoint
                onMessage: (data) => {
                    console.log('📨 [Training] WebSocket message:', data);
                    if (data.status === 'completed') {
                        setSuccess('Training completed successfully!');
                    } else if (data.status === 'failed') {
                        setError('Training failed: ' + (data.error_message || 'Unknown error'));
                    }
                    
                    // Update experiment status
                    setExperiments(prev => prev.map(exp => 
                        exp.id === experimentId ? { ...exp, status: data.status } : exp
                    ));
                },
                onError: (error) => {
                    console.error('❌ [Training] WebSocket error:', error);
                    setError('WebSocket connection error.');
                },
                onClose: () => {
                    console.log('🔒 [Training] WebSocket closed');
                    loadExperiments(); // Refresh experiments when WebSocket closes
                },
                onStatusChange: (status, oldStatus) => {
                    console.log(`🔄 [Training] Connection status changed: ${oldStatus} → ${status}`);
                },
                enablePolling: true,
                pollCallback: async (expId) => {
                    // HTTP fallback polling for training status
                    try {
                        const exp = await api.getFineTuningExperiment(expId);
                        if (exp.status === 'completed' || exp.status === 'failed') {
                            setExperiments(prev => prev.map(e => 
                                e.id === expId ? { ...e, status: exp.status } : e
                            ));
                        }
                    } catch (error) {
                        console.error('[Training] Polling error:', error);
                    }
                }
            });
            
            setWebSocket(ws);
            
            await api.startFineTuning(experimentId);
            setSuccess('Training started successfully!');
            await loadExperiments();
        } catch (err) {
            setError('Failed to start training: ' + (err.message || 'Unknown error'));
        }
    };

    const handleStopTraining = async (experimentId) => {
        try {
            await api.stopFineTuning(experimentId);
            setSuccess('Training stopped successfully!');
            
            if (webSocket) {
                webSocket.close();
                setWebSocket(null);
            }
            
            await loadExperiments();
        } catch (err) {
            setError('Failed to stop training: ' + (err.message || 'Unknown error'));
        }
    };

    const handleDeleteExperiment = async (experimentId) => {
        if (!window.confirm('Are you sure you want to delete this experiment?')) {
            return;
        }
        
        try {
            await api.deleteExperiment(experimentId);
            setSuccess('Experiment deleted successfully!');
            await loadExperiments();
        } catch (err) {
            setError('Failed to delete experiment: ' + (err.message || 'Unknown error'));
        }
    };

    const handleOpenDashboard = (experimentId) => {
        setSelectedTrainingExperiment(experimentId);
        setShowTrainingDashboard(true);
    };

    const handleCreateDataset = async () => {
        if (!newDatasetName || selectedDocuments.length === 0) {
            setError('Please provide a dataset name and select at least one document');
            return;
        }

        try {
            const payload = {
                name: newDatasetName,
                description: newDatasetDescription,
                document_ids: selectedDocuments.map(id => String(id)),  // Convert to strings
                questions_per_document: questionsPerDoc
            };

            console.log('📊 [QCA] Creating dataset with payload:', payload);
            const result = await api.createQCADataset(payload);
            console.log('📊 [QCA] Create dataset result:', result);
            
            setSuccess(`Dataset creation started! Task ID: ${result.task_id}`);
            setCreateDatasetDialogOpen(false);
            
            // Reset form
            setNewDatasetName('');
            setNewDatasetDescription('');
            setSelectedDocuments([]);
            setQuestionsPerDoc(5);
            
            // Use WebSocket as primary, with HTTP fallback (following EvaluationPage pattern)
            if (result.task_id) {
                console.log('🔌 [QCA] About to create WebSocket connection for task:', result.task_id);
                createQCAWebSocketConnection(result.task_id);
                console.log('🔌 [QCA] WebSocket connection initiated');
            } else {
                console.warn('⚠️ [QCA] No task_id in result, cannot create WebSocket connection');
            }
            
            await loadDatasets();
        } catch (err) {
            console.error('❌ [QCA] Failed to create dataset:', err);
            setError('Failed to create dataset: ' + (err.message || 'Unknown error'));
        }
    };

    // Create WebSocket connection for Q-C-A dataset creation (following EvaluationPage pattern)
    const createQCAWebSocketConnection = (taskId) => {
        // Check if we already have an active connection
        if (qcaActiveConnections.has(taskId)) {
            console.log(`🔄 Q-C-A WebSocket connection already exists for task ${taskId}`);
            return null;
        }

        console.log(`🔌 Creating Q-C-A WebSocket connection for task ${taskId}`);

        // Add to active connections
        setQcaActiveConnections(prev => new Set(prev).add(taskId));

        // Create connection using WebSocket service with 'qca-dataset' endpoint type
        const connection = webSocketService.connect(taskId, {
            endpointType: 'qca-dataset', // Use Q-C-A dataset endpoint
            onMessage: (message) => {
                handleQCAWebSocketMessage(taskId, message);
            },
            onError: (error) => {
                console.error(`❌ Q-C-A WebSocket error for task ${taskId}:`, error);
                handleQCAWebSocketError(taskId, error);
            },
            onClose: (event) => {
                console.log(`🔒 Q-C-A WebSocket closed for task ${taskId}:`, event?.code, event?.reason);
                handleQCAWebSocketClose(taskId, event);
            },
            onStatusChange: (status, oldStatus) => {
                console.log(`🔄 Q-C-A connection status changed for task ${taskId}: ${oldStatus} → ${status}`);
                setQcaConnectionStatuses(prev => new Map(prev).set(taskId, status));
            },
            enablePolling: true,
            pollCallback: async (taskId) => {
                // HTTP fallback polling for Q-C-A status
                try {
                    const status = await api.getQCADatasetStatus(taskId);
                    // Convert polling response to WebSocket message format
                    if (status.status === 'SUCCESS') {
                        handleQCAWebSocketMessage(taskId, { status: 'SUCCESS', data: status.data });
                    } else if (status.status === 'FAILURE' || status.status === 'ERROR') {
                        handleQCAWebSocketMessage(taskId, { status: 'FAILURE', error: status.data?.message || 'Unknown error' });
                    } else if (status.data?.progress !== undefined) {
                        handleQCAWebSocketMessage(taskId, { 
                            status: 'PROGRESS', 
                            progress: status.data.progress,
                            message: status.data.message || 'Processing...'
                        });
                    }
                } catch (error) {
                    console.error('Q-C-A polling error:', error);
                }
            }
        });

        return connection;
    };

    // Handle Q-C-A WebSocket messages
    const handleQCAWebSocketMessage = async (taskId, message) => {
        console.log(`📨 Q-C-A WebSocket message for task ${taskId}:`, message);

        const status = message.status || message.type;
        const progress = message.progress || message.data?.progress || 0;
        const messageText = message.message || message.data?.message || '';

        // Update progress state
        setQcaProgress(prev => new Map(prev).set(taskId, {
            status,
            progress,
            message: messageText,
            timestamp: new Date().toISOString()
        }));

        if (status === 'SUCCESS' || status === 'COMPLETED') {
            setSuccess('Q-C-A Dataset created successfully!');
            // Disconnect WebSocket and cleanup
            webSocketService.disconnect(taskId);
            setQcaActiveConnections(prev => {
                const newSet = new Set(prev);
                newSet.delete(taskId);
                return newSet;
            });
            // Reload datasets to show the new one
            await loadDatasets();
        } else if (status === 'FAILURE' || status === 'ERROR') {
            setError(`Q-C-A Dataset creation failed: ${message.error || messageText || 'Unknown error'}`);
            // Disconnect WebSocket and cleanup
            webSocketService.disconnect(taskId);
            setQcaActiveConnections(prev => {
                const newSet = new Set(prev);
                newSet.delete(taskId);
                return newSet;
            });
            await loadDatasets();
        }
    };

    // Handle Q-C-A WebSocket errors
    const handleQCAWebSocketError = (taskId, error) => {
        console.error(`Q-C-A WebSocket error for ${taskId}:`, error);
        // WebSocket service will auto-reconnect or fall back to polling
    };

    // Handle Q-C-A WebSocket close
    const handleQCAWebSocketClose = (taskId, event) => {
        console.log(`Q-C-A WebSocket closed for ${taskId}:`, event?.code);
        // Cleanup if this was a normal close after completion
        if (event?.code === 1000) {
            setQcaActiveConnections(prev => {
                const newSet = new Set(prev);
                newSet.delete(taskId);
                return newSet;
            });
        }
    };

    const handleViewDataset = async (datasetId) => {
        try {
            const data = await api.getDatasetDetails(datasetId);
            setDatasetDetails(data);
            setViewDatasetDialogOpen(true);
        } catch (err) {
            setError('Failed to load dataset details: ' + (err.message || 'Unknown error'));
        }
    };

    const getStatusChip = (status) => {
        if (!status) status = 'unknown';
        
        const statusColors = {
            pending: 'default',
            running: 'primary',
            processing: 'primary',
            completed: 'success',
            failed: 'error',
            stopped: 'warning',
            created: 'success',
            unknown: 'default'
        };
        
        const isProcessing = ['processing', 'running', 'pending'].includes(status.toLowerCase());
        
        return (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                {isProcessing && (
                    <CircularProgress 
                        size={12} 
                        sx={{ 
                            color: statusColors[status] === 'primary' ? '#1976d2' : '#666',
                        }} 
                    />
                )}
                <Chip 
                    label={String(status).charAt(0).toUpperCase() + String(status).slice(1)} 
                    color={statusColors[status] || 'default'}
                    size="small"
                />
            </Box>
        );
    };

    const formatDate = (dateString) => {
        if (!dateString) return 'N/A';
        try {
            return new Date(dateString).toLocaleString();
        } catch (error) {
            return 'Invalid Date';
        }
    };

    // Calculate metrics
    const metrics = {
        total_experiments: experiments.length,
        running_experiments: experiments.filter(exp => exp.status === 'running').length,
        completed_experiments: experiments.filter(exp => exp.status === 'completed').length,
        failed_experiments: experiments.filter(exp => exp.status === 'failed').length
    };

    if (showTrainingDashboard) {
        return (
            <TrainingDashboard
                experimentId={selectedTrainingExperiment}
                trainingUpdate={trainingUpdates}
                onClose={() => {
                    setShowTrainingDashboard(false);
                    setSelectedTrainingExperiment(null);
                    setTrainingUpdates(null);
                }}
            />
        );
    }

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
            {/* Error/Success Messages */}
            {error && (
                <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
                    {String(error)}
                </Alert>
            )}
            
            {success && (
                <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess(null)}>
                    {String(success)}
                </Alert>
            )}

            <Container maxWidth="xl" sx={{ py: 4 }}>
                <Typography variant="h4" gutterBottom>
                    Fine-tuning Management
                </Typography>

                {/* Tabs */}
                <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
                    <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
                        <Tab label="Experiments" />
                        <Tab label="Datasets" />
                        <Tab label="Training" />
                        <Tab label="Comparison" />
                    </Tabs>
                </Box>

                {/* Tab Content */}
                {activeTab === 0 && (
                    <Box>
                        {/* Metrics Cards */}
                        <Grid container spacing={3} sx={{ mb: 4 }}>
                            <Grid item xs={12} sm={6} md={3}>
                                <Card>
                                    <CardContent>
                                        <Typography color="textSecondary" gutterBottom>
                                            Total Experiments
                                        </Typography>
                                        <Typography variant="h4">
                                            {metrics.total_experiments}
                                        </Typography>
                                    </CardContent>
                                </Card>
                            </Grid>
                            <Grid item xs={12} sm={6} md={3}>
                                <Card>
                                    <CardContent>
                                        <Typography color="textSecondary" gutterBottom>
                                            Running
                                        </Typography>
                                        <Typography variant="h4" color="primary">
                                            {metrics.running_experiments}
                                        </Typography>
                                    </CardContent>
                                </Card>
                            </Grid>
                            <Grid item xs={12} sm={6} md={3}>
                                <Card>
                                    <CardContent>
                                        <Typography color="textSecondary" gutterBottom>
                                            Completed
                                        </Typography>
                                        <Typography variant="h4" color="success.main">
                                            {metrics.completed_experiments}
                                        </Typography>
                                    </CardContent>
                                </Card>
                            </Grid>
                            <Grid item xs={12} sm={6} md={3}>
                                <Card>
                                    <CardContent>
                                        <Typography color="textSecondary" gutterBottom>
                                            Failed
                                        </Typography>
                                        <Typography variant="h4" color="error.main">
                                            {metrics.failed_experiments}
                                        </Typography>
                                    </CardContent>
                                </Card>
                            </Grid>
                        </Grid>

                        {/* Create Experiment Form */}
                        <Card sx={{ mb: 4 }}>
                            <CardContent>
                                <Typography variant="h6" gutterBottom>
                                    Create New Experiment
                                </Typography>
                                
                                <Grid container spacing={2}>
                                    <Grid item xs={12} md={6}>
                                        <TextField
                                            fullWidth
                                            label="Experiment Name"
                                            value={experimentName}
                                            onChange={(e) => setExperimentName(e.target.value)}
                                        />
                                    </Grid>
                                    <Grid item xs={12} md={6}>
                                        <FormControl fullWidth>
                                            <InputLabel>Model</InputLabel>
                                            <Select
                                                value={selectedModelName}
                                                onChange={(e) => setSelectedModelName(e.target.value)}
                                            >
                                                {availableModels.map((model) => (
                                                    <MenuItem key={model.name} value={model.name}>
                                                        {model.name}
                                                    </MenuItem>
                                                ))}
                                            </Select>
                                        </FormControl>
                                    </Grid>
                                    <Grid item xs={12}>
                                        <TextField
                                            fullWidth
                                            label="Description"
                                            multiline
                                            rows={2}
                                            value={experimentDescription}
                                            onChange={(e) => setExperimentDescription(e.target.value)}
                                        />
                                    </Grid>
                                    <Grid item xs={12} md={6}>
                                        <FormControl fullWidth>
                                            <InputLabel>Dataset</InputLabel>
                                            <Select
                                                value={selectedDatasetId}
                                                onChange={(e) => setSelectedDatasetId(e.target.value)}
                                            >
                                                {datasets.map((dataset) => (
                                                    <MenuItem key={dataset.id} value={dataset.id}>
                                                        {dataset.name} - {dataset.description || 'No description'}
                                                    </MenuItem>
                                                ))}
                                            </Select>
                                        </FormControl>
                                    </Grid>
                                    <Grid item xs={12} md={6}>
                                        <TextField
                                            fullWidth
                                            label="Learning Rate"
                                            type="number"
                                            value={learningRate}
                                            onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                                            inputProps={{ step: 0.0001, min: 0 }}
                                        />
                                    </Grid>
                                    <Grid item xs={12} md={3}>
                                        <TextField
                                            fullWidth
                                            label="Epochs"
                                            type="number"
                                            value={numEpochs}
                                            onChange={(e) => setNumEpochs(parseInt(e.target.value))}
                                            inputProps={{ min: 1 }}
                                        />
                                    </Grid>
                                    <Grid item xs={12} md={3}>
                                        <TextField
                                            fullWidth
                                            label="Batch Size"
                                            type="number"
                                            value={batchSize}
                                            onChange={(e) => setBatchSize(parseInt(e.target.value))}
                                            inputProps={{ min: 1 }}
                                        />
                                    </Grid>
                                    <Grid item xs={12} md={3}>
                                        <TextField
                                            fullWidth
                                            label="LoRA R"
                                            type="number"
                                            value={loraR}
                                            onChange={(e) => setLoraR(parseInt(e.target.value))}
                                            inputProps={{ min: 1 }}
                                        />
                                    </Grid>
                                    <Grid item xs={12} md={3}>
                                        <TextField
                                            fullWidth
                                            label="LoRA Alpha"
                                            type="number"
                                            value={loraAlpha}
                                            onChange={(e) => setLoraAlpha(parseInt(e.target.value))}
                                            inputProps={{ min: 1 }}
                                        />
                                    </Grid>
                                    <Grid item xs={12}>
                                        <Button
                                            variant="contained"
                                            onClick={handleCreateExperiment}
                                            disabled={loading}
                                            startIcon={loading ? <CircularProgress size={20} /> : <AddIcon />}
                                        >
                                            Create Experiment
                                        </Button>
                                    </Grid>
                                </Grid>
                            </CardContent>
                        </Card>

                        {/* Experiments Table */}
                        <Card>
                            <CardContent>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                    <Typography variant="h6">Experiments</Typography>
                                    <Button
                                        variant="outlined"
                                        startIcon={<RefreshIcon />}
                                        onClick={loadExperiments}
                                        disabled={loading}
                                    >
                                        Refresh
                                    </Button>
                                </Box>
                                
                                <TableContainer component={Paper}>
                                    <Table>
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Name</TableCell>
                                                <TableCell>Model</TableCell>
                                                <TableCell>Status</TableCell>
                                                <TableCell>Created</TableCell>
                                                <TableCell>Actions</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {experiments.map((experiment) => (
                                                <TableRow key={experiment.id}>
                                                    <TableCell>
                                                        <Typography variant="subtitle2">
                                                            {experiment.name || 'Unnamed'}
                                                        </Typography>
                                                        <Typography variant="body2" color="textSecondary">
                                                            {experiment.description || 'No description'}
                                                        </Typography>
                                                    </TableCell>
                                                    <TableCell>{experiment.model_name || 'Unknown'}</TableCell>
                                                    <TableCell>{getStatusChip(experiment.status)}</TableCell>
                                                    <TableCell>{formatDate(experiment.created_at)}</TableCell>
                                                    <TableCell>
                                                        <Box sx={{ display: 'flex', gap: 1 }}>
                                                            {experiment.status === 'pending' && (
                                                                <Tooltip title="Start Training">
                                                                    <IconButton
                                                                        size="small"
                                                                        onClick={() => handleStartTraining(experiment.id)}
                                                                        color="primary"
                                                                    >
                                                                        <PlayIcon />
                                                                    </IconButton>
                                                                </Tooltip>
                                                            )}
                                                            
                                                            {experiment.status === 'running' && (
                                                                <>
                                                                    <Tooltip title="View Dashboard">
                                                                        <IconButton
                                                                            size="small"
                                                                            onClick={() => handleOpenDashboard(experiment.id)}
                                                                            color="primary"
                                                                        >
                                                                            <ViewIcon />
                                                                        </IconButton>
                                                                    </Tooltip>
                                                                    <Tooltip title="Stop Training">
                                                                        <IconButton
                                                                            size="small"
                                                                            onClick={() => handleStopTraining(experiment.id)}
                                                                            color="warning"
                                                                        >
                                                                            <StopIcon />
                                                                        </IconButton>
                                                                    </Tooltip>
                                                                </>
                                                            )}
                                                            
                                                            <Tooltip title="Delete">
                                                                <IconButton
                                                                    size="small"
                                                                    onClick={() => handleDeleteExperiment(experiment.id)}
                                                                    color="error"
                                                                >
                                                                    <DeleteIcon />
                                                                </IconButton>
                                                            </Tooltip>
                                                        </Box>
                                                    </TableCell>
                                                </TableRow>
                                            ))}
                                        </TableBody>
                                    </Table>
                                </TableContainer>
                            </CardContent>
                        </Card>
                    </Box>
                )}

                {activeTab === 1 && (
                    <Box>
                        <Typography variant="h6" gutterBottom>
                            Dataset Management
                        </Typography>
                        
                        <Button
                            variant="contained"
                            startIcon={<AddIcon />}
                            onClick={() => setCreateDatasetDialogOpen(true)}
                            sx={{ mb: 2 }}
                        >
                            Create Dataset
                        </Button>
                        
                        <TableContainer component={Paper}>
                            <Table>
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Name</TableCell>
                                        <TableCell>Description</TableCell>
                                        <TableCell>Samples</TableCell>
                                        <TableCell>Status</TableCell>
                                        <TableCell>Created</TableCell>
                                        <TableCell>Actions</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {datasets.map((dataset) => (
                                        <TableRow key={dataset.id}>
                                            <TableCell>{dataset.name || 'Unnamed'}</TableCell>
                                            <TableCell>{dataset.description || 'No description'}</TableCell>
                                            <TableCell>{dataset.num_samples || 0} samples</TableCell>
                                            <TableCell>{getStatusChip(dataset.status || 'completed')}</TableCell>
                                            <TableCell>{formatDate(dataset.created_at)}</TableCell>
                                            <TableCell>
                                                <Tooltip title="View Dataset">
                                                    <IconButton
                                                        size="small"
                                                        onClick={() => handleViewDataset(dataset.id)}
                                                    >
                                                        <ViewIcon />
                                                    </IconButton>
                                                </Tooltip>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    </Box>
                )}

                {activeTab === 2 && (
                    <Box>
                        <Typography variant="h6" gutterBottom>
                            Active Training Sessions
                        </Typography>
                        
                        {experiments.filter(exp => exp.status === 'running').length === 0 ? (
                            <Alert severity="info">
                                No active training sessions. Start an experiment to see it here.
                            </Alert>
                        ) : (
                            <Grid container spacing={3}>
                                {experiments.filter(exp => exp.status === 'running').map((experiment) => (
                                    <Grid item xs={12} md={6} key={experiment.id}>
                                        <Card>
                                            <CardContent>
                                                <Typography variant="h6">{experiment.name}</Typography>
                                                <Typography variant="body2" color="textSecondary">
                                                    Model: {experiment.model_name}
                                                </Typography>
                                                <Box sx={{ mt: 2 }}>
                                                    <Button
                                                        variant="outlined"
                                                        startIcon={<ViewIcon />}
                                                        onClick={() => handleOpenDashboard(experiment.id)}
                                                    >
                                                        Open Dashboard
                                                    </Button>
                                                </Box>
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                ))}
                            </Grid>
                        )}
                    </Box>
                )}

                {activeTab === 3 && (
                    <ExperimentComparison availableExperiments={experiments} />
                )}
            </Container>

            {/* Create Dataset Dialog */}
            <Dialog 
                open={createDatasetDialogOpen} 
                onClose={() => setCreateDatasetDialogOpen(false)} 
                maxWidth="md" 
                fullWidth
            >
                <DialogTitle>Create New Dataset</DialogTitle>
                <DialogContent>
                    <Grid container spacing={2} sx={{ mt: 1 }}>
                        <Grid item xs={12}>
                            <TextField
                                fullWidth
                                label="Dataset Name"
                                value={newDatasetName}
                                onChange={(e) => setNewDatasetName(e.target.value)}
                            />
                        </Grid>
                        <Grid item xs={12}>
                            <TextField
                                fullWidth
                                label="Description"
                                multiline
                                rows={3}
                                value={newDatasetDescription}
                                onChange={(e) => setNewDatasetDescription(e.target.value)}
                            />
                        </Grid>
                        <Grid item xs={12}>
                            <FormControl fullWidth>
                                <InputLabel>Documents</InputLabel>
                                <Select
                                    multiple
                                    value={selectedDocuments}
                                    onChange={(e) => setSelectedDocuments(e.target.value)}
                                >
                                    {availableDocuments.map((doc) => (
                                        <MenuItem key={doc.id} value={doc.id}>
                                            {doc.filename}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Grid>
                        <Grid item xs={12}>
                            <TextField
                                fullWidth
                                label="Questions Per Document"
                                type="number"
                                value={questionsPerDoc}
                                onChange={(e) => setQuestionsPerDoc(parseInt(e.target.value))}
                                inputProps={{ min: 1, max: 20 }}
                            />
                        </Grid>
                    </Grid>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setCreateDatasetDialogOpen(false)}>Cancel</Button>
                    <Button 
                        onClick={handleCreateDataset}
                        variant="contained"
                        disabled={!newDatasetName || selectedDocuments.length === 0}
                    >
                        Create Dataset
                    </Button>
                </DialogActions>
            </Dialog>

            {/* View Dataset Dialog */}
            <Dialog 
                open={viewDatasetDialogOpen} 
                onClose={() => setViewDatasetDialogOpen(false)} 
                maxWidth="lg" 
                fullWidth
            >
                <DialogTitle>Dataset Details</DialogTitle>
                <DialogContent>
                    {datasetDetails && (
                        <Box>
                            <Typography><strong>Name:</strong> {datasetDetails.name}</Typography>
                            <Typography><strong>Description:</strong> {datasetDetails.description}</Typography>
                            <Typography><strong>Samples:</strong> {datasetDetails.num_samples}</Typography>
                            <Typography><strong>Created:</strong> {formatDate(datasetDetails.created_at)}</Typography>
                        </Box>
                    )}
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setViewDatasetDialogOpen(false)}>Close</Button>
                </DialogActions>
            </Dialog>

            {/* Snackbar Notifications */}
            <Snackbar
                open={!!error}
                autoHideDuration={6000}
                onClose={() => setError(null)}
            >
                <Alert severity="error" onClose={() => setError(null)}>
                    {String(error)}
                </Alert>
            </Snackbar>

            <Snackbar
                open={!!success}
                autoHideDuration={4000}
                onClose={() => setSuccess(null)}
            >
                <Alert severity="success" onClose={() => setSuccess(null)}>
                    {String(success)}
                </Alert>
            </Snackbar>
        </Box>
    );
};

export default FinetuningPage;