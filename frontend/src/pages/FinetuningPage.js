import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
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
    Container
} from '@mui/material';
import {
    PlayArrow as PlayIcon,
    Stop as StopIcon,
    Delete as DeleteIcon,
    CloudUpload as UploadIcon,
    Info as InfoIcon,
    Refresh as RefreshIcon,
    Add as AddIcon,
    Edit as EditIcon,
    ArrowBack,
    ModelTraining,
    Dashboard,
    DataUsage,
    Assignment,
    Timeline,
    Close,
    Visibility as ViewIcon,
    GetApp as DownloadIcon,
    Transform as TransformIcon
} from '@mui/icons-material';
import api from '../services/api';
import { evaluationApi } from '../services/api';
import webSocketService from '../services/websocketService';
import TrainingDashboard from '../components/TrainingDashboard';
import ExperimentComparison from '../components/ExperimentComparison';

const SIDEBAR_WIDTH = 280;

const FinetuningPage = () => {
    const navigate = useNavigate();
    const [activeTab, setActiveTab] = useState(0); // 0: Experiments, 1: Datasets, 2: Training, 3: Comparison
    
    // Training Dashboard State
    const [selectedTrainingExperiment, setSelectedTrainingExperiment] = useState(null);
    const [showTrainingDashboard, setShowTrainingDashboard] = useState(false);
    
    // Experiment Setup State
    const [experiments, setExperiments] = useState([]);
    const [availableModels, setAvailableModels] = useState([]);
    const [datasets, setDatasets] = useState([]);
    const [evaluationDatasets, setEvaluationDatasets] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);
    
    // Individual form state variables (like EvaluationPage pattern)
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

    // Stable input refs to preserve focus/caret across renders
    const nameInputRef = useRef(null);
    const descInputRef = useRef(null);
    
    // Dataset Management State
    const [selectedDataset, setSelectedDataset] = useState(null);
    const [selectedEvaluationDataset, setSelectedEvaluationDataset] = useState(null);
    const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
    const [createDatasetDialogOpen, setCreateDatasetDialogOpen] = useState(false);
    const [viewDatasetDialogOpen, setViewDatasetDialogOpen] = useState(false);
    const [datasetDetails, setDatasetDetails] = useState(null);
    
    // Create Dataset State
    const [newDatasetName, setNewDatasetName] = useState('');
    const [newDatasetDescription, setNewDatasetDescription] = useState('');
    const [selectedDocuments, setSelectedDocuments] = useState([]);
    const [availableDocuments, setAvailableDocuments] = useState([]);
    const [questionsPerDoc, setQuestionsPerDoc] = useState(5);
    
    // Dialog and UI State
    const [openDialog, setOpenDialog] = useState(false);
    const [selectedExperiment, setSelectedExperiment] = useState(null);
    const [webSocket, setWebSocket] = useState(null);
    const [trainingLogs, setTrainingLogs] = useState([]);
    
    // WebSocket State for Q-C-A Dataset Creation
    const [activeConnections, setActiveConnections] = useState(() => new Set());
    const [connectionStatuses, setConnectionStatuses] = useState(() => new Map());
    const [datasetCreationProgress, setDatasetCreationProgress] = useState(() => new Map());
    
    // Memoized metrics calculation - only recalculates when experiments change
    const metrics = useMemo(() => {
        const total = experiments?.length || 0;
        const running = experiments?.filter(exp => exp.status === 'running').length || 0;
        const completed = experiments?.filter(exp => exp.status === 'completed').length || 0;
        const failed = experiments?.filter(exp => exp.status === 'failed').length || 0;
        
        return {
            total_experiments: total,
            running_experiments: running,
            completed_experiments: completed,
            failed_experiments: failed
        };
    }, [experiments]);

    // Memoized filtered experiment lists - prevent re-filtering on every render
    const runningExperiments = useMemo(() => 
        experiments.filter(exp => exp.status === 'running'), [experiments]
    );
    
    const finishedExperiments = useMemo(() => 
        experiments.filter(exp => exp.status === 'completed' || exp.status === 'failed'), [experiments]
    );

    // Load initial data
    useEffect(() => {
        loadInitialData();
    }, []);

    // Cleanup WebSocket connections on unmount
    useEffect(() => {
        return () => {
            // Disconnect all active Q-C-A WebSocket connections
            activeConnections.forEach(taskId => {
                webSocketService.disconnect(taskId);
            });
        };
    }, [activeConnections]);

    const loadInitialData = useCallback(async () => {
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
            setError('Failed to load initial data: ' + err.message);
        } finally {
            setLoading(false);
        }
    }, [loadExperiments, loadAvailableModels, loadDatasets, loadEvaluationDatasets, loadAvailableDocuments]);

    const loadExperiments = useCallback(async () => {
        try {
            const data = await api.getFineTuningExperiments();
            setExperiments(data.experiments || []);
        } catch (err) {
            console.error('Failed to load experiments:', err);
            setError('Failed to load experiments: ' + err.message);
        }
    }, []);

    const loadAvailableModels = useCallback(async () => {
        try {
            const data = await api.getAvailableHFModels();
            setAvailableModels(data.models || []);
        } catch (err) {
            console.error('Failed to load models:', err);
            setError('Failed to load available models: ' + err.message);
        }
    }, []);

    const loadDatasets = useCallback(async () => {
        try {
            const data = await api.getFineTuningDatasets();
            setDatasets(data.datasets || []);
        } catch (err) {
            console.error('Failed to load datasets:', err);
            setError('Failed to load datasets: ' + err.message);
        }
    }, []);

    const loadEvaluationDatasets = useCallback(async () => {
        try {
            const data = await evaluationApi.getDatasets();
            setEvaluationDatasets(data.datasets || []);
        } catch (err) {
            console.error('Failed to load evaluation datasets:', err);
            setError('Failed to load evaluation datasets: ' + err.message);
        }
    }, []);

    const loadAvailableDocuments = useCallback(async () => {
        try {
            const data = await api.getDocuments();
            setAvailableDocuments(data.documents || []);
        } catch (err) {
            console.error('Failed to load documents:', err);
            setError('Failed to load documents: ' + err.message);
        }
    }, []);

    const updateExperimentStatus = (experimentId, status) => {
        setExperiments((prev) => prev.map((exp) =>
            exp.id === experimentId ? { ...exp, status } : exp
        ));
    };

    const handleCreateExperiment = useCallback(async () => {
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
            
            // Add default values for advanced parameters (not shown in UI)
            formData.append('max_seq_length', '512');
            formData.append('warmup_ratio', '0.03');
            formData.append('weight_decay', '0.01');
            formData.append('gradient_accumulation_steps', '1');
            formData.append('logging_steps', '10');
            formData.append('save_steps', '500');
            formData.append('eval_steps', '100');
            formData.append('save_total_limit', '2');
            formData.append('load_best_model_at_end', 'true');
            formData.append('metric_for_best_model', 'eval_loss');
            formData.append('greater_is_better', 'false');
            formData.append('evaluation_strategy', 'steps');
            formData.append('save_strategy', 'steps');

            const result = await api.createFineTuningExperiment(formData);
            setSuccess(`Experiment "${result?.experiment?.name || 'New experiment'}" created successfully!`);
            
            // Immediately setup WebSocket for the new experiment (backend auto-starts training)
            const newExperimentId = result?.experiment?.id || result?.experiment?.experiment_id || result?.id;
            if (newExperimentId) {
                // Close any existing WebSocket
                if (webSocket) {
                    try { webSocket.close(); } catch {}
                    setWebSocket(null);
                }
                
                const ws = api.createFineTuningWebSocket(
                    newExperimentId,
                    (data) => {
                        // Training progress update received
                        const status = typeof data.status === 'string' ? data.status.toLowerCase() : data.status;
                        
                        // Update experiment status optimistically
                        throttledStatusUpdate(newExperimentId, status);
                        
                        // If dashboard is open for this experiment, update its metrics
                        if (selectedTrainingExperiment === newExperimentId) {
                            setTrainingMetrics(data);
                        }
                        
                        if (status === 'completed') {
                            debouncedSetSuccess('Training completed successfully!');
                            if (ws) ws.close();
                            setWebSocket(null);
                        } else if (status === 'failed' || status === 'cancelled') {
                            debouncedSetError('Training failed: ' + (data.error_message || 'Unknown error'));
                            if (ws) ws.close();
                            setWebSocket(null);
                        }
                    },
                    (error) => {
                        console.error('WebSocket error:', error);
                        debouncedSetError('WebSocket connection error.');
                    },
                    async () => {
                        // WebSocket closed for experiment
                        setWebSocket(null);
                        try { await loadExperiments(); } catch {}
                    }
                );
                setWebSocket(ws);
            }
            
            // Reset form to default values
            setExperimentName('');
            setExperimentDescription('');
            setSelectedModelName('');
            setSelectedDatasetId('');
            setLearningRate(0.0001);
            setNumEpochs(3);
            setBatchSize(4);
            setLoraR(16);
            setLoraAlpha(32);
            setLoraDropout(0.1);
            setTargetModules('q_proj,v_proj');
            
            // Reload experiments
            await loadExperiments();
        } catch (err) {
            debouncedSetError('Failed to create experiment: ' + err.message);
        } finally {
            setLoading(false);
        }
    }, [
        experimentName, selectedModelName, selectedDatasetId, experimentDescription, 
        learningRate, numEpochs, batchSize, loraR, loraAlpha, loraDropout, targetModules,
        webSocket, setWebSocket, loadExperiments
    ]);

    const handleStartTraining = useCallback(async (experimentId) => {
        try {
            await api.startFineTuningTraining(experimentId);
            debouncedSetSuccess('Training started successfully!');
            
            // Setup WebSocket for progress monitoring
            const ws = api.createFineTuningWebSocket(
                experimentId,
                (data) => {
                    // Training progress update received
                    
                    // Normalize status to lowercase for consistency with backend
                    const status = typeof data.status === 'string' ? data.status.toLowerCase() : data.status;
                    
                    // Optimistically update the experiment status in the list
                    throttledStatusUpdate(experimentId, status);
                    
                    // If dashboard is open for this experiment, update its metrics
                    if (selectedTrainingExperiment === experimentId) {
                        setTrainingMetrics(data);
                    }
                    
                    // Handle terminal statuses
                    if (status === 'completed') {
                        debouncedSetSuccess('Training completed successfully!');
                        // Training completed
                        if (ws) ws.close();
                        setWebSocket(null);
                    } else if (status === 'failed' || status === 'cancelled') {
                        debouncedSetError('Training failed: ' + (data.error_message || 'Unknown error'));
                        // Training failed
                        if (ws) ws.close();
                        setWebSocket(null);
                    }
                },
                (error) => {
                    console.error('WebSocket error:', error);
                    debouncedSetError('Real-time updates disconnected');
                },
                async () => {
                    // WebSocket connection closed
                    setWebSocket(null);
                    // Ensure final state is reflected
                    try { await loadExperiments(); } catch {}
                }
            );
            setWebSocket(ws);
            
            await loadExperiments();
        } catch (err) {
            debouncedSetError('Failed to start training: ' + err.message);
        }
    }, [webSocket, setWebSocket, loadExperiments]);

    const handleStopTraining = async (experimentId) => {
        try {
            await api.stopFineTuningTraining(experimentId);
            setSuccess('Training stopped successfully!');
            
            if (webSocket) {
                webSocket.close();
                setWebSocket(null);
            }
            
            await loadExperiments();
        } catch (err) {
            setError('Failed to stop training: ' + err.message);
        }
    };

    const handleDeleteExperiment = async (experimentId) => {
        if (!window.confirm('Are you sure you want to delete this experiment?')) {
            return;
        }
        
        try {
            await api.deleteFineTuningExperiment(experimentId);
            setSuccess('Experiment deleted successfully!');
            await loadExperiments();
        } catch (err) {
            setError('Failed to delete experiment: ' + err.message);
        }
    };

    const handleOpenDashboard = useCallback(async (experimentId) => {
        setTrainingMetrics(null); // Clear old metrics
        setSelectedTrainingExperiment(experimentId);
        setShowTrainingDashboard(true);
        
        // Fetch initial metrics immediately
        try {
            const data = await api.getFineTuningMetrics(experimentId);
            setTrainingMetrics(data);
        } catch (err) {
            debouncedSetError('Failed to load initial training data.');
        }
    }, [debouncedSetError]);

    // Dataset Management Functions
    const handleUploadDataset = async (file) => {
        setLoading(true);
        try {
            const result = await api.validateFineTuningDataset(file);
            if (result.valid) {
                // If validation passes, actually upload it
                const name = file.name.replace('.jsonl', '').replace('.json', '');
                const description = `Uploaded dataset: ${name}`;
                
                await api.uploadFineTuningDataset(file, name, description);
                setSuccess('Dataset uploaded successfully!');
                await loadDatasets();
                setUploadDialogOpen(false);
            } else {
                setError('Dataset validation failed: ' + result.message);
            }
        } catch (err) {
            setError('Failed to upload dataset: ' + err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleCreateDataset = async () => {
        setLoading(true);
        try {
            const response = await api.createFinetuningDataset({
                name: newDatasetName,
                description: newDatasetDescription,
                document_ids: selectedDocuments,
                questions_per_doc: questionsPerDoc
            });
            
            // Create WebSocket connection for progress monitoring
            if (response.task_id) {
                createWebSocketConnection(response.task_id);
            }
            
            setSuccess('Dataset creation started! Check back in a few minutes.');
            await loadDatasets();
            setCreateDatasetDialogOpen(false);
            // Reset form
            setNewDatasetName('');
            setNewDatasetDescription('');
            setSelectedDocuments([]);
            setQuestionsPerDoc(5);
        } catch (err) {
            setError('Failed to create dataset: ' + err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleViewDataset = async (datasetId) => {
        try {
            const data = await api.getFineTuningDatasetDetails(datasetId);
            setDatasetDetails(data);
            setViewDatasetDialogOpen(true);
        } catch (err) {
            setError('Failed to load dataset details: ' + err.message);
        }
    };

    const handleDeleteDataset = async (datasetId) => {
        if (!window.confirm('Are you sure you want to delete this dataset?')) {
            return;
        }
        
        try {
            await api.deleteFineTuningDataset(datasetId);
            setSuccess('Dataset deleted successfully!');
            await loadDatasets();
        } catch (err) {
            setError('Failed to delete dataset: ' + err.message);
        }
    };

    // WebSocket handling for Q-C-A Dataset Creation
    const createWebSocketConnection = (taskId) => {
        // Check if we already have an active connection
        if (activeConnections.has(taskId)) {
            // Q-C-A Connection already exists for task
            return null;
        }

        // Creating new Q-C-A WebSocket connection for task

        // Add to active connections
        setActiveConnections(prev => new Set(prev).add(taskId));

        // Create connection using WebSocket service with Q-C-A endpoint
        const connection = webSocketService.connect(taskId, {
            endpointType: 'qca-dataset', // Use Q-C-A endpoint instead of evaluation
            onMessage: (message) => {
                handleWebSocketMessage(taskId, message);
            },
            onError: (error) => {
                console.error(`❌ Q-C-A WebSocket error for task ${taskId}:`, error);
                handleWebSocketError(taskId, error);
            },
            onClose: (event) => {
                // Q-C-A WebSocket closed for task
                handleWebSocketClose(taskId, event);
            },
            onStatusChange: (status, oldStatus) => {
                // Q-C-A Connection status changed for task
                setConnectionStatuses(prev => new Map(prev).set(taskId, status));
            },
            enablePolling: true
        });

        return connection;
    };

    // Handle WebSocket messages for Q-C-A dataset creation
    const handleWebSocketMessage = (taskId, message) => {
        // Q-C-A WebSocket message received for task
        
        // Update progress
        if (message.progress !== undefined) {
            setDatasetCreationProgress(prev => new Map(prev).set(taskId, {
                progress: message.progress,
                status: message.status || 'running',
                message: message.message || '',
                timestamp: new Date().toISOString()
            }));
        }

        // Handle task completion
        if (message.status === 'SUCCESS') {
            handleDatasetCreationSuccess(taskId, message);
        } else if (message.status === 'FAILURE') {
            handleTaskFailure(taskId, message);
        } else {
            // For dataset tasks, refresh list on any status change
            // Q-C-A Dataset task update, refreshing datasets list
            loadDatasets();
        }
    };

    // Handle WebSocket errors
    const handleWebSocketError = (taskId, error) => {
        console.error(`❌ Q-C-A WebSocket error for task ${taskId}:`, error);
        // Error handling is managed by WebSocket service with auto-reconnection
    };

    // Handle WebSocket close
    const handleWebSocketClose = (taskId, event) => {
        // Q-C-A WebSocket closed for task
        // Remove from active connections
        setActiveConnections(prev => {
            const updated = new Set(prev);
            updated.delete(taskId);
            return updated;
        });
        
        // Clear connection status
        setConnectionStatuses(prev => {
            const updated = new Map(prev);
            updated.delete(taskId);
            return updated;
        });
    };

    // Handle dataset creation success
    const handleDatasetCreationSuccess = (taskId, message) => {
        // Q-C-A Dataset created successfully
        
        // Refresh datasets list multiple times to ensure we get the latest data
        const refreshDatasets = async () => {
            await loadDatasets();
            // Refresh again after a short delay to ensure backend DB is updated
            setTimeout(async () => {
                await loadDatasets();
            }, 1000);
        };
        refreshDatasets();

        // Show success message
        setError(''); // Clear any previous errors
        setSuccess('Dataset created successfully!');

        // Disconnect WebSocket after short delay
        setTimeout(() => {
            webSocketService.disconnect(taskId);
        }, 2000);
    };

    // Handle task failure
    const handleTaskFailure = (taskId, message) => {
        console.error('❌ Q-C-A Dataset creation failed:', message);
        setError(`Dataset creation failed: ${message.message || 'Unknown error'}`);
        
        // Disconnect WebSocket
        setTimeout(() => {
            webSocketService.disconnect(taskId);
        }, 1000);
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
        
        // Show spinner for processing/running states
        const isProcessing = ['processing', 'running', 'pending'].includes(status?.toLowerCase?.() || '');
        
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

    // Tab Content Components
    const ExperimentsTab = () => (
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
                                {metrics?.total_experiments || 0}
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
                                {metrics?.running_experiments || 0}
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
                                {metrics?.completed_experiments || 0}
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
                                {metrics?.failed_experiments || 0}
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>

            {/* Create New Experiment */}
            <Card sx={{ mb: 4 }}>
                <CardContent>
                    <Typography variant="h6" gutterBottom>
                        Create New Experiment
                    </Typography>
                    
                    <Grid container spacing={3}>
                        {/* Basic Information */}
                        <Grid item xs={12}>
                            <Typography variant="subtitle1" gutterBottom>
                                Basic Information
                            </Typography>
                        </Grid>
                        
                        <Grid item xs={12} md={6}>
                            <TextField
                                fullWidth
                                label="Experiment Name"
                                value={experimentName}
                                inputRef={nameInputRef}
                                autoComplete="off"
                                onChange={(e) => {
                                    const el = nameInputRef.current;
                                    const start = el ? el.selectionStart : null;
                                    const end = el ? el.selectionEnd : null;
                                    console.debug('✏️ Name change', { value: e.target.value, start, end });
                                    setExperimentName(e.target.value);
                                    // Restore caret after state update on next tick
                                    requestAnimationFrame(() => {
                                        if (nameInputRef.current && start !== null && end !== null) {
                                            try {
                                                nameInputRef.current.setSelectionRange(start, end);
                                                nameInputRef.current.focus();
                                            } catch {}
                                        }
                                    });
                                }}
                                required
                            />
                        </Grid>
                        
                        <Grid item xs={12} md={6}>
                            <FormControl fullWidth required>
                                <InputLabel>Base Model</InputLabel>
                                <Select
                                    value={selectedModelName}
                                    onChange={(e) => setSelectedModelName(e.target.value)}
                                    label="Base Model"
                                >
                                    {availableModels.map((model) => (
                                        <MenuItem key={model.name} value={model.name}>
                                            {model.display_name || model.name}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                        </Grid>
                        
                        <Grid item xs={12}>
                            <TextField
                                fullWidth
                                label="Description"
                                value={experimentDescription}
                                inputRef={descInputRef}
                                autoComplete="off"
                                onChange={(e) => {
                                    const el = descInputRef.current;
                                    const start = el ? el.selectionStart : null;
                                    const end = el ? el.selectionEnd : null;
                                    setExperimentDescription(e.target.value);
                                    requestAnimationFrame(() => {
                                        if (descInputRef.current && start !== null && end !== null) {
                                            try {
                                                descInputRef.current.setSelectionRange(start, end);
                                                descInputRef.current.focus();
                                            } catch {}
                                        }
                                    });
                                }}
                                multiline
                                rows={3}
                            />
                        </Grid>
                        
                        {/* Dataset Upload */}
                        <Grid item xs={12}>
                            <Typography variant="subtitle1" gutterBottom>
                                Dataset
                            </Typography>
                        </Grid>
                        
                        <Grid item xs={12}>
                            <FormControl fullWidth required>
                                <InputLabel>Select Dataset</InputLabel>
                                <Select
                                    value={selectedDatasetId}
                                    onChange={(e) => setSelectedDatasetId(e.target.value)}
                                    label="Select Dataset"
                                >
                                    {datasets.map((dataset) => (
                                        <MenuItem key={dataset.id} value={dataset.id}>
                                            {dataset.name} - {dataset.description || 'No description'}
                                        </MenuItem>
                                    ))}
                                </Select>
                            </FormControl>
                            {datasets.length === 0 && (
                                <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
                                    No finetuning datasets available. Create one from the Datasets tab.
                                </Typography>
                            )}
                        </Grid>
                        
                        {/* Training Parameters */}
                        <Grid item xs={12}>
                            <Divider sx={{ my: 2 }} />
                            <Typography variant="subtitle1" gutterBottom>
                                Training Parameters
                            </Typography>
                        </Grid>
                        
                        <Grid item xs={12} md={4}>
                            <TextField
                                fullWidth
                                label="Learning Rate"
                                type="number"
                                value={learningRate}
                                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                                inputProps={{ step: 0.00001, min: 0.000001, max: 0.1 }}
                            />
                        </Grid>
                        
                        <Grid item xs={12} md={4}>
                            <TextField
                                fullWidth
                                label="Number of Epochs"
                                type="number"
                                value={numEpochs}
                                onChange={(e) => setNumEpochs(parseInt(e.target.value))}
                                inputProps={{ min: 1, max: 100 }}
                            />
                        </Grid>
                        
                        <Grid item xs={12} md={4}>
                            <TextField
                                fullWidth
                                label="Batch Size"
                                type="number"
                                value={batchSize}
                                onChange={(e) => setBatchSize(parseInt(e.target.value))}
                                inputProps={{ min: 1, max: 32 }}
                            />
                        </Grid>
                        
                        {/* LoRA Parameters */}
                        <Grid item xs={12}>
                            <Divider sx={{ my: 2 }} />
                            <Typography variant="subtitle1" gutterBottom>
                                LoRA Parameters
                            </Typography>
                        </Grid>
                        
                        <Grid item xs={12} md={3}>
                            <TextField
                                fullWidth
                                label="LoRA Rank (r)"
                                type="number"
                                value={loraR}
                                onChange={(e) => setLoraR(parseInt(e.target.value))}
                                inputProps={{ min: 1, max: 256 }}
                            />
                        </Grid>
                        
                        <Grid item xs={12} md={3}>
                            <TextField
                                fullWidth
                                label="LoRA Alpha"
                                type="number"
                                value={loraAlpha}
                                onChange={(e) => setLoraAlpha(parseInt(e.target.value))}
                                inputProps={{ min: 1, max: 512 }}
                            />
                        </Grid>
                        
                        <Grid item xs={12} md={3}>
                            <TextField
                                fullWidth
                                label="LoRA Dropout"
                                type="number"
                                value={loraDropout}
                                onChange={(e) => setLoraDropout(parseFloat(e.target.value))}
                                inputProps={{ step: 0.01, min: 0, max: 1 }}
                            />
                        </Grid>
                        
                        <Grid item xs={12} md={3}>
                            <TextField
                                fullWidth
                                label="Target Modules"
                                value={targetModules}
                                onChange={(e) => setTargetModules(e.target.value)}
                                helperText="Comma-separated list (e.g., q_proj,v_proj)"
                            />
                        </Grid>
                        
                        {/* Action Buttons */}
                        <Grid item xs={12}>
                            <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                                <Button
                                    variant="contained"
                                    onClick={handleCreateExperiment}
                                    disabled={loading}
                                    startIcon={loading ? <CircularProgress size={20} /> : <AddIcon />}
                                >
                                    {loading ? 'Creating...' : 'Create Experiment'}
                                </Button>
                                
                                <Button
                                    variant="outlined"
                                    onClick={loadInitialData}
                                    disabled={loading}
                                    startIcon={<RefreshIcon />}
                                >
                                    Refresh
                                </Button>
                            </Box>
                        </Grid>
                    </Grid>
                </CardContent>
            </Card>

            {/* Experiments Table */}
            <Card>
                <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                        <Typography variant="h6">
                            Recent Experiments
                        </Typography>
                        <Button
                            variant="outlined"
                            onClick={loadExperiments}
                            disabled={loading}
                            startIcon={<RefreshIcon />}
                        >
                            Refresh
                        </Button>
                    </Box>
                    
                    {loading ? (
                        <CircularProgress />
                    ) : (
                        <TableContainer component={Paper}>
                            <Table>
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Name</TableCell>
                                        <TableCell>Model</TableCell>
                                        <TableCell>Status</TableCell>
                                        <TableCell>Created</TableCell>
                                        <TableCell>Progress</TableCell>
                                        <TableCell>Actions</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {experiments.map((experiment) => (
                                        <TableRow key={experiment.id}>
                                            <TableCell>
                                                <Typography variant="subtitle2">
                                                    {experiment.name}
                                                </Typography>
                                                <Typography variant="body2" color="textSecondary">
                                                    {experiment.description}
                                                </Typography>
                                            </TableCell>
                                            <TableCell>{experiment.model_name}</TableCell>
                                            <TableCell>{getStatusChip(experiment.status)}</TableCell>
                                            <TableCell>{formatDate(experiment.created_at)}</TableCell>
                                            <TableCell>
                                                {experiment.status === 'running' && (
                                                    <Box sx={{ width: '100%' }}>
                                                        <LinearProgress 
                                                            variant="determinate" 
                                                            value={experiment.progress || 0} 
                                                        />
                                                        <Typography variant="caption">
                                                            {experiment.progress || 0}%
                                                        </Typography>
                                                    </Box>
                                                )}
                                            </TableCell>
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
                                                        <Tooltip title="Stop Training">
                                                            <IconButton
                                                                size="small"
                                                                onClick={() => handleStopTraining(experiment.id)}
                                                                color="warning"
                                                            >
                                                                <StopIcon />
                                                            </IconButton>
                                                        </Tooltip>
                                                    )}
                                                    
                                                    <Tooltip title="View Logs">
                                                        <IconButton
                                                            size="small"
                                                            onClick={() => handleViewLogs(experiment.id)}
                                                        >
                                                            <InfoIcon />
                                                        </IconButton>
                                                    </Tooltip>
                                                    
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
                    )}
                </CardContent>
            </Card>
        </Box>
    );

    const DatasetsTab = () => (
        <Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Typography variant="h5" gutterBottom>
                    Finetuning Datasets
                </Typography>
                <Box sx={{ display: 'flex', gap: 2 }}>
                    <Button
                        variant="outlined"
                        startIcon={<UploadIcon />}
                        onClick={() => setUploadDialogOpen(true)}
                    >
                        Upload JSONL
                    </Button>
                    <Button
                        variant="contained"
                        startIcon={<AddIcon />}
                        onClick={() => {
                            setCreateDatasetDialogOpen(true);
                            loadAvailableDocuments(); // Load documents when dialog opens
                        }}
                    >
                        Create Dataset
                    </Button>
                </Box>
            </Box>
            
            {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                    {error && error.message ? error.message : (error || '')}
                </Alert>
            )}
            
            <Card>
                <CardContent>
                    <Typography variant="h6" gutterBottom>
                        Datasets
                    </Typography>
                    <TableContainer>
                        <Table>
                            <TableHead>
                                <TableRow>
                                    <TableCell>Name</TableCell>
                                    <TableCell>Description</TableCell>
                                    <TableCell>Size</TableCell>
                                    <TableCell>Status</TableCell>
                                    <TableCell>Created</TableCell>
                                    <TableCell>Actions</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {datasets.length === 0 ? (
                                    <TableRow>
                                        <TableCell colSpan={6} align="center" sx={{ py: 4 }}>
                                            <Typography variant="body2" color="text.secondary">
                                                No datasets available. Upload a JSONL file or create a new dataset.
                                            </Typography>
                                        </TableCell>
                                    </TableRow>
                                ) : (
                                    datasets.map((dataset) => (
                                        <TableRow key={dataset.id} hover>
                                            <TableCell>
                                                <Typography variant="subtitle2">
                                                    {dataset.name}
                                                </Typography>
                                            </TableCell>
                                            <TableCell>
                                                <Typography variant="body2" color="textSecondary">
                                                    {dataset.description || 'No description'}
                                                </Typography>
                                            </TableCell>
                                            <TableCell>{dataset.num_samples || 0} samples</TableCell>
                                            <TableCell>
                                                {getStatusChip(dataset.status || 'completed')}
                                            </TableCell>
                                            <TableCell>{formatDate(dataset.created_at)}</TableCell>
                                            <TableCell>
                                                <Box sx={{ display: 'flex', gap: 1 }}>
                                                    <Tooltip title="View Dataset">
                                                        <IconButton
                                                            size="small"
                                                            onClick={() => handleViewDataset(dataset.id)}
                                                        >
                                                            <ViewIcon />
                                                        </IconButton>
                                                    </Tooltip>
                                                    
                                                    <Tooltip title="Download">
                                                        <IconButton
                                                            size="small"
                                                            onClick={() => window.open(`/api/finetuning/datasets/${dataset.id}/download`)}
                                                        >
                                                            <DownloadIcon />
                                                        </IconButton>
                                                    </Tooltip>
                                                    
                                                    <Tooltip title="Delete">
                                                        <IconButton
                                                            size="small"
                                                            onClick={() => handleDeleteDataset(dataset.id)}
                                                            color="error"
                                                        >
                                                            <DeleteIcon />
                                                        </IconButton>
                                                    </Tooltip>
                                                </Box>
                                            </TableCell>
                                        </TableRow>
                                    ))
                                )}
                            </TableBody>
                        </Table>
                    </TableContainer>
                </CardContent>
            </Card>
        </Box>
    );

    const TrainingTab = () => (
        <Box>
            {!showTrainingDashboard ? (
                <>
                    <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
                        Training Monitoring
                    </Typography>
                    
                    <Grid container spacing={3}>
                        <Grid item xs={12} md={6}>
                            <Card>
                                <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                        Active Training Sessions
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                        Monitor ongoing training experiments in real-time
                                    </Typography>
                                    
                                    {runningExperiments.length === 0 ? (
                                        <Alert severity="info">
                                            No active training sessions. Start an experiment to begin monitoring.
                                        </Alert>
                                    ) : (
                                        <TableContainer component={Paper}>
                                            <Table size="small">
                                                <TableHead>
                                                    <TableRow>
                                                        <TableCell>Experiment</TableCell>
                                                        <TableCell>Model</TableCell>
                                                        <TableCell align="center">Progress</TableCell>
                                                        <TableCell align="center">Actions</TableCell>
                                                    </TableRow>
                                                </TableHead>
                                                <TableBody>
                                                    {experiments
                                                        .filter(exp => exp.status === 'running')
                                                        .map((experiment) => (
                                                            <TableRow key={experiment.id}>
                                                                <TableCell>
                                                                    <Typography variant="body2" fontWeight={600}>
                                                                        {experiment.name}
                                                                    </Typography>
                                                                    <Typography variant="caption" color="text.secondary">
                                                                        {experiment.description}
                                                                    </Typography>
                                                                </TableCell>
                                                                <TableCell>{experiment.base_model}</TableCell>
                                                                <TableCell align="center">
                                                                    <Chip
                                                                        label="Training"
                                                                        color="primary"
                                                                        size="small"
                                                                        icon={<CircularProgress size={12} />}
                                                                    />
                                                                </TableCell>
                                                                <TableCell align="center">
                                                                    <Button
                                                                        size="small"
                                                                        variant="outlined"
                                                                        startIcon={<ViewIcon />}
                                                                         onClick={() => handleOpenDashboard(experiment.id)}
                                                                    >
                                                                        Monitor
                                                                    </Button>
                                                                </TableCell>
                                                            </TableRow>
                                                        ))}
                                                </TableBody>
                                            </Table>
                                        </TableContainer>
                                    )}
                                </CardContent>
                            </Card>
                        </Grid>
                        
                        <Grid item xs={12} md={6}>
                            <Card>
                                <CardContent>
                                    <Typography variant="h6" gutterBottom>
                                        Recent Experiments
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                        View training metrics for completed experiments
                                    </Typography>
                                    
                                    {finishedExperiments.length === 0 ? (
                                        <Alert severity="info">
                                            No completed experiments found.
                                        </Alert>
                                    ) : (
                                        <TableContainer component={Paper}>
                                            <Table size="small">
                                                <TableHead>
                                                    <TableRow>
                                                        <TableCell>Experiment</TableCell>
                                                        <TableCell>Status</TableCell>
                                                        <TableCell align="center">Actions</TableCell>
                                                    </TableRow>
                                                </TableHead>
                                                <TableBody>
                                                    {experiments
                                                        .filter(exp => exp.status === 'completed' || exp.status === 'failed')
                                                        .slice(0, 5)
                                                        .map((experiment) => (
                                                            <TableRow key={experiment.id}>
                                                                <TableCell>
                                                                    <Typography variant="body2" fontWeight={600}>
                                                                        {experiment.name}
                                                                    </Typography>
                                                                </TableCell>
                                                                <TableCell>
                                                                    <Chip
                                                                        label={experiment.status}
                                                                        color={experiment.status === 'completed' ? 'success' : 'error'}
                                                                        size="small"
                                                                    />
                                                                </TableCell>
                                                                <TableCell align="center">
                                                                    <Button
                                                                        size="small"
                                                                        variant="outlined"
                                                                        startIcon={<ViewIcon />}
                                                                         onClick={() => handleOpenDashboard(experiment.id)}
                                                                    >
                                                                        View
                                                                    </Button>
                                                                </TableCell>
                                                            </TableRow>
                                                        ))}
                                                </TableBody>
                                            </Table>
                                        </TableContainer>
                                    )}
                                </CardContent>
                            </Card>
                        </Grid>
                    </Grid>
                </>
            ) : (
                <TrainingDashboard
                    experimentId={selectedTrainingExperiment}
                    onClose={() => {
                        setShowTrainingDashboard(false);
                        setSelectedTrainingExperiment(null);
                    }}
                />
            )}
        </Box>
    );

    return (
        <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
            {/* Sidebar */}
            <Drawer
                variant="permanent"
                sx={{
                    width: SIDEBAR_WIDTH,
                    flexShrink: 0,
                    '& .MuiDrawer-paper': {
                        width: SIDEBAR_WIDTH,
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
                            onClick={() => navigate(-1)}
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
                        <ModelTraining sx={{ 
                            mr: 1, 
                            color: '#2563eb',
                            fontSize: '1.5rem',
                        }} />
                        <Typography variant="h6" sx={{ 
                            fontWeight: 700,
                            color: '#0f172a',
                            fontSize: '1.125rem',
                        }}>
                            Finetuning
                        </Typography>
                    </Box>
                    <Divider sx={{ 
                        mb: 3,
                        borderColor: 'rgba(148, 163, 184, 0.2)',
                    }} />
                    
                    {/* Navigation Items */}
                    <List sx={{ p: 0 }}>
                        <ListItemButton
                            selected={activeTab === 0}
                            onClick={() => setActiveTab(0)}
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
                                color: activeTab === 0 ? '#2563eb' : '#64748b',
                                minWidth: '40px',
                            }}>
                                <Assignment />
                            </ListItemIcon>
                            <ListItemText 
                                primary="Experiments" 
                                primaryTypographyProps={{
                                    fontSize: '0.875rem',
                                    fontWeight: activeTab === 0 ? 600 : 500,
                                    color: activeTab === 0 ? '#2563eb' : '#475569',
                                }}
                            />
                        </ListItemButton>
                        
                        <ListItemButton
                            selected={activeTab === 1}
                            onClick={() => setActiveTab(1)}
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
                                color: activeTab === 1 ? '#2563eb' : '#64748b',
                                minWidth: '40px',
                            }}>
                                <DataUsage />
                            </ListItemIcon>
                            <ListItemText 
                                primary="Datasets" 
                                primaryTypographyProps={{
                                    fontSize: '0.875rem',
                                    fontWeight: activeTab === 1 ? 600 : 500,
                                    color: activeTab === 1 ? '#2563eb' : '#475569',
                                }}
                            />
                        </ListItemButton>
                        
                        <ListItemButton
                            selected={activeTab === 2}
                            onClick={() => setActiveTab(2)}
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
                                color: activeTab === 2 ? '#2563eb' : '#64748b',
                                minWidth: '40px',
                            }}>
                                <Timeline />
                            </ListItemIcon>
                            <ListItemText 
                                primary="Training" 
                                primaryTypographyProps={{
                                    fontSize: '0.875rem',
                                    fontWeight: activeTab === 2 ? 600 : 500,
                                    color: activeTab === 2 ? '#2563eb' : '#475569',
                                }}
                            />
                        </ListItemButton>
                        
                        <ListItemButton
                            selected={activeTab === 3}
                            onClick={() => setActiveTab(3)}
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
                                color: activeTab === 3 ? '#2563eb' : '#64748b',
                                minWidth: '40px',
                            }}>
                                <Dashboard />
                            </ListItemIcon>
                            <ListItemText 
                                primary="Compare" 
                                primaryTypographyProps={{
                                    fontSize: '0.875rem',
                                    fontWeight: activeTab === 3 ? 600 : 500,
                                    color: activeTab === 3 ? '#2563eb' : '#475569',
                                }}
                            />
                        </ListItemButton>
                    </List>
                </Box>
            </Drawer>

            {/* Main Content */}
            <Box
                component="main"
                sx={{
                    flexGrow: 1,
                    p: 3,
                    bgcolor: 'background.default',
                    minHeight: '100vh',
                }}
            >
                {/* Tab Content */}
                {activeTab === 0 && <ExperimentsTab />}
                {activeTab === 1 && <DatasetsTab />}
                {activeTab === 2 && <TrainingTab />}
                {activeTab === 3 && <ExperimentComparison availableExperiments={experiments} />}
            </Box>

            {/* Upload Dataset Dialog */}
            <Dialog open={uploadDialogOpen} onClose={() => setUploadDialogOpen(false)} maxWidth="sm" fullWidth>
                <DialogTitle>Upload Dataset</DialogTitle>
                <DialogContent>
                    <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                        Upload a JSONL file with training data. Each line should contain a JSON object with "instruction" and "response" fields.
                    </Typography>
                    <Button
                        variant="outlined"
                        component="label"
                        startIcon={<UploadIcon />}
                        fullWidth
                        sx={{ mt: 2 }}
                    >
                        Choose JSONL File
                        <input
                            type="file"
                            hidden
                            accept=".jsonl,.json"
                            onChange={(e) => {
                                const file = e.target.files[0];
                                if (file) {
                                    handleUploadDataset(file);
                                }
                            }}
                        />
                    </Button>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setUploadDialogOpen(false)}>Cancel</Button>
                </DialogActions>
            </Dialog>

            {/* Create Dataset Dialog */}
            <Dialog open={createDatasetDialogOpen} onClose={() => setCreateDatasetDialogOpen(false)} maxWidth="md" fullWidth>
                <DialogTitle>Create Finetuning Dataset</DialogTitle>
                <DialogContent>
                    <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                        Generate a new finetuning dataset from your documents using Q-C-A format.
                    </Typography>
                    <TextField
                        fullWidth
                        label="Dataset Name"
                        value={newDatasetName || ''}
                        onChange={(e) => setNewDatasetName(e.target.value)}
                        sx={{ mt: 2, mb: 2 }}
                    />
                    <TextField
                        fullWidth
                        label="Description"
                        value={newDatasetDescription || ''}
                        onChange={(e) => setNewDatasetDescription(e.target.value)}
                        multiline
                        rows={3}
                        sx={{ mb: 2 }}
                    />
                    
                    {/* Document Selection */}
                    <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>
                        Select Documents
                    </Typography>
                    
                    {/* Select All Documents Option */}
                    {availableDocuments.length > 0 && (
                        <Box sx={{ mb: 2 }}>
                            <FormControlLabel
                                control={
                                    <Checkbox
                                        checked={selectedDocuments.length === availableDocuments.length}
                                        indeterminate={selectedDocuments.length > 0 && selectedDocuments.length < availableDocuments.length}
                                        onChange={(e) => {
                                            if (e.target.checked) {
                                                setSelectedDocuments(availableDocuments.map(doc => doc?.id).filter(id => id));
                                            } else {
                                                setSelectedDocuments([]);
                                            }
                                        }}
                                    />
                                }
                                label="Select All Documents"
                            />
                        </Box>
                    )}
                    
                    <List sx={{ maxHeight: 300, overflow: 'auto', border: '1px solid #ddd', borderRadius: 1, mb: 2 }}>
                        {availableDocuments.length === 0 ? (
                            <ListItem>
                                <ListItemText 
                                    primary="No documents available"
                                    secondary="Please upload some documents first"
                                />
                            </ListItem>
                        ) : (
                            availableDocuments.map((doc) => (
                                <ListItem key={doc?.id || 'unknown'}>
                                    <ListItemText 
                                        primary={doc?.filename || 'Unknown Document'}
                                        secondary={`Status: ${doc?.status || 'Unknown'} | Size: ${doc?.size_mb ? `${doc.size_mb.toFixed(1)} MB` : 'Unknown'}`}
                                    />
                                    <ListItemSecondaryAction>
                                        <Checkbox
                                            checked={selectedDocuments.includes(doc?.id)}
                                            onChange={(e) => {
                                                if (e.target.checked) {
                                                    setSelectedDocuments([...selectedDocuments, doc.id]);
                                                } else {
                                                    setSelectedDocuments(selectedDocuments.filter(id => id !== doc.id));
                                                }
                                            }}
                                        />
                                    </ListItemSecondaryAction>
                                </ListItem>
                            ))
                        )}
                    </List>
                    
                    <TextField
                        fullWidth
                        label="Questions per Document"
                        type="number"
                        value={questionsPerDoc || 5}
                        onChange={(e) => setQuestionsPerDoc(parseInt(e.target.value))}
                        inputProps={{ min: 1, max: 20 }}
                        sx={{ mb: 2 }}
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setCreateDatasetDialogOpen(false)}>Cancel</Button>
                    <Button 
                        variant="contained"
                        onClick={handleCreateDataset}
                        disabled={!newDatasetName || !selectedDocuments?.length || loading}
                    >
                        Create Dataset
                    </Button>
                </DialogActions>
            </Dialog>

            {/* View Dataset Dialog */}
            <Dialog open={viewDatasetDialogOpen} onClose={() => setViewDatasetDialogOpen(false)} maxWidth="lg" fullWidth>
                <DialogTitle>Dataset Details</DialogTitle>
                <DialogContent>
                    {datasetDetails ? (
                        <Box>
                            <Typography variant="h6" gutterBottom>
                                {datasetDetails.name}
                            </Typography>
                            <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                                {datasetDetails.description}
                            </Typography>
                            
                            <Typography variant="subtitle1" gutterBottom>
                                Sample Data:
                            </Typography>
                            <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
                                <Table size="small">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Instruction</TableCell>
                                            <TableCell>Input</TableCell>
                                            <TableCell>Output</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {(datasetDetails.samples || []).slice(0, 10).map((sample, index) => (
                                            <TableRow key={index}>
                                                <TableCell sx={{ verticalAlign: 'top', minWidth: 200, maxWidth: 400 }}>
                                                    <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                                                        {sample.instruction}
                                                    </Typography>
                                                </TableCell>
                                                <TableCell sx={{ verticalAlign: 'top', minWidth: 300, maxWidth: 500 }}>
                                                    <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                                                        {sample.input}
                                                    </Typography>
                                                </TableCell>
                                                <TableCell sx={{ verticalAlign: 'top', minWidth: 300, maxWidth: 500 }}>
                                                    <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                                                        {sample.output}
                                                    </Typography>
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </Box>
                    ) : (
                        <CircularProgress />
                    )}
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setViewDatasetDialogOpen(false)}>Close</Button>
                </DialogActions>
            </Dialog>

            {/* Training Logs Dialog */}
            <Dialog
                open={openDialog}
                onClose={() => setOpenDialog(false)}
                maxWidth="md"
                fullWidth
            >
                <DialogTitle>Training Logs</DialogTitle>
                <DialogContent>
                    <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                        {trainingLogs.length > 0 ? (
                            trainingLogs.map((log, index) => (
                                <Box key={index} sx={{ mb: 1, p: 1, bgcolor: 'grey.100', borderRadius: 1 }}>
                                    <Typography variant="caption" color="textSecondary">
                                        {formatDate(log.timestamp)}
                                    </Typography>
                                    <Typography variant="body2">
                                        {log.message}
                                    </Typography>
                                </Box>
                            ))
                        ) : (
                            <Typography color="textSecondary">
                                No logs available yet.
                            </Typography>
                        )}
                    </Box>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setOpenDialog(false)}>
                        Close
                    </Button>
                </DialogActions>
            </Dialog>

            {/* Notifications */}
            <Snackbar
                open={!!error}
                autoHideDuration={6000}
                onClose={() => setError(null)}
            >
                <Alert severity="error" onClose={() => setError(null)}>
                    {error && error.message ? error.message : (error || '')}
                </Alert>
            </Snackbar>

            <Snackbar
                open={!!success}
                autoHideDuration={4000}
                onClose={() => setSuccess(null)}
            >
                <Alert severity="success" onClose={() => setSuccess(null)}>
                    {success && success.message ? success.message : (success || '')}
                </Alert>
            </Snackbar>
        </Box>
    );
};

export default FinetuningPage;
