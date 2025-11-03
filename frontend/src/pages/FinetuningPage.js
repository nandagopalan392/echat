import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '@mui/material/styles';
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
    Add,
    Settings as SettingsIcon,
    Timeline as TimelineIcon,
    Assessment as AssessmentIcon,
    DataObject as DataIcon,
    Memory as MemoryIcon,
    Speed as SpeedIcon,
    Timer as TimerIcon,
    TrendingUp as TrendingUpIcon,
    TrendingUp,
    Close as CloseIcon,
    Refresh as RefreshIcon,
    Storage,
    Compare
} from '@mui/icons-material';

import api from '../services/api';
import { evaluationApi } from '../services/api';
import webSocketService from '../services/websocketService';
import TrainingDashboard from '../components/TrainingDashboard';
import ExperimentComparison from '../components/ExperimentComparison';
import LossChart from '../components/charts/LossChart';
import LearningRateChart from '../components/charts/LearningRateChart';
import ReactECharts from 'echarts-for-react';

const SIDEBAR_WIDTH = 280;

// Multi-experiment chart components
const MultiExperimentLossChart = ({ experimentsData, experiments }) => {
    const theme = useTheme();
    
    console.log('LossChart - experimentsData:', experimentsData);
    console.log('LossChart - experiments:', experiments);
    console.log('LossChart - Object.keys(experimentsData):', Object.keys(experimentsData));
    
    const chartOptions = {
        animation: false,
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'cross',
                label: {
                    backgroundColor: '#6a7985'
                }
            },
            formatter: function (params) {
                let result = `Step: ${params[0].axisValue}<br/>`;
                params.forEach(param => {
                    const color = param.color;
                    const value = param.value && param.value[1] !== undefined ? param.value[1].toFixed(4) : 'N/A';
                    result += `<span style="color: ${color};">●</span> ${param.seriesName}: ${value}<br/>`;
                });
                return result;
            }
        },
        legend: {
            data: [],
            right: 10,
            textStyle: { color: theme.palette.text.primary }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'value',
            name: 'Training Step',
            nameLocation: 'middle',
            nameGap: 25,
            axisLabel: { color: theme.palette.text.secondary },
            axisLine: { lineStyle: { color: theme.palette.divider } }
        },
        yAxis: {
            type: 'value',
            name: 'Loss',
            nameLocation: 'middle',
            nameGap: 40,
            axisLabel: { color: theme.palette.text.secondary },
            axisLine: { lineStyle: { color: theme.palette.divider } }
        },
        series: []
    };

    const colors = ['#1976d2', '#d32f2f', '#388e3c', '#f57c00', '#7b1fa2', '#0288d1'];
    
    experiments.forEach((exp, index) => {
        const data = experimentsData[exp.id];
        console.log(`LossChart - Processing experiment ${exp.name} (${exp.id}):`, data);
        
        if (data && data.metrics) {
            const color = colors[index % colors.length];
            
            console.log(`Experiment ${exp.name} metrics structure:`, data.metrics);
            
            // Training loss series
            if (data.metrics.train_losses && data.metrics.train_losses.length > 0) {
                console.log(`Train losses sample:`, data.metrics.train_losses.slice(0, 3));
                console.log(`First train loss data point:`, data.metrics.train_losses[0]);
                console.log(`Train losses data type:`, typeof data.metrics.train_losses[0]);
                
                chartOptions.series.push({
                    name: `${exp.name} (Train)`,
                    type: 'line',
                    data: data.metrics.train_losses,
                    lineStyle: { color, width: 2 },
                    showSymbol: false,
                    smooth: true
                });
                chartOptions.legend.data.push(`${exp.name} (Train)`);
            }
            
            // Validation loss series
            if (data.metrics.eval_losses && data.metrics.eval_losses.length > 0) {
                console.log(`Eval losses sample:`, data.metrics.eval_losses.slice(0, 3));
                chartOptions.series.push({
                    name: `${exp.name} (Val)`,
                    type: 'line',
                    data: data.metrics.eval_losses,
                    lineStyle: { color, width: 2, type: 'dashed' },
                    showSymbol: false,
                    smooth: true
                });
                chartOptions.legend.data.push(`${exp.name} (Val)`);
            }
        }
    });

    console.log(`Final chart series count: ${chartOptions.series.length}`);
    console.log(`Chart legend data:`, chartOptions.legend.data);
    console.log(`Chart series:`, chartOptions.series);

    return (
        <Box sx={{ height: '100%', width: '100%' }}>
            <ReactECharts 
                option={chartOptions} 
                style={{ height: '320px', width: '100%' }}
                theme="light"
            />
        </Box>
    );
};

const MultiExperimentLearningRateChart = ({ experimentsData, experiments }) => {
    const theme = useTheme();
    
    const chartOptions = {
        animation: false,
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'cross',
                label: {
                    backgroundColor: '#6a7985'
                }
            },
            formatter: function (params) {
                let result = `Step: ${params[0].axisValue}<br/>`;
                params.forEach(param => {
                    const color = param.color;
                    const value = param.value && param.value[1] !== undefined ? param.value[1].toExponential(2) : 'N/A';
                    result += `<span style="color: ${color};">●</span> ${param.seriesName}: ${value}<br/>`;
                });
                return result;
            }
        },
        legend: {
            data: [],
            right: 10,
            textStyle: { color: theme.palette.text.primary }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'value',
            name: 'Training Step',
            nameLocation: 'middle',
            nameGap: 25,
            axisLabel: { color: theme.palette.text.secondary },
            axisLine: { lineStyle: { color: theme.palette.divider } }
        },
        yAxis: {
            type: 'log',
            name: 'Learning Rate',
            nameLocation: 'middle',
            nameGap: 50,
            axisLabel: { 
                color: theme.palette.text.secondary,
                formatter: function (value) {
                    return value.toExponential(0);
                }
            },
            axisLine: { lineStyle: { color: theme.palette.divider } }
        },
        series: []
    };

    const colors = ['#1976d2', '#d32f2f', '#388e3c', '#f57c00', '#7b1fa2', '#0288d1'];
    
    experiments.forEach((exp, index) => {
        const data = experimentsData[exp.id];
        if (data && data.metrics && data.metrics.learning_rates && data.metrics.learning_rates.length > 0) {
            const color = colors[index % colors.length];
            
            console.log(`Learning rates sample for ${exp.name}:`, data.metrics.learning_rates.slice(0, 3));
            
            chartOptions.series.push({
                name: exp.name,
                type: 'line',
                data: data.metrics.learning_rates,
                lineStyle: { color, width: 2 },
                showSymbol: false,
                smooth: true
            });
            chartOptions.legend.data.push(exp.name);
        }
    });

    return (
        <Box sx={{ height: '100%', width: '100%' }}>
            <ReactECharts 
                option={chartOptions} 
                style={{ height: '320px', width: '100%' }}
                theme="light"
            />
        </Box>
    );
};

const ResourceUsageBarChart = ({ experimentsData, experiments }) => {
    const theme = useTheme();
    
    const experimentNames = experiments.map(exp => exp.name);
    const gpuData = experiments.map(exp => {
        const data = experimentsData[exp.id];
        return data && data.system ? data.system.peak_gpu_memory : 0;
    });
    const memoryData = experiments.map(exp => {
        const data = experimentsData[exp.id];
        return data && data.system ? data.system.peak_cpu_memory : 0;
    });

    const chartOptions = {
        animation: false,
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            },
            formatter: function (params) {
                let result = `${params[0].axisValue}<br/>`;
                params.forEach(param => {
                    const unit = param.seriesName === 'Peak GPU Memory' ? 'GB' : 'GB';
                    result += `<span style="color: ${param.color};">●</span> ${param.seriesName}: ${param.value} ${unit}<br/>`;
                });
                return result;
            }
        },
        legend: {
            data: ['Peak GPU Memory', 'Peak CPU Memory'],
            right: 10,
            textStyle: { color: theme.palette.text.primary }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: experimentNames,
            axisLabel: { color: theme.palette.text.secondary },
            axisLine: { lineStyle: { color: theme.palette.divider } }
        },
        yAxis: {
            type: 'value',
            name: 'Memory Usage (GB)',
            nameLocation: 'middle',
            nameGap: 40,
            axisLabel: { color: theme.palette.text.secondary },
            axisLine: { lineStyle: { color: theme.palette.divider } }
        },
        series: [
            {
                name: 'Peak GPU Memory',
                type: 'bar',
                data: gpuData,
                itemStyle: { color: '#1976d2' }
            },
            {
                name: 'Peak CPU Memory',
                type: 'bar',
                data: memoryData,
                itemStyle: { color: '#388e3c' }
            }
        ]
    };

    return (
        <Box sx={{ height: '300px', width: '100%' }}>
            <ReactECharts 
                option={chartOptions} 
                style={{ height: '100%', width: '100%' }}
                theme="light"
            />
        </Box>
    );
};

const FinetuningPage = () => {
    const navigate = useNavigate();
    
    // Sidebar and experiment selection state
    const [sidebarSelection, setSidebarSelection] = useState('training');
    const [selectedExperimentId, setSelectedExperimentId] = useState(null);
    
    // Comparison state - for selecting multiple experiments
    const [selectedExperimentsForComparison, setSelectedExperimentsForComparison] = useState([]);
    
    // Comparison charts state
    const [comparisonMetrics, setComparisonMetrics] = useState({});
    const [loadingMetrics, setLoadingMetrics] = useState(false);
    
    // Dashboard state for inline viewing in comparison
    const [comparisonDashboardExpId, setComparisonDashboardExpId] = useState(null);
    
    // Legacy state - keeping for compatibility
    const [activeTab, setActiveTab] = useState(0);
    const [selectedTrainingExperiment, setSelectedTrainingExperiment] = useState(null);
    const [showTrainingDashboard, setShowTrainingDashboard] = useState(false);
    
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

    // Helper function to get the active experiment (latest created if multiple running)
    const getActiveExperiment = () => {
        if (selectedExperimentId) {
            return experiments.find(exp => exp.id === selectedExperimentId);
        }
        
        // Find running experiments, sorted by creation date (latest first)
        const runningExperiments = experiments
            .filter(exp => exp.status === 'running')
            .sort((a, b) => new Date(b.created_at || b.timestamp) - new Date(a.created_at || a.timestamp));
        
        return runningExperiments.length > 0 ? runningExperiments[0] : null;
    };

    // Get current active experiment
    const activeExperiment = getActiveExperiment();

    // Load initial data
    useEffect(() => {
        loadInitialData();
    }, []);

    // Clear comparison selection when switching away from comparison tab
    useEffect(() => {
        if (sidebarSelection !== 'comparison') {
            setSelectedExperimentsForComparison([]);
        }
    }, [sidebarSelection]);

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

            const result = await api.createFineTuningExperiment(formData);
            setSuccess(`Experiment "${result?.experiment?.name || 'New experiment'}" created successfully!`);
            
            // Reset form
            setExperimentName('');
            setExperimentDescription('');
            setSelectedModelName('');
            setSelectedDatasetId('');
            
            await loadExperiments();
        } catch (err) {
            setError('Failed to create experiment: ' + (err.message || 'Unknown error'));
        } finally {
            setLoading(false);
        }
    };

    const handleStartTraining = async (experimentId) => {
        try {
            // Set up WebSocket for training updates
            const ws = webSocketService.connect(experimentId, {
                onMessage: (data) => {
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
                    setError('WebSocket connection error.');
                },
                onClose: () => {
                    loadExperiments(); // Refresh experiments when WebSocket closes
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
                document_ids: selectedDocuments,
                questions_per_document: questionsPerDoc
            };

            await api.createQCADataset(payload);
            setSuccess('Dataset creation started successfully!');
            setCreateDatasetDialogOpen(false);
            
            // Reset form
            setNewDatasetName('');
            setNewDatasetDescription('');
            setSelectedDocuments([]);
            setQuestionsPerDoc(5);
            
            await loadDatasets();
        } catch (err) {
            setError('Failed to create dataset: ' + (err.message || 'Unknown error'));
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

    const handleDeleteDataset = async (datasetId) => {
        try {
            await api.deleteDataset(datasetId);
            setSuccess('Dataset deleted successfully!');
            await loadDatasets();
        } catch (err) {
            setError('Failed to delete dataset: ' + (err.message || 'Unknown error'));
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

    // Load comparison metrics for selected experiments
    const loadComparisonMetrics = async () => {
        console.log('loadComparisonMetrics called with:', selectedExperimentsForComparison);
        
        if (selectedExperimentsForComparison.length === 0) {
            setComparisonMetrics({});
            return;
        }

        setLoadingMetrics(true);
        try {
            const metricsData = {};
            let hasRealMetrics = false;
            
            for (let i = 0; i < selectedExperimentsForComparison.length; i++) {
                const expId = selectedExperimentsForComparison[i];
                console.log(`Fetching metrics for experiment: ${expId}`);
                try {
                    // Try to fetch real metrics only
                    const response = await api.getFineTuningMetrics(expId);
                    console.log(`Response for ${expId}:`, response);
                    
                    // Only include if it has actual training data
                    if (response && response.metrics && 
                        (response.metrics.train_losses?.length > 0 || 
                         response.metrics.eval_losses?.length > 0 ||
                         response.metrics.learning_rates?.length > 0)) {
                        metricsData[expId] = response;
                        hasRealMetrics = true;
                        console.log(`Added real metrics for ${expId}`);
                    } else {
                        console.log(`No valid training data for ${expId}:`, response);
                    }
                } catch (error) {
                    console.log(`Error fetching metrics for experiment ${expId}:`, error);
                    // Don't add mock data - just skip this experiment
                }
            }
            
            console.log('Final metricsData:', metricsData);
            console.log('hasRealMetrics:', hasRealMetrics);
            setComparisonMetrics(hasRealMetrics ? metricsData : {});
        } catch (err) {
            console.error('Failed to load comparison metrics:', err);
            setError('Failed to load metrics for comparison');
        } finally {
            setLoadingMetrics(false);
        }
    };

    // Load metrics when selected experiments change
    useEffect(() => {
        if (selectedExperimentsForComparison.length > 0) {
            loadComparisonMetrics();
        } else {
            setComparisonMetrics({});
        }
    }, [selectedExperimentsForComparison]);

    // Calculate metrics
    const metrics = {
        total_experiments: experiments.length,
        running_experiments: experiments.filter(exp => exp.status === 'running').length,
        completed_experiments: experiments.filter(exp => exp.status === 'completed').length,
        failed_experiments: experiments.filter(exp => exp.status === 'failed').length
    };

    // Instead of completely replacing the page, we'll show dashboard content within comparison
    // No need for the full page replacement check here

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
                        <TrendingUp sx={{ 
                            mr: 1, 
                            color: '#2563eb',
                            fontSize: '1.5rem',
                        }} />
                        <Typography variant="h6" sx={{ 
                            fontWeight: 700,
                            color: '#0f172a',
                            fontSize: '1.125rem',
                        }}>
                            Fine-tuning
                        </Typography>
                    </Box>
                    <Divider sx={{ 
                        mb: 3,
                        borderColor: 'rgba(148, 163, 184, 0.2)',
                    }} />
                    
                    {/* Navigation Items */}
                    <List sx={{ p: 0 }}>
                        <ListItemButton
                            selected={sidebarSelection === 'training'}
                            onClick={() => setSidebarSelection('training')}
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
                                color: sidebarSelection === 'training' ? '#2563eb' : '#64748b',
                                minWidth: '40px',
                            }}>
                                <TrendingUp />
                            </ListItemIcon>
                            <ListItemText 
                                primary="Training" 
                                primaryTypographyProps={{
                                    fontSize: '0.875rem',
                                    fontWeight: sidebarSelection === 'training' ? 600 : 500,
                                    color: sidebarSelection === 'training' ? '#2563eb' : '#475569',
                                }}
                            />
                        </ListItemButton>
                        
                        <ListItemButton
                            selected={sidebarSelection === 'create'}
                            onClick={() => setSidebarSelection('create')}
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
                                color: sidebarSelection === 'create' ? '#2563eb' : '#64748b',
                                minWidth: '40px',
                            }}>
                                <Add />
                            </ListItemIcon>
                            <ListItemText 
                                primary="Create Experiment" 
                                primaryTypographyProps={{
                                    fontSize: '0.875rem',
                                    fontWeight: sidebarSelection === 'create' ? 600 : 500,
                                    color: sidebarSelection === 'create' ? '#2563eb' : '#475569',
                                }}
                            />
                        </ListItemButton>
                        
                        <ListItemButton
                            selected={sidebarSelection === 'datasets'}
                            onClick={() => setSidebarSelection('datasets')}
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
                                color: sidebarSelection === 'datasets' ? '#2563eb' : '#64748b',
                                minWidth: '40px',
                            }}>
                                <Storage />
                            </ListItemIcon>
                            <ListItemText 
                                primary="Datasets" 
                                primaryTypographyProps={{
                                    fontSize: '0.875rem',
                                    fontWeight: sidebarSelection === 'datasets' ? 600 : 500,
                                    color: sidebarSelection === 'datasets' ? '#2563eb' : '#475569',
                                }}
                            />
                        </ListItemButton>
                        
                        <ListItemButton
                            selected={sidebarSelection === 'comparison'}
                            onClick={() => setSidebarSelection('comparison')}
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
                                color: sidebarSelection === 'comparison' ? '#2563eb' : '#64748b',
                                minWidth: '40px',
                            }}>
                                <Compare />
                            </ListItemIcon>
                            <ListItemText 
                                primary="Comparison" 
                                primaryTypographyProps={{
                                    fontSize: '0.875rem',
                                    fontWeight: sidebarSelection === 'comparison' ? 600 : 500,
                                    color: sidebarSelection === 'comparison' ? '#2563eb' : '#475569',
                                }}
                            />
                        </ListItemButton>
                    </List>
                </Box>
            </Drawer>

            {/* Main Content */}
            <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                {/* Error/Success Messages */}
                {error && (
                    <Alert severity="error" sx={{ m: 2 }} onClose={() => setError(null)}>
                        {String(error)}
                    </Alert>
                )}
                
                {success && (
                    <Alert severity="success" sx={{ m: 2 }} onClose={() => setSuccess(null)}>
                        {String(success)}
                    </Alert>
                )}

                {/* Content based on sidebar selection */}
                {sidebarSelection === 'training' && (
                    <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                        {/* Active Experiment Charts */}
                        {activeExperiment ? (
                            <Box sx={{ flex: 1, p: 2 }}>
                                <Typography variant="h5" sx={{ mb: 2 }}>
                                    Training Dashboard - {activeExperiment.name}
                                </Typography>
                                <TrainingDashboard
                                    experimentId={activeExperiment.id}
                                    onClose={() => {}} // No close button in embedded mode
                                />
                            </Box>
                        ) : (
                            <Box sx={{ p: 4, textAlign: 'center' }}>
                                <Typography variant="h6" color="text.secondary">
                                    No active experiments
                                </Typography>
                                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                                    Start a new experiment to see training metrics here
                                </Typography>
                                <Button 
                                    variant="contained" 
                                    sx={{ mt: 2 }}
                                    onClick={() => setSidebarSelection('create')}
                                >
                                    Create Experiment
                                </Button>
                            </Box>
                        )}

                        {/* Experiments List */}
                        <Box sx={{ 
                            borderTop: 1, 
                            borderColor: 'divider', 
                            p: 2, 
                            maxHeight: '40vh', 
                            overflow: 'auto' 
                        }}>
                            <Typography variant="h6" sx={{ mb: 2 }}>
                                All Experiments
                            </Typography>
                            {experiments.length === 0 ? (
                                <Typography variant="body2" color="text.secondary">
                                    No experiments found
                                </Typography>
                            ) : (
                                <Grid container spacing={2}>
                                    {experiments.map((exp) => (
                                        <Grid item xs={12} sm={6} md={4} lg={3} key={exp.id}>
                                            <Card 
                                                sx={{ 
                                                    cursor: 'pointer',
                                                    border: selectedExperimentId === exp.id ? 2 : 1,
                                                    borderColor: selectedExperimentId === exp.id ? 'primary.main' : 'divider',
                                                    '&:hover': { 
                                                        boxShadow: 2,
                                                        borderColor: 'primary.main'
                                                    }
                                                }}
                                                onClick={() => setSelectedExperimentId(
                                                    selectedExperimentId === exp.id ? null : exp.id
                                                )}
                                            >
                                                <CardContent sx={{ p: 2 }}>
                                                    <Typography variant="subtitle2" noWrap>
                                                        {exp.name}
                                                    </Typography>
                                                    <Typography variant="body2" color="text.secondary" noWrap>
                                                        {exp.model_name}
                                                    </Typography>
                                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
                                                        <Chip 
                                                            label={exp.status} 
                                                            size="small"
                                                            color={
                                                                exp.status === 'running' ? 'primary' :
                                                                exp.status === 'completed' ? 'success' :
                                                                exp.status === 'failed' ? 'error' : 'default'
                                                            }
                                                        />
                                                        <Typography variant="caption" color="text.secondary">
                                                            {formatDate(exp.created_at || exp.timestamp)}
                                                        </Typography>
                                                    </Box>
                                                </CardContent>
                                            </Card>
                                        </Grid>
                                    ))}
                                </Grid>
                            )}
                        </Box>
                    </Box>
                )}

                {/* Other sidebar content will be added here */}
                {sidebarSelection === 'create' && (
                    <Box sx={{ flex: 1, overflow: 'auto', p: 3 }}>
                        <Typography variant="h5" sx={{ mb: 3 }}>
                            Create New Experiment
                        </Typography>
                        
                        <Box component="form" sx={{ maxWidth: 600 }}>
                            <Grid container spacing={3}>
                                {/* Basic Information */}
                                <Grid item xs={12}>
                                    <Typography variant="h6" sx={{ mb: 2 }}>
                                        Basic Information
                                    </Typography>
                                </Grid>
                                
                                <Grid item xs={12}>
                                    <TextField
                                        fullWidth
                                        label="Experiment Name"
                                        value={experimentName}
                                        onChange={(e) => setExperimentName(e.target.value)}
                                        required
                                        variant="outlined"
                                    />
                                </Grid>
                                
                                <Grid item xs={12}>
                                    <TextField
                                        fullWidth
                                        label="Description"
                                        value={experimentDescription}
                                        onChange={(e) => setExperimentDescription(e.target.value)}
                                        multiline
                                        rows={3}
                                        variant="outlined"
                                    />
                                </Grid>
                                
                                <Grid item xs={12} sm={6}>
                                    <FormControl fullWidth required>
                                        <InputLabel>Base Model</InputLabel>
                                        <Select
                                            value={selectedModelName}
                                            onChange={(e) => setSelectedModelName(e.target.value)}
                                            label="Base Model"
                                        >
                                            {availableModels.map((model) => (
                                                <MenuItem key={model.name} value={model.name}>
                                                    {model.name} ({model.size})
                                                </MenuItem>
                                            ))}
                                        </Select>
                                    </FormControl>
                                </Grid>
                                
                                <Grid item xs={12} sm={6}>
                                    <FormControl fullWidth required>
                                        <InputLabel>Dataset</InputLabel>
                                        <Select
                                            value={selectedDatasetId}
                                            onChange={(e) => setSelectedDatasetId(e.target.value)}
                                            label="Dataset"
                                        >
                                            {datasets.map((dataset) => (
                                                <MenuItem key={dataset.id} value={dataset.id}>
                                                    {dataset.name} ({dataset.size} examples)
                                                </MenuItem>
                                            ))}
                                        </Select>
                                    </FormControl>
                                </Grid>
                                
                                {/* Training Parameters */}
                                <Grid item xs={12}>
                                    <Typography variant="h6" sx={{ mb: 2, mt: 2 }}>
                                        Training Parameters
                                    </Typography>
                                </Grid>
                                
                                <Grid item xs={12} sm={4}>
                                    <TextField
                                        fullWidth
                                        label="Learning Rate"
                                        type="number"
                                        value={learningRate}
                                        onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                                        inputProps={{ step: 0.00001, min: 0.00001, max: 0.01 }}
                                        variant="outlined"
                                    />
                                </Grid>
                                
                                <Grid item xs={12} sm={4}>
                                    <TextField
                                        fullWidth
                                        label="Number of Epochs"
                                        type="number"
                                        value={numEpochs}
                                        onChange={(e) => setNumEpochs(parseInt(e.target.value))}
                                        inputProps={{ step: 1, min: 1, max: 20 }}
                                        variant="outlined"
                                    />
                                </Grid>
                                
                                <Grid item xs={12} sm={4}>
                                    <TextField
                                        fullWidth
                                        label="Batch Size"
                                        type="number"
                                        value={batchSize}
                                        onChange={(e) => setBatchSize(parseInt(e.target.value))}
                                        inputProps={{ step: 1, min: 1, max: 32 }}
                                        variant="outlined"
                                    />
                                </Grid>
                                
                                {/* LoRA Parameters */}
                                <Grid item xs={12}>
                                    <Typography variant="h6" sx={{ mb: 2, mt: 2 }}>
                                        LoRA Parameters
                                    </Typography>
                                </Grid>
                                
                                <Grid item xs={12} sm={4}>
                                    <TextField
                                        fullWidth
                                        label="LoRA R"
                                        type="number"
                                        value={loraR}
                                        onChange={(e) => setLoraR(parseInt(e.target.value))}
                                        inputProps={{ step: 1, min: 1, max: 64 }}
                                        variant="outlined"
                                    />
                                </Grid>
                                
                                <Grid item xs={12} sm={4}>
                                    <TextField
                                        fullWidth
                                        label="LoRA Alpha"
                                        type="number"
                                        value={loraAlpha}
                                        onChange={(e) => setLoraAlpha(parseInt(e.target.value))}
                                        inputProps={{ step: 1, min: 1, max: 128 }}
                                        variant="outlined"
                                    />
                              
                                </Grid>
                                
                                <Grid item xs={12} sm={4}>
                                    <TextField
                                        fullWidth
                                        label="LoRA Dropout"
                                        type="number"
                                        value={loraDropout}
                                        onChange={(e) => setLoraDropout(parseFloat(e.target.value))}
                                        inputProps={{ step: 0.01, min: 0, max: 1 }}
                                        variant="outlined"
                                    />
                                </Grid>
                                
                                <Grid item xs={12}>
                                    <TextField
                                        fullWidth
                                        label="Target Modules"
                                        value={targetModules}
                                        onChange={(e) => setTargetModules(e.target.value)}
                                        placeholder="q_proj,v_proj"
                                        variant="outlined"
                                        helperText="Comma-separated list of target modules for LoRA"
                                    />
                                </Grid>
                                
                                {/* Submit Button */}
                                <Grid item xs={12}>
                                    <Box sx={{ display: 'flex', gap: 2, mt: 3 }}>
                                        <Button
                                            variant="contained"
                                            onClick={handleCreateExperiment}
                                            disabled={loading || !experimentName || !selectedModelName || !selectedDatasetId}
                                            sx={{ minWidth: 120 }}
                                        >
                                            {loading ? <CircularProgress size={20} /> : 'Create Experiment'}
                                        </Button>
                                        
                                        <Button
                                            variant="outlined"
                                            onClick={() => {
                                                // Reset form
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
                                            }}
                                        >
                                            Reset Form
                                        </Button>
                                    </Box>
                                </Grid>
                            </Grid>
                        </Box>
                    </Box>
                )}

                {sidebarSelection === 'datasets' && (
                    <Box sx={{ flex: 1, overflow: 'auto', p: 3 }}>
                        <Typography variant="h5" sx={{ mb: 3 }}>
                            Dataset Management
                        </Typography>
                        
                        {/* Dataset Actions */}
                        <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
                            <Button
                                variant="contained"
                                startIcon={<UploadIcon />}
                                onClick={() => setUploadDialogOpen(true)}
                            >
                                Upload Dataset
                            </Button>
                            <Button
                                variant="outlined"
                                startIcon={<AddIcon />}
                                onClick={() => setCreateDatasetDialogOpen(true)}
                            >
                                Create from Documents
                            </Button>
                        </Box>
                        
                        {/* Dataset List */}
                        <Card>
                            <CardContent>
                                <Typography variant="h6" sx={{ mb: 2 }}>
                                    Available Datasets
                                </Typography>
                                {datasets.length === 0 ? (
                                    <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
                                        No datasets available. Upload or create a dataset to get started.
                                    </Typography>
                                ) : (
                                    <TableContainer>
                                        <Table>
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell>Name</TableCell>
                                                    <TableCell>Size</TableCell>
                                                    <TableCell>Created</TableCell>
                                                    <TableCell>Actions</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                {datasets.map((dataset) => (
                                                    <TableRow key={dataset.id}>
                                                        <TableCell>
                                                            <Typography variant="subtitle2">
                                                                {dataset.name}
                                                            </Typography>
                                                        </TableCell>
                                                        <TableCell>
                                                            {dataset.size} examples
                                                        </TableCell>
                                                        <TableCell>
                                                            {formatDate(dataset.created_at)}
                                                        </TableCell>
                                                        <TableCell>
                                                            <Box sx={{ display: 'flex', gap: 1 }}>
                                                                <Tooltip title="View Details">
                                                                    <IconButton
                                                                        size="small"
                                                                        onClick={() => {
                                                                            setDatasetDetails(dataset);
                                                                            setViewDatasetDialogOpen(true);
                                                                        }}
                                                                    >
                                                                        <ViewIcon />
                                                                    </IconButton>
                                                                </Tooltip>
                                                                <Tooltip title="Download">
                                                                    <IconButton
                                                                        size="small"
                                                                        onClick={() => {
                                                                            // Download functionality
                                                                            window.open(`/api/datasets/${dataset.id}/download`, '_blank');
                                                                        }}
                                                                    >
                                                                        <DownloadIcon />
                                                                    </IconButton>
                                                                </Tooltip>
                                                                <Tooltip title="Delete">
                                                                    <IconButton
                                                                        size="small"
                                                                        color="error"
                                                                        onClick={() => {
                                                                            if (window.confirm(`Are you sure you want to delete dataset "${dataset.name}"?`)) {
                                                                                // Delete functionality
                                                                                handleDeleteDataset(dataset.id);
                                                                            }
                                                                        }}
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
                )}

                {sidebarSelection === 'comparison' && (
                    <Box sx={{ flex: 1, overflow: 'auto', p: 3 }}>
                        <Typography variant="h5" sx={{ mb: 3 }}>
                            Experiment Comparison
                        </Typography>
                        
                        {/* Experiment Selection for Comparison */}
                        <Card sx={{ mb: 3 }}>
                            <CardContent>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                    <Typography variant="h6">
                                        Select Experiments to Compare
                                    </Typography>
                                    <Box sx={{ display: 'flex', gap: 1 }}>
                                        <Button 
                                            size="small" 
                                            variant="outlined"
                                            onClick={() => setSelectedExperimentsForComparison([])}
                                            disabled={selectedExperimentsForComparison.length === 0}
                                        >
                                            Clear All
                                        </Button>
                                        <Chip 
                                            label={`${selectedExperimentsForComparison.length} selected`}
                                            color={selectedExperimentsForComparison.length > 1 ? 'primary' : 'default'}
                                            size="small"
                                        />
                                    </Box>
                                </Box>
                                
                                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                                    Select 2-4 experiments to compare their training metrics, performance, and configurations. 
                                    Click on experiment cards to select/deselect them.
                                </Typography>
                                
                                {experiments.length === 0 ? (
                                    <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
                                        No experiments available for comparison. Create some experiments first.
                                    </Typography>
                                ) : (
                                    <Grid container spacing={2}>
                                        {experiments.map((exp) => {
                                            const isSelected = selectedExperimentsForComparison.includes(exp.id);
                                            const canSelect = selectedExperimentsForComparison.length < 4 || isSelected;
                                            
                                            return (
                                                <Grid item xs={12} sm={6} md={3} key={exp.id}>
                                                    <Card 
                                                        sx={{ 
                                                            cursor: canSelect ? 'pointer' : 'not-allowed',
                                                            border: isSelected ? 2 : 1,
                                                            borderColor: isSelected ? 'primary.main' : 'divider',
                                                            backgroundColor: isSelected ? 'rgba(37, 99, 235, 0.04)' : 'background.paper',
                                                            opacity: canSelect ? 1 : 0.6,
                                                            '&:hover': canSelect ? { 
                                                                boxShadow: 2,
                                                                borderColor: 'primary.main'
                                                            } : {}
                                                        }}
                                                        onClick={() => {
                                                            if (!canSelect) return;
                                                            
                                                            setSelectedExperimentsForComparison(prev => {
                                                                if (prev.includes(exp.id)) {
                                                                    return prev.filter(id => id !== exp.id);
                                                                } else {
                                                                    return [...prev, exp.id];
                                                                }
                                                            });
                                                        }}
                                                    >
                                                        <CardContent sx={{ p: 2, position: 'relative' }}>
                                                            {isSelected && (
                                                                <Chip 
                                                                    label="✓" 
                                                                    size="small" 
                                                                    color="primary"
                                                                    sx={{ 
                                                                        position: 'absolute', 
                                                                        top: 8, 
                                                                        right: 8,
                                                                        minWidth: 24,
                                                                        height: 24
                                                                    }}
                                                                />
                                                            )}
                                                            <Typography variant="subtitle2" noWrap sx={{ pr: isSelected ? 4 : 0 }}>
                                                                {exp.name}
                                                            </Typography>
                                                            <Typography variant="body2" color="text.secondary" noWrap>
                                                                {exp.model_name}
                                                            </Typography>
                                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
                                                                <Chip 
                                                                    label={exp.status} 
                                                                    size="small"
                                                                    color={
                                                                        exp.status === 'running' ? 'primary' :
                                                                        exp.status === 'completed' ? 'success' :
                                                                        exp.status === 'failed' ? 'error' : 'default'
                                                                    }
                                                                />
                                                                <Typography variant="caption" color="text.secondary">
                                                                    {formatDate(exp.created_at || exp.timestamp)}
                                                                </Typography>
                                                            </Box>
                                                        </CardContent>
                                                    </Card>
                                                </Grid>
                                            );
                                        })}
                                    </Grid>
                                )}
                                
                                {selectedExperimentsForComparison.length > 0 && selectedExperimentsForComparison.length < 2 && (
                                    <Alert severity="info" sx={{ mt: 2 }}>
                                        Select at least 2 experiments to start comparison.
                                    </Alert>
                                )}
                                
                                {selectedExperimentsForComparison.length >= 4 && (
                                    <Alert severity="warning" sx={{ mt: 2 }}>
                                        Maximum 4 experiments can be compared at once.
                                    </Alert>
                                )}
                            </CardContent>
                        </Card>

                        {/* Comparison Results */}
                        {selectedExperimentsForComparison.length >= 2 && (
                            <Card>
                                <CardContent>
                                    <Typography variant="h6" sx={{ mb: 3 }}>
                                        Comparison Results ({selectedExperimentsForComparison.length} experiments)
                                    </Typography>
                                    
                                    {/* Quick Stats Table */}
                                    <TableContainer component={Paper} sx={{ mb: 3 }}>
                                        <Table size="small">
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell><strong>Experiment</strong></TableCell>
                                                    <TableCell><strong>Model</strong></TableCell>
                                                    <TableCell><strong>Status</strong></TableCell>
                                                    <TableCell><strong>Created</strong></TableCell>
                                                    <TableCell><strong>Actions</strong></TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                {experiments
                                                    .filter(exp => selectedExperimentsForComparison.includes(exp.id))
                                                    .map((exp) => (
                                                        <TableRow key={exp.id}>
                                                            <TableCell>
                                                                <Typography variant="subtitle2">
                                                                    {exp.name}
                                                                </Typography>
                                                                <Typography variant="caption" color="text.secondary">
                                                                    {exp.description || 'No description'}
                                                                </Typography>
                                                            </TableCell>
                                                            <TableCell>{exp.model_name || exp.base_model || 'N/A'}</TableCell>
                                                            <TableCell>
                                                                <Chip 
                                                                    label={exp.status} 
                                                                    size="small"
                                                                    color={
                                                                        exp.status === 'running' ? 'primary' :
                                                                        exp.status === 'completed' ? 'success' :
                                                                        exp.status === 'failed' ? 'error' : 'default'
                                                                    }
                                                                />
                                                            </TableCell>
                                                            <TableCell>
                                                                <Typography variant="body2">
                                                                    {formatDate(exp.created_at || exp.timestamp)}
                                                                </Typography>
                                                            </TableCell>
                                                            <TableCell>
                                                                <Box sx={{ display: 'flex', gap: 1 }}>
                                                                    <Button 
                                                                        size="small" 
                                                                        variant="outlined"
                                                                        onClick={() => {
                                                                            setComparisonDashboardExpId(
                                                                                comparisonDashboardExpId === exp.id ? null : exp.id
                                                                            );
                                                                        }}
                                                                    >
                                                                        {comparisonDashboardExpId === exp.id ? 'Hide Dashboard' : 'Dashboard'}
                                                                    </Button>
                                                                    {exp.status === 'running' && (
                                                                        <Button 
                                                                            size="small" 
                                                                            variant="outlined" 
                                                                            color="error"
                                                                            onClick={() => handleStopTraining(exp.id)}
                                                                        >
                                                                            Stop
                                                                        </Button>
                                                                    )}
                                                                </Box>
                                                            </TableCell>
                                                        </TableRow>
                                                    ))}
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                    
                                    {/* Detailed Parameter Comparison */}
                                    <Typography variant="h6" sx={{ mb: 2 }}>
                                        Parameter Comparison
                                    </Typography>
                                    
                                    <Grid container spacing={3}>
                                        {experiments
                                            .filter(exp => selectedExperimentsForComparison.includes(exp.id))
                                            .map((exp) => (
                                                <Grid item xs={12} md={6} lg={selectedExperimentsForComparison.length > 2 ? 6 : 12} key={exp.id}>
                                                    <Card variant="outlined">
                                                        <CardContent>
                                                            <Typography variant="h6" gutterBottom color="primary">
                                                                {exp.name}
                                                            </Typography>
                                                            
                                                            <Box sx={{ mt: 2 }}>
                                                                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                                                                    Configuration
                                                                </Typography>
                                                                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                                                                    <Chip label={`Model: ${exp.model_name || exp.base_model || 'N/A'}`} size="small" variant="outlined" />
                                                                    <Chip 
                                                                        label={`Status: ${exp.status}`} 
                                                                        size="small" 
                                                                        color={
                                                                            exp.status === 'running' ? 'primary' :
                                                                            exp.status === 'completed' ? 'success' :
                                                                            exp.status === 'failed' ? 'error' : 'default'
                                                                        } 
                                                                    />
                                                                </Box>
                                                            </Box>
                                                            
                                                            {exp.parameters && (
                                                                <Box sx={{ mt: 2 }}>
                                                                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                                                                        Training Parameters
                                                                    </Typography>
                                                                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                                                                        {exp.parameters.learning_rate && (
                                                                            <Chip label={`Learning Rate: ${exp.parameters.learning_rate}`} size="small" variant="outlined" />
                                                                        )}
                                                                        {exp.parameters.epochs && (
                                                                            <Chip label={`Epochs: ${exp.parameters.epochs}`} size="small" variant="outlined" />
                                                                        )}
                                                                        {exp.parameters.batch_size && (
                                                                            <Chip label={`Batch Size: ${exp.parameters.batch_size}`} size="small" variant="outlined" />
                                                                        )}
                                                                        {exp.parameters.lora_r && (
                                                                            <Chip label={`LoRA R: ${exp.parameters.lora_r}`} size="small" variant="outlined" />
                                                                        )}
                                                                        {exp.parameters.lora_alpha && (
                                                                            <Chip label={`LoRA Alpha: ${exp.parameters.lora_alpha}`} size="small" variant="outlined" />
                                                                        )}
                                                                    </Box>
                                                                </Box>
                                                            )}

                                                            <Box sx={{ mt: 2 }}>
                                                                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                                                                    Timeline
                                                                </Typography>
                                                                <Typography variant="body2">
                                                                    Created: {formatDate(exp.created_at || exp.timestamp)}
                                                                </Typography>
                                                                {exp.started_at && (
                                                                    <Typography variant="body2">
                                                                        Started: {formatDate(exp.started_at)}
                                                                    </Typography>
                                                                )}
                                                                {exp.completed_at && (
                                                                    <Typography variant="body2">
                                                                        Completed: {formatDate(exp.completed_at)}
                                                                    </Typography>
                                                                )}
                                                            </Box>
                                                        </CardContent>
                                                    </Card>
                                                </Grid>
                                            ))}
                                    </Grid>
                                    
                                    {/* Inline Experiment Dashboard */}
                                    {comparisonDashboardExpId && (
                                        <Box sx={{ mt: 4 }}>
                                            <Typography variant="h6" gutterBottom sx={{ 
                                                fontWeight: 600,
                                                color: '#0f172a',
                                                mb: 3
                                            }}>
                                                📈 Training Dashboard: {experiments.find(e => e.id === comparisonDashboardExpId)?.name}
                                            </Typography>
                                            
                                            <Card>
                                                <CardContent sx={{ p: 2 }}>
                                                    <TrainingDashboard
                                                        experimentId={comparisonDashboardExpId}
                                                        onClose={() => setComparisonDashboardExpId(null)}
                                                        embedded={true}
                                                    />
                                                </CardContent>
                                            </Card>
                                        </Box>
                                    )}
                                    
                                    {/* Comparison Charts */}
                                    {selectedExperimentsForComparison.length >= 2 && (
                                        <Box sx={{ mt: 4 }}>
                                            <Typography variant="h6" gutterBottom sx={{ 
                                                fontWeight: 600,
                                                color: '#0f172a',
                                                mb: 3
                                            }}>
                                                📊 Training Metrics Comparison ({selectedExperimentsForComparison.length} experiments)
                                            </Typography>
                                            
                                            {loadingMetrics ? (
                                                <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                                                    <CircularProgress />
                                                </Box>
                                            ) : Object.keys(comparisonMetrics).length > 0 ? (
                                                <Grid container spacing={3}>
                                                    {/* Training Loss Comparison */}
                                                    <Grid item xs={12} md={6}>
                                                        <Card sx={{ height: '400px' }}>
                                                            <CardContent sx={{ height: '100%', p: 2 }}>
                                                                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
                                                                    Training Loss Comparison
                                                                </Typography>
                                                                <MultiExperimentLossChart 
                                                                    experimentsData={comparisonMetrics}
                                                                    experiments={experiments.filter(exp => selectedExperimentsForComparison.includes(exp.id))}
                                                                />
                                                            </CardContent>
                                                        </Card>
                                                    </Grid>
                                                    
                                                    {/* Learning Rate Comparison */}
                                                    <Grid item xs={12} md={6}>
                                                        <Card sx={{ height: '400px' }}>
                                                            <CardContent sx={{ height: '100%', p: 2 }}>
                                                                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
                                                                    Learning Rate Comparison
                                                                </Typography>
                                                                <MultiExperimentLearningRateChart 
                                                                    experimentsData={comparisonMetrics}
                                                                    experiments={experiments.filter(exp => selectedExperimentsForComparison.includes(exp.id))}
                                                                />
                                                            </CardContent>
                                                        </Card>
                                                    </Grid>
                                                    
                                                    {/* Resource Usage Comparison */}
                                                    <Grid item xs={12}>
                                                        <Card>
                                                            <CardContent sx={{ p: 2 }}>
                                                                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 600, mb: 2 }}>
                                                                    Peak Resource Usage Comparison
                                                                </Typography>
                                                                <ResourceUsageBarChart 
                                                                    experimentsData={comparisonMetrics}
                                                                    experiments={experiments.filter(exp => selectedExperimentsForComparison.includes(exp.id))}
                                                                />
                                                            </CardContent>
                                                        </Card>
                                                    </Grid>
                                                </Grid>
                                            ) : (
                                                <Alert severity="warning" sx={{ mt: 2 }}>
                                                    📊 No training metrics available for the selected experiments. Charts will appear here once experiments have completed training with recorded metrics.
                                                </Alert>
                                            )}
                                        </Box>
                                    )}
                                </CardContent>
                            </Card>
                        )}
                    </Box>
                )}
            </Box>

            {/* Dataset Upload Dialog */}
            <Dialog
                open={uploadDialogOpen}
                onClose={() => setUploadDialogOpen(false)}
                maxWidth="md"
                fullWidth
            >
                <DialogTitle>Upload Dataset</DialogTitle>
                <DialogContent>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                        Upload a dataset file in CSV or JSON format for fine-tuning.
                    </Typography>
                    {/* Add file upload component here */}
                    <Button variant="outlined" component="label" sx={{ mb: 2 }}>
                        Choose File
                        <input type="file" hidden accept=".csv,.json,.jsonl" />
                    </Button>
                    <Typography variant="body2" color="text.secondary">
                        Supported formats: CSV, JSON, JSONL
                    </Typography>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setUploadDialogOpen(false)}>Cancel</Button>
                    <Button variant="contained">Upload</Button>
                </DialogActions>
            </Dialog>

            {/* Create Dataset Dialog */}
            <Dialog
                open={createDatasetDialogOpen}
                onClose={() => setCreateDatasetDialogOpen(false)}
                maxWidth="md"
                fullWidth
            >
                <DialogTitle>Create Dataset from Documents</DialogTitle>
                <DialogContent>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                        Create a Q&A dataset by generating questions from your documents.
                    </Typography>
                    
                    <TextField
                        fullWidth
                        label="Dataset Name"
                        value={newDatasetName}
                        onChange={(e) => setNewDatasetName(e.target.value)}
                        sx={{ mb: 2 }}
                        required
                    />
                    
                    <TextField
                        fullWidth
                        label="Description"
                        multiline
                        rows={3}
                        value={newDatasetDescription}
                        onChange={(e) => setNewDatasetDescription(e.target.value)}
                        sx={{ mb: 2 }}
                    />
                    
                    <TextField
                        fullWidth
                        label="Questions per Document"
                        type="number"
                        value={questionsPerDoc}
                        onChange={(e) => setQuestionsPerDoc(parseInt(e.target.value))}
                        inputProps={{ min: 1, max: 10 }}
                        sx={{ mb: 3 }}
                    />
                    
                    <Typography variant="subtitle2" sx={{ mb: 2 }}>
                        Select Documents:
                    </Typography>
                    
                    <Box sx={{ maxHeight: 300, overflow: 'auto', border: 1, borderColor: 'divider', borderRadius: 1 }}>
                        {availableDocuments.length === 0 ? (
                            <Typography variant="body2" color="text.secondary" sx={{ p: 2 }}>
                                No documents available. Upload some documents first.
                            </Typography>
                        ) : (
                            availableDocuments.map((doc) => (
                                <Box key={doc.id} sx={{ p: 1, borderBottom: 1, borderColor: 'divider' }}>
                                    <FormControlLabel
                                        control={
                                            <Checkbox
                                                checked={selectedDocuments.includes(doc.id)}
                                                onChange={(e) => {
                                                    if (e.target.checked) {
                                                        setSelectedDocuments([...selectedDocuments, doc.id]);
                                                    } else {
                                                        setSelectedDocuments(selectedDocuments.filter(id => id !== doc.id));
                                                    }
                                                }}
                                            />
                                        }
                                        label={doc.title || doc.filename || `Document ${doc.id}`}
                                    />
                                </Box>
                            ))
                        )}
                    </Box>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setCreateDatasetDialogOpen(false)}>Cancel</Button>
                    <Button 
                        variant="contained" 
                        onClick={handleCreateDataset}
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
                <DialogTitle>
                    Dataset Details
                    {datasetDetails && (
                        <Typography variant="subtitle2" color="text.secondary">
                            {datasetDetails.name}
                        </Typography>
                    )}
                </DialogTitle>
                <DialogContent>
                    {datasetDetails ? (
                        <Box>
                            <Grid container spacing={2} sx={{ mb: 3 }}>
                                <Grid item xs={6}>
                                    <Typography variant="body2" color="text.secondary">Description</Typography>
                                    <Typography variant="body1">{datasetDetails.description || 'No description'}</Typography>
                                </Grid>
                                <Grid item xs={6}>
                                    <Typography variant="body2" color="text.secondary">Size</Typography>
                                    <Typography variant="body1">{datasetDetails.size} examples</Typography>
                                </Grid>
                                <Grid item xs={6}>
                                    <Typography variant="body2" color="text.secondary">Created</Typography>
                                    <Typography variant="body1">{formatDate(datasetDetails.created_at)}</Typography>
                                </Grid>
                                <Grid item xs={6}>
                                    <Typography variant="body2" color="text.secondary">Format</Typography>
                                    <Typography variant="body1">{datasetDetails.format || 'JSON'}</Typography>
                                </Grid>
                            </Grid>
                            
                            {datasetDetails.samples && (
                                <Box>
                                    <Typography variant="h6" sx={{ mb: 2 }}>Sample Data</Typography>
                                    <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                                        {datasetDetails.samples.slice(0, 5).map((sample, index) => (
                                            <Card key={index} sx={{ mb: 2, p: 2 }}>
                                                <Typography variant="subtitle2">Example {index + 1}</Typography>
                                                <Typography variant="body2" sx={{ mt: 1 }}>
                                                    <strong>Input:</strong> {sample.input || sample.question || 'No input'}
                                                </Typography>
                                                <Typography variant="body2" sx={{ mt: 1 }}>
                                                    <strong>Output:</strong> {sample.output || sample.answer || 'No output'}
                                                </Typography>
                                            </Card>
                                        ))}
                                    </Box>
                                </Box>
                            )}
                        </Box>
                    ) : (
                        <Typography>Loading dataset details...</Typography>
                    )}
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setViewDatasetDialogOpen(false)}>Close</Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
};

export default FinetuningPage;
