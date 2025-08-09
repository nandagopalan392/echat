import React, { useState, useEffect } from 'react';
import { 
    Box, 
    Card, 
    CardContent, 
    Typography, 
    Grid, 
    Button,
    Chip,
    Container,
    AppBar,
    Toolbar,
    IconButton,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    ThemeProvider,
    LinearProgress,
    Alert,
    Fade,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    TextField,
    List,
    ListItem,
    ListItemText,
    ListItemSecondaryAction,
    Checkbox,
    FormControlLabel,
    Switch,
    Drawer,
    ListItemButton,
    ListItemIcon,
    Divider,
    CircularProgress,
    Tabs,
    Tab,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    OutlinedInput
} from '@mui/material';
import { 
    ArrowBack, 
    Assessment, 
    TrendingUp, 
    Timer, 
    CheckCircle,
    BarChart,
    Refresh,
    Download,
    FilterList,
    Add,
    Delete,
    PlayArrow,
    Stop,
    Upload,
    Folder,
    Description,
    Dashboard,
    DataUsage,
    Assignment,
    Timeline,
    Close,
    ExpandMore
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';
import { theme } from '../theme';
import { api, evaluationApi } from '../services/api';

const SIDEBAR_WIDTH = 280;

const EvaluationPage = () => {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState(0); // 0: Overview, 1: Datasets, 2: Evaluation
    const [error, setError] = useState('');
    const [timeRange, setTimeRange] = useState('7d');
    
    // Dataset management state
    const [datasets, setDatasets] = useState([]);
    const [createDatasetOpen, setCreateDatasetOpen] = useState(false);
    const [newDatasetName, setNewDatasetName] = useState('');
    const [newDatasetDescription, setNewDatasetDescription] = useState('');
    const [selectedDocuments, setSelectedDocuments] = useState([]);
    const [availableDocuments, setAvailableDocuments] = useState([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [numQuestionsPerDoc, setNumQuestionsPerDoc] = useState(3);
    const [difficultyLevels, setDifficultyLevels] = useState(['easy', 'medium', 'hard']);
    
    // Dataset generation progress state
    const [showProgressDialog, setShowProgressDialog] = useState(false);
    const [datasetGenerationProgress, setDatasetGenerationProgress] = useState({
        status: 'starting',
        progress: 0,
        current_document: '',
        total_documents: 0,
        completed_documents: 0,
        error: null
    });
    const [currentDatasetId, setCurrentDatasetId] = useState(null);
    
    // Test case state
    const [testCases, setTestCases] = useState([]);
    const [createTestCaseOpen, setCreateTestCaseOpen] = useState(false);
    const [selectedDataset, setSelectedDataset] = useState('');
    const [selectedModels, setSelectedModels] = useState([]);
    const [selectedRetriever, setSelectedRetriever] = useState('');
    // Dialog-specific state variables
    const [selectedDatasetId, setSelectedDatasetId] = useState('');
    const [selectedModelId, setSelectedModelId] = useState('');
    const [selectedRetrieverId, setSelectedRetrieverId] = useState('');
    // Test case retrieval configuration
    const [testRetrievalConfig, setTestRetrievalConfig] = useState({
        similarity_threshold: 0.2,
        keyword_similarity_weight: 0.7,
        max_chunks: 5,
        search_type: 'similarity',
        reranker_enabled: false,
        reranker_model: '',
        auto_merging_enabled: false,
        auto_merging_similarity_threshold: 0.8
    });
    const [showAdvancedRetrieval, setShowAdvancedRetrieval] = useState(false);
    const [availableModels, setAvailableModels] = useState([]);
    const [availableRetrievers, setAvailableRetrievers] = useState([]);
    const [rerankerModels, setRerankerModels] = useState([]);
    const [runningTestCase, setRunningTestCase] = useState(null);
    const [testCaseResults, setTestCaseResults] = useState([]);
    
    // Evaluation results state
    const [evaluationResults, setEvaluationResults] = useState([]);
    const [filteredResults, setFilteredResults] = useState([]);
    const [filterModel, setFilterModel] = useState('');
    const [filterDataset, setFilterDataset] = useState('');
    
    // Master-Detail view state for dataset contents
    const [selectedDatasetForDetail, setSelectedDatasetForDetail] = useState(null);
    const [datasetDetails, setDatasetDetails] = useState(null);
    const [loadingDatasetDetails, setLoadingDatasetDetails] = useState(false);
    
    const [evaluationData, setEvaluationData] = useState({
        overall: {
            groundedness: 0,
            contextRelevance: 0,
            answerQuality: 0,
            averageLatency: 0
        },
        historical: [],
        detailed: []
    });
    const [refreshing, setRefreshing] = useState(false);

    useEffect(() => {
        loadEvaluationData();
        loadDatasets();
        loadAvailableDocuments();
        loadAvailableModels();
        loadAvailableRetrievers();
        loadRerankerModels();
        loadEvaluationResults();
        loadTestCaseResults();
    }, [timeRange]);

    // Load datasets
    const loadDatasets = async () => {
        try {
            const response = await evaluationApi.getDatasets();
            // Transform the data to match frontend expectations and format dates
            const transformedDatasets = (response.datasets || []).map(dataset => ({
                ...dataset,
                documentCount: dataset.document_count, // Map backend field to frontend field
                createdAt: dataset.created_at ? new Date(dataset.created_at).toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                }) : 'Unknown'
            }));
            setDatasets(transformedDatasets);
        } catch (error) {
            console.error('Error loading datasets:', error);
            // Show empty array on error instead of mock data for datasets tab
            setDatasets([]);
        }
    };

    // Handle dataset selection for master-detail view
    const handleDatasetClick = async (dataset) => {
        setSelectedDatasetForDetail(dataset);
        setLoadingDatasetDetails(true);
        setDatasetDetails(null);
        
        try {
            const response = await evaluationApi.getDatasetDetails(dataset.id);
            setDatasetDetails(response);
        } catch (error) {
            console.error('Error loading dataset details:', error);
            setError(`Failed to load dataset details: ${error.message || 'Unknown error'}`);
        } finally {
            setLoadingDatasetDetails(false);
        }
    };

    // Close master-detail view
    const handleCloseDatasetDetail = () => {
        setSelectedDatasetForDetail(null);
        setDatasetDetails(null);
        setLoadingDatasetDetails(false);
    };

    // Load available documents
    const loadAvailableDocuments = async () => {
        try {
            const response = await api.call('/api/documents');
            setAvailableDocuments(response.documents || []);
        } catch (error) {
            console.error('Error loading documents:', error);
            // Show empty array instead of mock data for dataset creation
            setAvailableDocuments([]);
        }
    };

    // Load available models
    const loadAvailableModels = async () => {
        try {
            const response = await api.call('/api/models/available');
            setAvailableModels(response.llm_models || []);
        } catch (error) {
            console.error('Error loading models:', error);
            // Show empty array instead of mock data
            setAvailableModels([]);
        }
    };

    // Load available retrievers
    const loadAvailableRetrievers = async () => {
        try {
            // Add retrievers list - these would typically come from an API
            setAvailableRetrievers([
                { id: 'vector', name: 'Vector Search', description: 'Standard vector similarity search' },
                { id: 'bm25', name: 'BM25', description: 'Traditional keyword-based search' },
                { id: 'hybrid', name: 'Hybrid Search', description: 'Combination of vector and BM25' },
                { id: 'rerank', name: 'Reranked Search', description: 'Vector search with reranking' }
            ]);
        } catch (error) {
            console.error('Error loading retrievers:', error);
            setAvailableRetrievers([]);
        }
    };

    // Delete dataset function
    const handleDeleteDataset = async (datasetId, datasetName) => {
        if (!window.confirm(`Are you sure you want to delete the dataset "${datasetName}"? This action cannot be undone.`)) {
            return;
        }
        
        try {
            await evaluationApi.deleteDataset(datasetId);
            // Refresh the datasets list
            await loadDatasets();
            setError(''); // Clear any previous errors
        } catch (error) {
            console.error('Error deleting dataset:', error);
            setError(`Failed to delete dataset: ${error.message || 'Unknown error'}`);
        }
    };

    // Load evaluation results
    const loadEvaluationResults = async () => {
        try {
            const response = await evaluationApi.getResults();
            setEvaluationResults(response.results || []);
            setFilteredResults(response.results || []);
        } catch (error) {
            console.error('Error loading evaluation results:', error);
            // Mock evaluation results as fallback
            const mockResults = [
                {
                    id: 1,
                    dataset: "Customer Support QA",
                    model: "llama3.1",
                    groundedness: 0.85,
                    contextRelevance: 0.78,
                    answerQuality: 0.82,
                    avgLatency: 1.2,
                    runDate: "2024-08-03",
                    status: "Completed"
                },
                {
                    id: 2,
                    dataset: "Technical Documentation",
                    model: "mistral",
                    groundedness: 0.79,
                    contextRelevance: 0.81,
                    answerQuality: 0.77,
                    avgLatency: 0.9,
                    runDate: "2024-08-02",
                    status: "Completed"
                },
                {
                    id: 3,
                    dataset: "Customer Support QA",
                    model: "phi3",
                    groundedness: 0.73,
                    contextRelevance: 0.75,
                    answerQuality: 0.71,
                    avgLatency: 0.7,
                    runDate: "2024-08-01",
                    status: "Completed"
                }
            ];
            setEvaluationResults(mockResults);
            setFilteredResults(mockResults);
        }
    };

    // Load test case results from database
    const loadTestCaseResults = async () => {
        try {
            const response = await evaluationApi.getResults();
            console.log('Loaded test case results:', response);
            
            // Transform the results to match the expected format for the test case table
            const transformedResults = (response.results || []).map(result => ({
                id: result.id,
                dataset_id: result.dataset_id,
                dataset_name: result.dataset,
                models: [result.model], // Convert single model to array for compatibility
                results: [{
                    model: result.model,
                    groundedness: result.groundedness,
                    context_relevance: result.context_relevance,
                    answer_quality: result.answer_quality,
                    avg_latency: result.avg_latency,
                    total_questions: result.total_questions
                }],
                status: result.status.toLowerCase(),
                created_at: result.started_at || result.run_date,
                completed_at: result.completed_at
            }));
            
            setTestCaseResults(transformedResults);
        } catch (error) {
            console.error('Error loading test case results:', error);
            // Keep empty array on error - real results will be added when tests are run
            setTestCaseResults([]);
        }
    };

    // Tab change handler
    const handleTabChange = (event, newValue) => {
        setActiveTab(newValue);
    };

    // Create dataset handler
    const handleCreateDataset = async () => {
        try {
            if (!newDatasetName.trim()) {
                setError('Dataset name is required');
                return;
            }

            if (selectedDocuments.length === 0) {
                setError('Please select at least one document');
                return;
            }

            if (!selectedModel) {
                setError('Please select a model for dataset generation');
                return;
            }

            const datasetConfig = {
                name: newDatasetName,
                description: newDatasetDescription,
                document_ids: selectedDocuments,
                model_name: selectedModel,
                num_questions_per_doc: numQuestionsPerDoc,
                difficulty_levels: difficultyLevels
            };

            // Close creation dialog and show progress dialog IMMEDIATELY
            setCreateDatasetOpen(false);
            console.log('Setting showProgressDialog to true');
            setShowProgressDialog(true);
            setError(''); // Clear any previous errors
            
            // Reset the progress state to initial
            setDatasetGenerationProgress({
                status: 'starting',
                progress: 0,
                current_document: '',
                total_documents: selectedDocuments.length,
                completed_documents: 0,
                error: null
            });
            console.log('Progress dialog should now be visible with initial state');

            // Start dataset creation
            console.log('Sending dataset creation request:', datasetConfig);
            const response = await evaluationApi.createDataset(datasetConfig);
            console.log('Dataset creation response:', response);

            if (response.dataset_id) {
                console.log('Setting current dataset ID:', response.dataset_id);
                setCurrentDatasetId(response.dataset_id);
                
                // Start polling for progress
                pollDatasetProgress(response.dataset_id);
                
                // Reset form
                setNewDatasetName('');
                setNewDatasetDescription('');
                setSelectedDocuments([]);
                setSelectedModel('');
                setNumQuestionsPerDoc(3);
                setDifficultyLevels(['easy', 'medium', 'hard']);
            } else {
                // Handle case where response doesn't have dataset_id
                setDatasetGenerationProgress(prev => ({
                    ...prev,
                    status: 'error',
                    error: 'Failed to start dataset creation - no dataset ID returned'
                }));
            }
        } catch (error) {
            // Update progress dialog to show error instead of hiding it
            setDatasetGenerationProgress(prev => ({
                ...prev,
                status: 'error',
                error: 'Failed to create dataset: ' + (error.message || 'Unknown error')
            }));
            console.error('Error creating dataset:', error);
        }
    };

    // Poll dataset generation progress
    const pollDatasetProgress = async (datasetId) => {
        console.log('Starting progress polling for dataset ID:', datasetId);
        
        const pollInterval = setInterval(async () => {
            try {
                console.log('Polling progress for dataset ID:', datasetId);
                const progress = await evaluationApi.getDatasetProgress(datasetId);
                console.log('Progress response:', progress);
                setDatasetGenerationProgress(progress);
                
                if (progress.status === 'completed') {
                    console.log('Dataset generation completed');
                    clearInterval(pollInterval);
                    // Refresh datasets list
                    await loadDatasets();
                    // Auto-close progress dialog after a short delay
                    setTimeout(() => {
                        setShowProgressDialog(false);
                        setCurrentDatasetId(null);
                    }, 2000);
                } else if (progress.status === 'error') {
                    console.log('Dataset generation failed:', progress.error);
                    clearInterval(pollInterval);
                    setError('Dataset generation failed: ' + (progress.error || 'Unknown error'));
                    setTimeout(() => {
                        setShowProgressDialog(false);
                        setCurrentDatasetId(null);
                    }, 3000);
                }
            } catch (error) {
                console.error('Error polling progress:', error);
                clearInterval(pollInterval);
                setError('Failed to get progress updates');
                setTimeout(() => {
                    setShowProgressDialog(false);
                    setCurrentDatasetId(null);
                }, 3000);
            }
        }, 2000); // Poll every 2 seconds
    };

    // Run test case handler
    const handleRunTestCase = async (datasetId, models, retriever) => {
        let testCaseId;
        try {
            setRunningTestCase(`${datasetId}-${retriever || 'default'}`);
            
            // Create test case entry
            testCaseId = Date.now();
            const testCase = {
                id: testCaseId,
                dataset_id: datasetId,
                dataset_name: datasets.find(d => d.id === datasetId)?.name || 'Unknown Dataset',
                models: models,
                retriever: retriever || 'vector',
                status: 'running',
                created_at: new Date().toISOString(),
                results: null
            };
            
            setTestCaseResults(prev => [...prev, testCase]);
            
            const response = await evaluationApi.runTestCase(datasetId, models, retriever);
            
            if (response.success) {
                // Update test case with results
                setTestCaseResults(prev => 
                    prev.map(tc => 
                        tc.id === testCaseId 
                            ? { ...tc, status: 'completed', results: response.results, completed_at: new Date().toISOString() }
                            : tc
                    )
                );
                
                setEvaluationResults([...evaluationResults, ...response.results]);
                setFilteredResults([...filteredResults, ...response.results]);
            } else {
                // Update test case with error
                setTestCaseResults(prev => 
                    prev.map(tc => 
                        tc.id === testCaseId 
                            ? { ...tc, status: 'failed', error: response.error, completed_at: new Date().toISOString() }
                            : tc
                    )
                );
            }
            
            setRunningTestCase(null);
        } catch (error) {
            setError('Failed to run test case');
            setRunningTestCase(null);
            
            // Update test case with error if testCaseId exists
            if (testCaseId) {
                setTestCaseResults(prev => 
                    prev.map(tc => 
                        tc.id === testCaseId 
                            ? { ...tc, status: 'failed', error: error.message, completed_at: new Date().toISOString() }
                            : tc
                    )
                );
            }
            
            console.error('Error running test case:', error);
        }
    };

    // Handle test case creation from dialog
    const handleCreateTestCase = async () => {
        if (!selectedDatasetId || !selectedModelId) {
            setError('Please select dataset and model');
            return;
        }

        try {
            setRunningTestCase(true);
            setError(null);

            const testCaseData = {
                dataset_id: selectedDatasetId,
                models: [selectedModelId], // Convert single model to array for backend compatibility
                retriever: 'vector_similarity', // Default retriever
                retrieval_config: testRetrievalConfig,
                created_at: new Date().toISOString()
            };

            const response = await fetch('/api/evaluation/test-cases', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify(testCaseData)
            });

            if (!response.ok) {
                throw new Error('Failed to create test case');
            }

            const result = await response.json();
            
            // Add to test case results
            const newTestCase = {
                id: result.test_case_id,
                dataset_name: datasets.find(d => d.id === selectedDatasetId)?.name || 'Unknown',
                models: [selectedModelId], // Convert single model to array for backend compatibility
                retriever: 'Vector Similarity', // Default retriever name
                status: 'running',
                created_at: new Date().toISOString(),
                retrieval_config: testRetrievalConfig
            };

            setTestCaseResults(prev => [newTestCase, ...prev]);
            
            // Close dialog and reset form
            setCreateTestCaseOpen(false);
            setSelectedDatasetId('');
            setSelectedModelId('');
            
            // Start polling for test case status
            pollTestCaseStatus(result.test_case_id);

        } catch (error) {
            console.error('Error creating test case:', error);
            setError('Failed to create test case: ' + error.message);
        } finally {
            setRunningTestCase(false);
        }
    };

    const handleTestRetrievalConfigChange = (key, value) => {
        setTestRetrievalConfig(prev => ({
            ...prev,
            [key]: value
        }));
    };

    // Load reranker models for test case configuration
    const loadRerankerModels = async () => {
        try {
            const response = await fetch('/api/retrieval/reranker-models', {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });

            if (response.ok) {
                const data = await response.json();
                // Extract models array from the response object
                const models = data.models || [];
                // Ensure models is always an array
                const modelArray = Array.isArray(models) ? models : [];
                setRerankerModels(modelArray);
                
                // Set default reranker model if none selected
                if (modelArray.length > 0 && !testRetrievalConfig.reranker_model) {
                    setTestRetrievalConfig(prev => ({
                        ...prev,
                        reranker_model: modelArray[0].name
                    }));
                }
            } else {
                console.error('Failed to load reranker models:', response.status);
                setRerankerModels([]);
            }
        } catch (error) {
            console.error('Error loading reranker models:', error);
            setRerankerModels([]);
        }
    };

    const pollTestCaseStatus = async (testCaseId) => {
        const poll = async () => {
            try {
                const response = await fetch(`/api/evaluation/test-cases/${testCaseId}`, {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('token')}`
                    }
                });

                if (response.ok) {
                    const testCase = await response.json();
                    
                    setTestCaseResults(prev => 
                        prev.map(tc => 
                            tc.id === testCaseId 
                                ? { ...tc, ...testCase }
                                : tc
                        )
                    );

                    // Continue polling if still running
                    if (testCase.status === 'running') {
                        setTimeout(poll, 2000); // Poll every 2 seconds
                    }
                }
            } catch (error) {
                console.error('Error polling test case status:', error);
            }
        };

        poll();
    };

    // Filter evaluation results
    const filterResults = () => {
        let filtered = evaluationResults;
        
        if (filterModel) {
            filtered = filtered.filter(result => result.model === filterModel);
        }
        
        if (filterDataset) {
            filtered = filtered.filter(result => result.dataset === filterDataset);
        }
        
        setFilteredResults(filtered);
    };

    useEffect(() => {
        filterResults();
    }, [filterModel, filterDataset, evaluationResults]);

    const loadEvaluationData = async () => {
        try {
            setLoading(true);
            setError('');
            
            // Load real evaluation overview data from new API endpoint
            const response = await api.call(`/api/evaluation/overview?time_range=${timeRange}`);
            
            // Use the real data structure from the API
            const transformedData = {
                overall: {
                    groundedness: response.overall.groundedness,
                    contextRelevance: response.overall.contextRelevance,
                    answerQuality: response.overall.answerQuality,
                    averageLatency: response.overall.averageLatency,
                    totalQueries: response.overall.totalEvaluations
                },
                historical: response.historical.map(item => ({
                    date: item.date,
                    groundedness: item.groundedness,
                    contextRelevance: item.contextRelevance,
                    answerQuality: item.answerQuality,
                    latency: item.latency,
                    queries: item.queries
                })),
                latencyDistribution: response.latencyDistribution.map(item => ({
                    range: item.range,
                    count: item.count
                })),
                detailed: response.detailed.map(item => ({
                    query: item.query,
                    groundedness: item.groundedness,
                    contextRelevance: item.contextRelevance,
                    answerQuality: item.answerQuality,
                    latency: item.latency,
                    timestamp: item.timestamp,
                    model: item.model,
                    dataset: item.dataset
                }))
            };
            
            setEvaluationData(transformedData);
            
        } catch (error) {
            console.error('Error loading evaluation data:', error);
            setError('Failed to load evaluation data. Please try again.');
            
            // Use empty data as fallback instead of mock data
            const emptyData = {
                overall: {
                    groundedness: 0,
                    contextRelevance: 0,
                    answerQuality: 0,
                    averageLatency: 0,
                    totalQueries: 0
                },
                historical: [],
                latencyDistribution: [],
                detailed: []
            };
            setEvaluationData(emptyData);
        } finally {
            setLoading(false);
        }
    };

    const handleRefresh = async () => {
        setRefreshing(true);
        await loadEvaluationData();
        setRefreshing(false);
    };

    const getMetricColor = (value, isLatency = false) => {
        if (isLatency) {
            if (value < 1000) return theme.palette.success.main;
            if (value < 2000) return theme.palette.warning.main;
            return theme.palette.error.main;
        } else {
            if (value >= 0.8) return theme.palette.success.main;
            if (value >= 0.6) return theme.palette.warning.main;
            return theme.palette.error.main;
        }
    };

    const getScoreLabel = (value) => {
        if (value >= 0.8) return 'Excellent';
        if (value >= 0.6) return 'Good';
        if (value >= 0.4) return 'Fair';
        return 'Poor';
    };

    const metricsOverviewOptions = {
        title: {
            text: 'Evaluation Metrics Overview',
            left: 'center',
            textStyle: {
                fontSize: 18,
                fontWeight: 'bold',
                color: theme.palette.text.primary
            }
        },
        tooltip: {
            trigger: 'item',
            formatter: '{a} <br/>{b}: {c}% ({d}%)'
        },
        legend: {
            orient: 'vertical',
            left: 'left',
            textStyle: {
                color: theme.palette.text.primary
            }
        },
        series: [
            {
                name: 'Metrics',
                type: 'pie',
                radius: ['40%', '70%'],
                center: ['50%', '60%'],
                data: [
                    { 
                        value: (evaluationData.overall.groundedness * 100).toFixed(1), 
                        name: 'Groundedness',
                        itemStyle: { color: theme.palette.success.main }
                    },
                    { 
                        value: (evaluationData.overall.contextRelevance * 100).toFixed(1), 
                        name: 'Context Relevance',
                        itemStyle: { color: theme.palette.primary.main }
                    },
                    { 
                        value: (evaluationData.overall.answerQuality * 100).toFixed(1), 
                        name: 'Answer Quality',
                        itemStyle: { color: theme.palette.secondary.main }
                    }
                ],
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowOffsetX: 0,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                },
                label: {
                    show: true,
                    formatter: '{b}: {c}%'
                }
            }
        ]
    };

    const historicalTrendsOptions = {
        title: {
            text: 'Historical Performance Trends',
            left: 'center',
            textStyle: {
                color: theme.palette.text.primary
            }
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'cross'
            }
        },
        legend: {
            data: ['Groundedness', 'Context Relevance', 'Answer Quality'],
            top: 40,
            textStyle: {
                color: theme.palette.text.primary
            }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            boundaryGap: false,
            data: evaluationData.historical.map(item => item.date),
            axisLabel: {
                color: theme.palette.text.secondary
            }
        },
        yAxis: {
            type: 'value',
            min: 0,
            max: 1,
            axisLabel: {
                formatter: '{value}',
                color: theme.palette.text.secondary
            }
        },
        series: [
            {
                name: 'Groundedness',
                type: 'line',
                data: evaluationData.historical.map(item => item.groundedness.toFixed(3)),
                smooth: true,
                lineStyle: {
                    color: theme.palette.success.main,
                    width: 3
                },
                areaStyle: {
                    color: {
                        type: 'linear',
                        x: 0, y: 0, x2: 0, y2: 1,
                        colorStops: [
                            { offset: 0, color: theme.palette.success.light + '40' },
                            { offset: 1, color: theme.palette.success.light + '10' }
                        ]
                    }
                }
            },
            {
                name: 'Context Relevance',
                type: 'line',
                data: evaluationData.historical.map(item => item.contextRelevance.toFixed(3)),
                smooth: true,
                lineStyle: {
                    color: theme.palette.primary.main,
                    width: 3
                },
                areaStyle: {
                    color: {
                        type: 'linear',
                        x: 0, y: 0, x2: 0, y2: 1,
                        colorStops: [
                            { offset: 0, color: theme.palette.primary.light + '40' },
                            { offset: 1, color: theme.palette.primary.light + '10' }
                        ]
                    }
                }
            },
            {
                name: 'Answer Quality',
                type: 'line',
                data: evaluationData.historical.map(item => item.answerQuality.toFixed(3)),
                smooth: true,
                lineStyle: {
                    color: theme.palette.secondary.main,
                    width: 3
                },
                areaStyle: {
                    color: {
                        type: 'linear',
                        x: 0, y: 0, x2: 0, y2: 1,
                        colorStops: [
                            { offset: 0, color: theme.palette.secondary.light + '40' },
                            { offset: 1, color: theme.palette.secondary.light + '10' }
                        ]
                    }
                }
            }
        ]
    };

    const latencyTrendsOptions = {
        title: {
            text: 'Response Latency Analysis',
            left: 'center',
            textStyle: {
                color: theme.palette.text.primary
            }
        },
        tooltip: {
            trigger: 'axis',
            formatter: function (params) {
                return `${params[0].name}<br/>Latency: ${params[0].value}ms`;
            }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: evaluationData.historical.map(item => item.date),
            axisLabel: {
                color: theme.palette.text.secondary
            }
        },
        yAxis: {
            type: 'value',
            axisLabel: {
                formatter: '{value}ms',
                color: theme.palette.text.secondary
            }
        },
        series: [
            {
                name: 'Latency',
                data: evaluationData.historical.map(item => Math.round(item.latency)),
                type: 'bar',
                itemStyle: {
                    color: {
                        type: 'linear',
                        x: 0, y: 0, x2: 0, y2: 1,
                        colorStops: [
                            { offset: 0, color: theme.palette.info.light },
                            { offset: 1, color: theme.palette.info.main }
                        ]
                    }
                },
                emphasis: {
                    itemStyle: {
                        color: theme.palette.info.dark
                    }
                }
            }
        ]
    };

    if (loading) {
        return (
            <ThemeProvider theme={theme}>
                <Box sx={{ flexGrow: 1, bgcolor: 'background.default', minHeight: '100vh' }}>
                    <AppBar position="static" elevation={0}>
                        <Toolbar>
                            <IconButton
                                edge="start"
                                color="inherit"
                                onClick={() => navigate(-1)}
                                sx={{ mr: 2 }}
                            >
                                <ArrowBack />
                            </IconButton>
                            <Assessment sx={{ mr: 2 }} />
                            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                                AI Evaluation Dashboard
                            </Typography>
                        </Toolbar>
                    </AppBar>
                    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '60vh' }}>
                        <Box sx={{ textAlign: 'center' }}>
                            <LinearProgress sx={{ width: 300, mb: 2 }} />
                            <Typography variant="h6">Loading evaluation data...</Typography>
                        </Box>
                    </Box>
                </Box>
            </ThemeProvider>
        );
    }

    return (
        <ThemeProvider theme={theme}>
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
                            <Assessment sx={{ 
                                mr: 1, 
                                color: '#2563eb',
                                fontSize: '1.5rem',
                            }} />
                            <Typography variant="h6" sx={{ 
                                fontWeight: 700,
                                color: '#0f172a',
                                fontSize: '1.125rem',
                            }}>
                                Evaluation
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
                                    <Dashboard />
                                </ListItemIcon>
                                <ListItemText 
                                    primary="Overview" 
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
                                    primary="Evaluations" 
                                    primaryTypographyProps={{
                                        fontSize: '0.875rem',
                                        fontWeight: activeTab === 2 ? 600 : 500,
                                        color: activeTab === 2 ? '#2563eb' : '#475569',
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
                        overflow: 'auto',
                    }}
                >
                    {/* Header */}
                    <Box sx={{ mb: 4 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                            <Typography variant="h4" sx={{ fontWeight: 300 }}>
                                {activeTab === 0 && 'Dashboard Overview'}
                                {activeTab === 1 && 'Dataset Management'}
                                {activeTab === 2 && 'Evaluation Results'}
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 2 }}>
                                <FormControl sx={{ minWidth: 120 }}>
                                    <InputLabel>Time Range</InputLabel>
                                    <Select
                                        value={timeRange}
                                        label="Time Range"
                                        onChange={(e) => setTimeRange(e.target.value)}
                                        size="small"
                                    >
                                        <MenuItem value="7d">Last 7 days</MenuItem>
                                        <MenuItem value="30d">Last 30 days</MenuItem>
                                        <MenuItem value="90d">Last 90 days</MenuItem>
                                    </Select>
                                </FormControl>
                                <Button
                                    variant="outlined"
                                    startIcon={<Refresh />}
                                    onClick={handleRefresh}
                                    disabled={refreshing}
                                    size="small"
                                >
                                    Refresh
                                </Button>
                                <Button
                                    variant="outlined"
                                    startIcon={<Download />}
                                    onClick={() => {/* Add export functionality */}}
                                    size="small"
                                >
                                    Export
                                </Button>
                            </Box>
                        </Box>
                        
                        {error && (
                            <Fade in={!!error}>
                                <Alert severity="warning" sx={{ mb: 2 }}>
                                    {error}
                                </Alert>
                            </Fade>
                        )}
                    </Box>

                    {/* Tab Content */}
                    {activeTab === 0 && (
                        <Box>
                            {/* Overview Metrics Cards */}
                    <Grid container spacing={3} sx={{ mb: 4 }}>
                        <Grid item xs={12} sm={6} md={3}>
                            <Card sx={{ 
                                height: '100%',
                                background: 'linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%)',
                                border: '1px solid',
                                borderColor: 'success.light'
                            }}>
                                <CardContent>
                                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                        <CheckCircle sx={{ color: 'success.main', mr: 1, fontSize: 28 }} />
                                        <Typography variant="h6" fontWeight={600}>Groundedness</Typography>
                                    </Box>
                                    <Typography variant="h3" sx={{ color: 'success.main', mb: 1, fontWeight: 700 }}>
                                        {(evaluationData.overall.groundedness * 100).toFixed(1)}%
                                    </Typography>
                                    <Chip 
                                        label={getScoreLabel(evaluationData.overall.groundedness)} 
                                        size="small"
                                        sx={{ 
                                            bgcolor: 'success.main', 
                                            color: 'white',
                                            fontWeight: 600
                                        }}
                                    />
                                    <LinearProgress 
                                        variant="determinate" 
                                        value={evaluationData.overall.groundedness * 100} 
                                        sx={{ mt: 2, height: 6, borderRadius: 3 }}
                                    />
                                </CardContent>
                            </Card>
                        </Grid>

                        <Grid item xs={12} sm={6} md={3}>
                            <Card sx={{ 
                                height: '100%',
                                background: 'linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%)',
                                border: '1px solid',
                                borderColor: 'primary.light'
                            }}>
                                <CardContent>
                                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                        <TrendingUp sx={{ color: 'primary.main', mr: 1, fontSize: 28 }} />
                                        <Typography variant="h6" fontWeight={600}>Context Relevance</Typography>
                                    </Box>
                                    <Typography variant="h3" sx={{ color: 'primary.main', mb: 1, fontWeight: 700 }}>
                                        {(evaluationData.overall.contextRelevance * 100).toFixed(1)}%
                                    </Typography>
                                    <Chip 
                                        label={getScoreLabel(evaluationData.overall.contextRelevance)} 
                                        size="small"
                                        sx={{ 
                                            bgcolor: 'primary.main', 
                                            color: 'white',
                                            fontWeight: 600
                                        }}
                                    />
                                    <LinearProgress 
                                        variant="determinate" 
                                        value={evaluationData.overall.contextRelevance * 100} 
                                        sx={{ mt: 2, height: 6, borderRadius: 3 }}
                                        color="primary"
                                    />
                                </CardContent>
                            </Card>
                        </Grid>

                        <Grid item xs={12} sm={6} md={3}>
                            <Card sx={{ 
                                height: '100%',
                                background: 'linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%)',
                                border: '1px solid',
                                borderColor: 'secondary.light'
                            }}>
                                <CardContent>
                                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                        <BarChart sx={{ color: 'secondary.main', mr: 1, fontSize: 28 }} />
                                        <Typography variant="h6" fontWeight={600}>Answer Quality</Typography>
                                    </Box>
                                    <Typography variant="h3" sx={{ color: 'secondary.main', mb: 1, fontWeight: 700 }}>
                                        {(evaluationData.overall.answerQuality * 100).toFixed(1)}%
                                    </Typography>
                                    <Chip 
                                        label={getScoreLabel(evaluationData.overall.answerQuality)} 
                                        size="small"
                                        sx={{ 
                                            bgcolor: 'secondary.main', 
                                            color: 'white',
                                            fontWeight: 600
                                        }}
                                    />
                                    <LinearProgress 
                                        variant="determinate" 
                                        value={evaluationData.overall.answerQuality * 100} 
                                        sx={{ mt: 2, height: 6, borderRadius: 3 }}
                                        color="secondary"
                                    />
                                </CardContent>
                            </Card>
                        </Grid>

                        <Grid item xs={12} sm={6} md={3}>
                            <Card sx={{ 
                                height: '100%',
                                background: 'linear-gradient(135deg, #fff3e0 0%, #ffcc02 100%)',
                                border: '1px solid',
                                borderColor: 'warning.light'
                            }}>
                                <CardContent>
                                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                        <Timer sx={{ color: getMetricColor(evaluationData.overall.averageLatency, true), mr: 1, fontSize: 28 }} />
                                        <Typography variant="h6" fontWeight={600}>Avg Latency</Typography>
                                    </Box>
                                    <Typography variant="h3" sx={{ color: getMetricColor(evaluationData.overall.averageLatency, true), mb: 1, fontWeight: 700 }}>
                                        {Math.round(evaluationData.overall.averageLatency)}ms
                                    </Typography>
                                    <Chip 
                                        label={evaluationData.overall.averageLatency < 1000 ? 'Fast' : evaluationData.overall.averageLatency < 2000 ? 'Normal' : 'Slow'} 
                                        size="small"
                                        sx={{ 
                                            bgcolor: getMetricColor(evaluationData.overall.averageLatency, true), 
                                            color: 'white',
                                            fontWeight: 600
                                        }}
                                    />
                                    <Box sx={{ mt: 2, p: 1, bgcolor: 'rgba(255,255,255,0.3)', borderRadius: 1 }}>
                                        <Typography variant="caption" color="text.secondary">
                                            Target: &lt; 1000ms
                                        </Typography>
                                    </Box>
                                </CardContent>
                            </Card>
                        </Grid>
                    </Grid>

                    {/* Charts */}
                    <Grid container spacing={3} sx={{ mb: 4 }}>
                        <Grid item xs={12} md={6}>
                            <Card sx={{ height: 450 }}>
                                <CardContent sx={{ height: '100%' }}>
                                    <ReactECharts option={metricsOverviewOptions} style={{ height: '100%' }} />
                                </CardContent>
                            </Card>
                        </Grid>
                        
                        <Grid item xs={12} md={6}>
                            <Card sx={{ height: 450 }}>
                                <CardContent sx={{ height: '100%' }}>
                                    <ReactECharts option={latencyTrendsOptions} style={{ height: '100%' }} />
                                </CardContent>
                            </Card>
                        </Grid>
                    </Grid>

                    <Grid container spacing={3} sx={{ mb: 4 }}>
                        <Grid item xs={12}>
                            <Card>
                                <CardContent>
                                    <ReactECharts option={historicalTrendsOptions} style={{ height: '500px' }} />
                                </CardContent>
                            </Card>
                        </Grid>
                    </Grid>

                    {/* Detailed Results Table */}
                    <Card>
                        <CardContent>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                                <Typography variant="h5" fontWeight={600}>
                                    Recent Evaluation Results
                                </Typography>
                                <Button startIcon={<FilterList />} variant="outlined">
                                    Filter Results
                                </Button>
                            </Box>
                            <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
                                <Table stickyHeader>
                                    <TableHead>
                                        <TableRow>
                                            <TableCell sx={{ fontWeight: 600 }}>Timestamp</TableCell>
                                            <TableCell sx={{ fontWeight: 600 }}>Query</TableCell>
                                            <TableCell sx={{ fontWeight: 600 }}>Model</TableCell>
                                            <TableCell sx={{ fontWeight: 600 }}>Dataset</TableCell>
                                            <TableCell align="center" sx={{ fontWeight: 600 }}>Groundedness</TableCell>
                                            <TableCell align="center" sx={{ fontWeight: 600 }}>Context Relevance</TableCell>
                                            <TableCell align="center" sx={{ fontWeight: 600 }}>Answer Quality</TableCell>
                                            <TableCell align="center" sx={{ fontWeight: 600 }}>Latency (ms)</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {evaluationData.detailed.slice(0, 10).map((row, index) => (
                                            <TableRow key={row.id || `result-${index}`} hover>
                                                <TableCell>
                                                    <Typography variant="body2" color="text.secondary">
                                                        {new Date(row.timestamp).toLocaleString()}
                                                    </Typography>
                                                </TableCell>
                                                <TableCell sx={{ maxWidth: 300 }}>
                                                    <Typography variant="body2" noWrap>
                                                        {row.query}
                                                    </Typography>
                                                </TableCell>
                                                <TableCell>
                                                    <Chip 
                                                        label={row.model || 'Unknown'} 
                                                        size="small" 
                                                        variant="outlined"
                                                        color={row.model && row.model.includes('gpt-4') ? 'primary' : 'default'}
                                                    />
                                                </TableCell>
                                                <TableCell>
                                                    <Chip 
                                                        label={row.dataset || 'Unknown'} 
                                                        size="small" 
                                                        variant="outlined"
                                                        color="secondary"
                                                    />
                                                </TableCell>
                                                <TableCell align="center">
                                                    <Typography 
                                                        sx={{ 
                                                            color: getMetricColor(row.groundedness),
                                                            fontWeight: 600
                                                        }}
                                                    >
                                                        {(row.groundedness * 100).toFixed(1)}%
                                                    </Typography>
                                                </TableCell>
                                                <TableCell align="center">
                                                    <Typography 
                                                        sx={{ 
                                                            color: getMetricColor(row.contextRelevance),
                                                            fontWeight: 600
                                                        }}
                                                    >
                                                        {(row.contextRelevance * 100).toFixed(1)}%
                                                    </Typography>
                                                </TableCell>
                                                <TableCell align="center">
                                                    <Typography 
                                                        sx={{ 
                                                            color: getMetricColor(row.answerQuality),
                                                            fontWeight: 600
                                                        }}
                                                    >
                                                        {(row.answerQuality * 100).toFixed(1)}%
                                                    </Typography>
                                                </TableCell>
                                                <TableCell align="center">
                                                    <Typography 
                                                        sx={{ 
                                                            color: getMetricColor(row.latency, true),
                                                            fontWeight: 600
                                                        }}
                                                    >
                                                        {Math.round(row.latency)}
                                                    </Typography>
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                        {evaluationData.detailed.length === 0 && (
                                            <TableRow>
                                                <TableCell colSpan={8} align="center">
                                                    <Typography variant="body2" color="text.secondary" sx={{ py: 4 }}>
                                                        No evaluation results found. Run some test cases to see results here.
                                                    </Typography>
                                                </TableCell>
                                            </TableRow>
                                        )}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </CardContent>
                    </Card>
                        </Box>
                    )}

                    {/* Datasets Tab */}
                    {activeTab === 1 && (
                        <Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                                <Typography variant="h5" gutterBottom>
                                    Evaluation Datasets
                                </Typography>
                                <Button
                                    variant="contained"
                                    startIcon={<Add />}
                                    onClick={() => setCreateDatasetOpen(true)}
                                >
                                    Create Dataset
                                </Button>
                            </Box>
                            
                            {error && (
                                <Alert severity="error" sx={{ mb: 2 }}>
                                    {error}
                                </Alert>
                            )}
                            
                            <Box sx={{ display: 'flex', gap: 2, height: 'calc(100vh - 200px)', minHeight: '600px' }}>
                                {/* Dataset List (Master) */}
                                <Card sx={{ 
                                    flex: selectedDatasetForDetail ? '0 0 400px' : 1, 
                                    transition: 'flex 0.3s ease',
                                    display: 'flex',
                                    flexDirection: 'column'
                                }}>
                                    <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                                        <Typography variant="h6" gutterBottom>
                                            Datasets
                                        </Typography>
                                        <TableContainer sx={{ 
                                            flex: 1,
                                            overflow: 'auto',
                                            '&::-webkit-scrollbar': {
                                                width: '8px',
                                            },
                                            '&::-webkit-scrollbar-track': {
                                                background: '#f1f1f1',
                                                borderRadius: '4px',
                                            },
                                            '&::-webkit-scrollbar-thumb': {
                                                background: '#c1c1c1',
                                                borderRadius: '4px',
                                            },
                                            '&::-webkit-scrollbar-thumb:hover': {
                                                background: '#a8a8a8',
                                            },
                                        }}>
                                            <Table size="small">
                                                <TableHead>
                                                    <TableRow>
                                                        <TableCell>Name</TableCell>
                                                        <TableCell>Description</TableCell>
                                                        <TableCell>Docs</TableCell>
                                                        <TableCell>Status</TableCell>
                                                        <TableCell>Actions</TableCell>
                                                    </TableRow>
                                                </TableHead>
                                                <TableBody>
                                                    {datasets.length === 0 ? (
                                                        <TableRow>
                                                            <TableCell colSpan={5} align="center" sx={{ py: 4 }}>
                                                                <Typography variant="body2" color="text.secondary">
                                                                    No datasets available. Create your first dataset to get started.
                                                                </Typography>
                                                            </TableCell>
                                                        </TableRow>
                                                    ) : (
                                                        datasets.map((dataset) => (
                                                            <TableRow 
                                                                key={dataset?.id || 'unknown'}
                                                                hover
                                                                sx={{ 
                                                                    cursor: 'pointer',
                                                                    backgroundColor: selectedDatasetForDetail?.id === dataset?.id ? 'action.selected' : 'inherit'
                                                                }}
                                                            >
                                                                <TableCell 
                                                                    onClick={() => handleDatasetClick(dataset)}
                                                                    sx={{ 
                                                                        color: 'primary.main',
                                                                        fontWeight: selectedDatasetForDetail?.id === dataset?.id ? 600 : 400,
                                                                        '&:hover': { textDecoration: 'underline' }
                                                                    }}
                                                                >
                                                                    {dataset?.name || 'Unknown'}
                                                                </TableCell>
                                                                <TableCell onClick={() => handleDatasetClick(dataset)}>
                                                                    {dataset?.description || 'No description'}
                                                                </TableCell>
                                                                <TableCell onClick={() => handleDatasetClick(dataset)}>
                                                                    {dataset?.documentCount || 0}
                                                                </TableCell>
                                                                <TableCell onClick={() => handleDatasetClick(dataset)}>
                                                                    <Chip 
                                                                        label={dataset?.status || 'Unknown'}
                                                                        color={
                                                                            dataset?.status === "Ready" ? "success" : 
                                                                            dataset?.status === "Processing" ? "warning" :
                                                                            dataset?.status === "Error" ? "error" : "default"
                                                                        }
                                                                        size="small"
                                                                    />
                                                                </TableCell>
                                                                <TableCell>
                                                                    <IconButton 
                                                                        size="small" 
                                                                        color="error"
                                                                        onClick={(e) => {
                                                                            e.stopPropagation();
                                                                            handleDeleteDataset(dataset?.id, dataset?.name);
                                                                        }}
                                                                        title="Delete dataset"
                                                                    >
                                                                        <Delete />
                                                                    </IconButton>
                                                                </TableCell>
                                                            </TableRow>
                                                        ))
                                                    )}
                                                </TableBody>
                                            </Table>
                                        </TableContainer>
                                    </CardContent>
                                </Card>
                                
                                {/* Dataset Details (Detail) */}
                                {selectedDatasetForDetail && (
                                    <Card sx={{ 
                                        flex: 1,
                                        display: 'flex',
                                        flexDirection: 'column',
                                        minWidth: 0 // Prevent overflow
                                    }}>
                                        <CardContent sx={{ 
                                            flex: 1, 
                                            display: 'flex', 
                                            flexDirection: 'column',
                                            padding: 2,
                                            paddingTop: 1
                                        }}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                                <Typography variant="h6">
                                                    Dataset Details
                                                </Typography>
                                                <IconButton 
                                                    size="small" 
                                                    onClick={handleCloseDatasetDetail}
                                                    title="Close details"
                                                >
                                                    <Close />
                                                </IconButton>
                                            </Box>
                                            
                                            {loadingDatasetDetails ? (
                                                <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                                                    <CircularProgress />
                                                </Box>
                                            ) : datasetDetails ? (
                                                <Box sx={{ 
                                                    flex: 1, 
                                                    overflow: 'auto',
                                                    maxHeight: 'calc(100vh - 300px)',
                                                    paddingRight: 1,
                                                    '&::-webkit-scrollbar': {
                                                        width: '8px',
                                                    },
                                                    '&::-webkit-scrollbar-track': {
                                                        background: '#f1f1f1',
                                                        borderRadius: '4px',
                                                    },
                                                    '&::-webkit-scrollbar-thumb': {
                                                        background: '#c1c1c1',
                                                        borderRadius: '4px',
                                                    },
                                                    '&::-webkit-scrollbar-thumb:hover': {
                                                        background: '#a8a8a8',
                                                    },
                                                }}>
                                                    {/* Dataset Metadata */}
                                                    <Grid container spacing={2} sx={{ mb: 3 }}>
                                                        <Grid item xs={12}>
                                                            <Typography variant="h6" gutterBottom>
                                                                {datasetDetails.name}
                                                            </Typography>
                                                            <Typography variant="body2" color="text.secondary" paragraph>
                                                                {datasetDetails.description}
                                                            </Typography>
                                                        </Grid>
                                                        <Grid item xs={6}>
                                                            <Typography variant="body2" color="text.secondary">Created</Typography>
                                                            <Typography variant="body1">{datasetDetails.created_at}</Typography>
                                                        </Grid>
                                                        <Grid item xs={6}>
                                                            <Typography variant="body2" color="text.secondary">Status</Typography>
                                                            <Chip 
                                                                label={datasetDetails.status}
                                                                color={
                                                                    datasetDetails.status === "Ready" ? "success" : 
                                                                    datasetDetails.status === "Processing" ? "warning" :
                                                                    datasetDetails.status === "Error" ? "error" : "default"
                                                                }
                                                                size="small"
                                                            />
                                                        </Grid>
                                                        <Grid item xs={6}>
                                                            <Typography variant="body2" color="text.secondary">Documents</Typography>
                                                            <Typography variant="body1">{datasetDetails.document_count}</Typography>
                                                        </Grid>
                                                        <Grid item xs={6}>
                                                            <Typography variant="body2" color="text.secondary">Questions</Typography>
                                                            <Typography variant="body1">{datasetDetails.question_count}</Typography>
                                                        </Grid>
                                                    </Grid>
                                                    
                                                    {/* Generation Metadata */}
                                                    {datasetDetails.generation_metadata && Object.keys(datasetDetails.generation_metadata).length > 0 && (
                                                        <Box sx={{ mb: 3 }}>
                                                            <Typography variant="h6" gutterBottom>
                                                                Generation Metadata
                                                            </Typography>
                                                            <Grid container spacing={2}>
                                                                {Object.entries(datasetDetails.generation_metadata).map(([key, value]) => (
                                                                    <Grid item xs={6} key={key}>
                                                                        <Typography variant="body2" color="text.secondary">
                                                                            {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                                                        </Typography>
                                                                        <Typography variant="body1">{String(value)}</Typography>
                                                                    </Grid>
                                                                ))}
                                                            </Grid>
                                                        </Box>
                                                    )}
                                                    
                                                    {/* Document Accordion - Show questions grouped by document */}
                                                    {datasetDetails.documents_data && Object.keys(datasetDetails.documents_data).length > 0 ? (
                                                        <Box>
                                                            <Typography variant="h6" gutterBottom sx={{ mb: 3 }}>
                                                                Dataset Contents by Document
                                                            </Typography>
                                                            
                                                            {/* Document Accordions */}
                                                            <Box sx={{ width: '100%' }}>
                                                                {Object.entries(datasetDetails.documents_data).map(([docName, questions], index) => (
                                                                    <Accordion 
                                                                        key={index}
                                                                        defaultExpanded={true}
                                                                        sx={{ 
                                                                            mb: 2,
                                                                            borderRadius: '12px !important',
                                                                            boxShadow: '0 2px 12px rgba(0,0,0,0.08)',
                                                                            border: '1px solid #e0e7ff',
                                                                            overflow: 'hidden',
                                                                            '&:before': {
                                                                                display: 'none',
                                                                            },
                                                                            '&.Mui-expanded': {
                                                                                backgroundColor: '#f0f4ff',
                                                                                borderColor: '#c7d2fe',
                                                                                boxShadow: '0 4px 20px rgba(99, 102, 241, 0.15)',
                                                                            },
                                                                            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                                                                        }}
                                                                    >
                                                                        <AccordionSummary 
                                                                            expandIcon={<ExpandMore sx={{ 
                                                                                transition: 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                                                                                color: '#6366f1'
                                                                            }} />}
                                                                            sx={{ 
                                                                                backgroundColor: 'transparent',
                                                                                color: '#1e293b',
                                                                                minHeight: 64,
                                                                                borderRadius: '12px',
                                                                                px: 3,
                                                                                py: 2,
                                                                                '& .MuiAccordionSummary-content': {
                                                                                    margin: '0',
                                                                                    alignItems: 'center'
                                                                                },
                                                                                '& .MuiAccordionSummary-expandIconWrapper': {
                                                                                    color: '#6366f1'
                                                                                },
                                                                                '& .MuiAccordionSummary-expandIconWrapper.Mui-expanded': {
                                                                                    transform: 'rotate(180deg)',
                                                                                },
                                                                                '&:hover': {
                                                                                    backgroundColor: 'rgba(99, 102, 241, 0.05)',
                                                                                },
                                                                                transition: 'all 0.2s ease-in-out',
                                                                            }}
                                                                        >
                                                                            <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                                                                                <Box sx={{ 
                                                                                    width: 40,
                                                                                    height: 40,
                                                                                    borderRadius: '10px',
                                                                                    backgroundColor: '#6366f1',
                                                                                    display: 'flex',
                                                                                    alignItems: 'center',
                                                                                    justifyContent: 'center',
                                                                                    mr: 3,
                                                                                    flexShrink: 0
                                                                                }}>
                                                                                    <Description sx={{ color: 'white', fontSize: 20 }} />
                                                                                </Box>
                                                                                <Box sx={{ flex: 1 }}>
                                                                                    <Typography variant="h6" sx={{ 
                                                                                        fontWeight: 600,
                                                                                        color: '#1e293b',
                                                                                        fontSize: '1.1rem',
                                                                                        mb: 0.5
                                                                                    }}>
                                                                                        {docName}
                                                                                    </Typography>
                                                                                    <Typography variant="body2" sx={{ 
                                                                                        color: '#64748b',
                                                                                        fontSize: '0.875rem'
                                                                                    }}>
                                                                                        {questions.length} questions generated
                                                                                    </Typography>
                                                                                </Box>
                                                                            </Box>
                                                                        </AccordionSummary>
                                                                        <AccordionDetails sx={{ 
                                                                            p: 0,
                                                                            backgroundColor: 'transparent'
                                                                        }}>
                                                                            <Box sx={{ 
                                                                                mx: 3,
                                                                                mb: 3,
                                                                                mt: 1,
                                                                                p: 2, 
                                                                                backgroundColor: 'rgba(255, 255, 255, 0.7)',
                                                                                borderRadius: '8px',
                                                                                border: '1px solid rgba(99, 102, 241, 0.1)'
                                                                            }}>
                                                                                <Typography variant="body2" color="text.secondary" sx={{ 
                                                                                    mb: 2,
                                                                                    fontStyle: 'italic',
                                                                                    color: '#64748b'
                                                                                }}>
                                                                                    Questions and answers generated from this document
                                                                                </Typography>
                                                                            </Box>
                                                                            
                                                                            <Box sx={{ px: 3, pb: 2 }}>
                                                                                {questions.map((item, questionIndex) => (
                                                                                    <Box 
                                                                                        key={questionIndex} 
                                                                                        sx={{ 
                                                                                            mb: 3,
                                                                                            p: 3, 
                                                                                            backgroundColor: 'rgba(255, 255, 255, 0.8)',
                                                                                            borderRadius: '12px',
                                                                                            border: '1px solid rgba(99, 102, 241, 0.1)',
                                                                                            transition: 'all 0.2s ease-in-out',
                                                                                            '&:hover': {
                                                                                                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                                                                                                borderColor: 'rgba(99, 102, 241, 0.2)',
                                                                                                transform: 'translateY(-1px)',
                                                                                                boxShadow: '0 4px 12px rgba(99, 102, 241, 0.1)',
                                                                                            }
                                                                                        }}
                                                                                    >
                                                                                        <Typography variant="subtitle1" gutterBottom sx={{ 
                                                                                            color: '#6366f1', 
                                                                                            fontWeight: 700,
                                                                                            fontSize: '1rem',
                                                                                            mb: 2
                                                                                        }}>
                                                                                            Question {questionIndex + 1}
                                                                                        </Typography>
                                                                                        
                                                                                        <Box sx={{ mb: 3 }}>
                                                                                            <Typography variant="body2" sx={{ 
                                                                                                fontWeight: 600, 
                                                                                                mb: 1.5,
                                                                                                color: '#1e293b',
                                                                                                fontSize: '0.875rem'
                                                                                            }}>
                                                                                                Query:
                                                                                            </Typography>
                                                                                            <Box sx={{ 
                                                                                                p: 2.5, 
                                                                                                backgroundColor: '#f8fafc',
                                                                                                borderRadius: '8px',
                                                                                                border: '1px solid #e2e8f0',
                                                                                                position: 'relative',
                                                                                                '&:before': {
                                                                                                    content: '""',
                                                                                                    position: 'absolute',
                                                                                                    left: 0,
                                                                                                    top: 0,
                                                                                                    bottom: 0,
                                                                                                    width: '4px',
                                                                                                    backgroundColor: '#6366f1',
                                                                                                    borderRadius: '2px 0 0 2px'
                                                                                                }
                                                                                            }}>
                                                                                                <Typography variant="body2" sx={{ 
                                                                                                    color: '#334155',
                                                                                                    lineHeight: 1.6,
                                                                                                    fontSize: '0.875rem'
                                                                                                }}>
                                                                                                    {item.query || 'No query available'}
                                                                                                </Typography>
                                                                                            </Box>
                                                                                        </Box>
                                                                                        
                                                                                        {item.expected_response && (
                                                                                            <Box sx={{ mb: 3 }}>
                                                                                                <Typography variant="body2" sx={{ 
                                                                                                    fontWeight: 600, 
                                                                                                    mb: 1.5,
                                                                                                    color: '#1e293b',
                                                                                                    fontSize: '0.875rem'
                                                                                                }}>
                                                                                                    Expected Response:
                                                                                                </Typography>
                                                                                                <Box sx={{ 
                                                                                                    p: 2.5,
                                                                                                    backgroundColor: '#f0fdf4',
                                                                                                    borderRadius: '8px',
                                                                                                    border: '1px solid #dcfce7',
                                                                                                    position: 'relative',
                                                                                                    '&:before': {
                                                                                                        content: '""',
                                                                                                        position: 'absolute',
                                                                                                        left: 0,
                                                                                                        top: 0,
                                                                                                        bottom: 0,
                                                                                                        width: '4px',
                                                                                                        backgroundColor: '#22c55e',
                                                                                                        borderRadius: '2px 0 0 2px'
                                                                                                    }
                                                                                                }}>
                                                                                                    <Typography variant="body2" sx={{ 
                                                                                                        color: '#166534',
                                                                                                        lineHeight: 1.6,
                                                                                                        fontSize: '0.875rem'
                                                                                                    }}>
                                                                                                        {item.expected_response}
                                                                                                    </Typography>
                                                                                                </Box>
                                                                                            </Box>
                                                                                        )}
                                                                                        
                                                                                        {item.expected_chunks && item.expected_chunks.length > 0 && (
                                                                                            <Box sx={{ mb: 3 }}>
                                                                                                <Typography variant="body2" sx={{ 
                                                                                                    fontWeight: 600, 
                                                                                                    mb: 1.5,
                                                                                                    color: '#1e293b',
                                                                                                    fontSize: '0.875rem'
                                                                                                }}>
                                                                                                    Expected Sources:
                                                                                                </Typography>
                                                                                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                                                                                                    {item.expected_chunks.map((chunk, chunkIndex) => (
                                                                                                        <Box key={chunkIndex} sx={{ 
                                                                                                            p: 2,
                                                                                                            backgroundColor: '#fefce8',
                                                                                                            borderRadius: '8px',
                                                                                                            border: '1px solid #fde047',
                                                                                                            display: 'flex',
                                                                                                            alignItems: 'flex-start',
                                                                                                            gap: 1.5,
                                                                                                            transition: 'all 0.2s ease-in-out',
                                                                                                            '&:hover': {
                                                                                                                backgroundColor: '#fefbf0',
                                                                                                                borderColor: '#eab308',
                                                                                                            }
                                                                                                        }}>
                                                                                                            <Box sx={{
                                                                                                                width: 6,
                                                                                                                height: 6,
                                                                                                                borderRadius: '50%',
                                                                                                                backgroundColor: '#eab308',
                                                                                                                mt: 1,
                                                                                                                flexShrink: 0
                                                                                                            }} />
                                                                                                            <Box sx={{ flex: 1 }}>
                                                                                                                <Typography variant="body2" sx={{ 
                                                                                                                    fontWeight: 600, 
                                                                                                                    color: '#92400e',
                                                                                                                    mb: 0.5,
                                                                                                                    fontSize: '0.875rem'
                                                                                                                }}>
                                                                                                                    📄 {chunk.source || chunk.title || 'Unknown source'}
                                                                                                                </Typography>
                                                                                                                {chunk.text && (
                                                                                                                    <Typography variant="caption" sx={{ 
                                                                                                                        color: '#a16207',
                                                                                                                        display: 'block',
                                                                                                                        lineHeight: 1.4,
                                                                                                                        fontSize: '0.75rem'
                                                                                                                    }}>
                                                                                                                        {chunk.text.substring(0, 150)}...
                                                                                                                    </Typography>
                                                                                                                )}
                                                                                                            </Box>
                                                                                                        </Box>
                                                                                                    ))}
                                                                                                </Box>
                                                                                            </Box>
                                                                                        )}
                                                                                        
                                                                                        {item.metadata && Object.keys(item.metadata).length > 0 && (
                                                                                            <Box sx={{ 
                                                                                                mt: 3, 
                                                                                                pt: 3, 
                                                                                                borderTop: '1px solid #e2e8f0'
                                                                                            }}>
                                                                                                <Typography variant="caption" sx={{ 
                                                                                                    color: '#64748b',
                                                                                                    fontWeight: 500,
                                                                                                    fontSize: '0.75rem',
                                                                                                    fontFamily: 'monospace',
                                                                                                    backgroundColor: '#f1f5f9',
                                                                                                    p: 1,
                                                                                                    borderRadius: '4px',
                                                                                                    display: 'inline-block'
                                                                                                }}>
                                                                                                    Metadata: {Object.entries(item.metadata).map(([key, value]) => `${key}: ${value}`).join(', ')}
                                                                                                </Typography>
                                                                                            </Box>
                                                                                        )}
                                                                                    </Box>
                                                                                ))}
                                                                            </Box>
                                                                        </AccordionDetails>
                                                                    </Accordion>
                                                                ))}
                                                            </Box>
                                                        </Box>
                                                    ) : (
                                                        /* Fallback to sample items if documents_data is not available */
                                                        datasetDetails.sample_items && datasetDetails.sample_items.length > 0 && (
                                                            <Box>
                                                                <Typography variant="h6" gutterBottom>
                                                                    Sample Questions ({datasetDetails.sample_items.length} of {datasetDetails.total_items})
                                                                </Typography>
                                                                {datasetDetails.sample_items.map((item, index) => (
                                                                    <Card key={index} sx={{ mb: 2, backgroundColor: 'grey.50' }}>
                                                                        <CardContent>
                                                                            <Typography variant="subtitle2" gutterBottom>
                                                                                Question {index + 1}
                                                                            </Typography>
                                                                            <Typography variant="body2" paragraph>
                                                                                <strong>Query:</strong> {item.query}
                                                                            </Typography>
                                                                            {item.expected_response && (
                                                                                <Typography variant="body2" paragraph>
                                                                                    <strong>Expected Response:</strong> {item.expected_response}
                                                                                </Typography>
                                                                            )}
                                                                            {item.expected_chunks && item.expected_chunks.length > 0 && (
                                                                                <Box>
                                                                                    <Typography variant="body2" gutterBottom>
                                                                                        <strong>Expected Sources:</strong>
                                                                                    </Typography>
                                                                                    {item.expected_chunks.map((chunk, chunkIndex) => (
                                                                                        <Typography key={chunkIndex} variant="body2" color="text.secondary" sx={{ ml: 2, mb: 1 }}>
                                                                                            • {chunk.source || chunk.title || 'Unknown source'}
                                                                                            {chunk.text && `: ${chunk.text.substring(0, 100)}...`}
                                                                                        </Typography>
                                                                                    ))}
                                                                                </Box>
                                                                            )}
                                                                        </CardContent>
                                                                    </Card>
                                                                ))}
                                                            </Box>
                                                        )
                                                    )}
                                                </Box>
                                            ) : (
                                                <Typography variant="body2" color="text.secondary" align="center" sx={{ py: 4 }}>
                                                    Failed to load dataset details
                                                </Typography>
                                            )}
                                        </CardContent>
                                    </Card>
                                )}
                            </Box>
                        </Box>
                    )}

                    {/* Evaluations Tab */}
                    {activeTab === 2 && (
                        <Box>
                            <Card>
                                <CardContent>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                                        <Typography variant="h5" gutterBottom>
                                            Evaluations
                                        </Typography>
                                        <Button
                                            variant="contained"
                                            startIcon={<PlayArrow />}
                                            onClick={() => setCreateTestCaseOpen(true)}
                                        >
                                            Create Test
                                        </Button>
                                    </Box>
                                    
                                    {error && (
                                        <Alert severity="error" sx={{ mb: 2 }}>
                                            {error}
                                        </Alert>
                                    )}
                                    
                                    {/* Test Case Results Table */}
                                    <TableContainer>
                                        <Table>
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell>Test ID</TableCell>
                                                    <TableCell>Dataset</TableCell>
                                                    <TableCell>Model</TableCell>
                                                    <TableCell>Groundedness</TableCell>
                                                    <TableCell>Context Relevance</TableCell>
                                                    <TableCell>Answer Quality</TableCell>
                                                    <TableCell>Avg Latency (s)</TableCell>
                                                    <TableCell>Run Date</TableCell>
                                                    <TableCell>Status</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                {testCaseResults.length === 0 ? (
                                                    <TableRow>
                                                        <TableCell colSpan={9} align="center" sx={{ py: 4 }}>
                                                            <Typography variant="body2" color="text.secondary">
                                                                No test cases run yet. Create your first test case to get started.
                                                            </Typography>
                                                        </TableCell>
                                                    </TableRow>
                                                ) : (
                                                    testCaseResults.map((testCase) => (
                                                        <TableRow key={testCase.id}>
                                                            <TableCell>#{testCase.id}</TableCell>
                                                            <TableCell>{testCase.dataset_name}</TableCell>
                                                            <TableCell>
                                                                <Chip 
                                                                    label={testCase.models && testCase.models.length > 0 ? testCase.models[0] : 'N/A'} 
                                                                    size="small" 
                                                                    variant="outlined"
                                                                />
                                                            </TableCell>
                                                            <TableCell>
                                                                {testCase.results && testCase.results.length > 0 ? (
                                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                                        <Typography variant="body2" fontWeight="bold">
                                                                            {(testCase.results[0].groundedness * 100).toFixed(1)}%
                                                                        </Typography>
                                                                        <LinearProgress 
                                                                            variant="determinate" 
                                                                            value={testCase.results[0].groundedness * 100}
                                                                            sx={{ 
                                                                                width: 60, 
                                                                                height: 6,
                                                                                '& .MuiLinearProgress-bar': {
                                                                                    backgroundColor: testCase.results[0].groundedness >= 0.8 ? theme.palette.success.main :
                                                                                                   testCase.results[0].groundedness >= 0.6 ? theme.palette.warning.main :
                                                                                                   theme.palette.error.main
                                                                                }
                                                                            }}
                                                                        />
                                                                    </Box>
                                                                ) : '-'}
                                                            </TableCell>
                                                            <TableCell>
                                                                {testCase.results && testCase.results.length > 0 ? (
                                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                                        <Typography variant="body2" fontWeight="bold">
                                                                            {(testCase.results[0].context_relevance * 100).toFixed(1)}%
                                                                        </Typography>
                                                                        <LinearProgress 
                                                                            variant="determinate" 
                                                                            value={testCase.results[0].context_relevance * 100}
                                                                            sx={{ 
                                                                                width: 60, 
                                                                                height: 6,
                                                                                '& .MuiLinearProgress-bar': {
                                                                                    backgroundColor: testCase.results[0].context_relevance >= 0.8 ? theme.palette.success.main :
                                                                                                   testCase.results[0].context_relevance >= 0.6 ? theme.palette.warning.main :
                                                                                                   theme.palette.error.main
                                                                                }
                                                                            }}
                                                                        />
                                                                    </Box>
                                                                ) : '-'}
                                                            </TableCell>
                                                            <TableCell>
                                                                {testCase.results && testCase.results.length > 0 ? (
                                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                                        <Typography variant="body2" fontWeight="bold">
                                                                            {(testCase.results[0].answer_quality * 100).toFixed(1)}%
                                                                        </Typography>
                                                                        <LinearProgress 
                                                                            variant="determinate" 
                                                                            value={testCase.results[0].answer_quality * 100}
                                                                            sx={{ 
                                                                                width: 60, 
                                                                                height: 6,
                                                                                '& .MuiLinearProgress-bar': {
                                                                                    backgroundColor: testCase.results[0].answer_quality >= 0.8 ? theme.palette.success.main :
                                                                                                   testCase.results[0].answer_quality >= 0.6 ? theme.palette.warning.main :
                                                                                                   theme.palette.error.main
                                                                                }
                                                                            }}
                                                                        />
                                                                    </Box>
                                                                ) : '-'}
                                                            </TableCell>
                                                            <TableCell>
                                                                {testCase.results && testCase.results.length > 0 ? (
                                                                    <Typography 
                                                                        variant="body2" 
                                                                        fontWeight="bold"
                                                                        color={
                                                                            testCase.results[0].avg_latency < 1 ? theme.palette.success.main :
                                                                            testCase.results[0].avg_latency < 2 ? theme.palette.warning.main :
                                                                            theme.palette.error.main
                                                                        }
                                                                    >
                                                                        {testCase.results[0].avg_latency.toFixed(2)}s
                                                                    </Typography>
                                                                ) : '-'}
                                                            </TableCell>
                                                            <TableCell>
                                                                {new Date(testCase.created_at).toLocaleDateString()}
                                                            </TableCell>
                                                            <TableCell>
                                                                <Chip 
                                                                    label={testCase.status}
                                                                    color={
                                                                        testCase.status === "completed" ? "success" : 
                                                                        testCase.status === "running" ? "warning" :
                                                                        testCase.status === "failed" ? "error" : "default"
                                                                    }
                                                                    size="small"
                                                                    icon={testCase.status === "running" ? <CircularProgress size={12} /> : undefined}
                                                                />
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
                    )}

                    {/* Create Dataset Dialog */}
                    <Dialog 
                        open={createDatasetOpen} 
                        onClose={() => {
                            setCreateDatasetOpen(false);
                            // Reset form when closing
                            setNewDatasetName('');
                            setNewDatasetDescription('');
                            setSelectedDocuments([]);
                            setSelectedModel('');
                            setNumQuestionsPerDoc(3);
                            setDifficultyLevels(['easy', 'medium', 'hard']);
                            setError('');
                        }}
                        maxWidth="md"
                        fullWidth
                    >
                        <DialogTitle>Create New Dataset</DialogTitle>
                        <DialogContent>
                            <Box sx={{ pt: 1 }}>
                                {error && (
                                    <Alert severity="error" sx={{ mb: 2 }}>
                                        {error}
                                    </Alert>
                                )}
                                
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
                                    sx={{ mb: 3 }}
                                />

                                <Grid container spacing={2} sx={{ mb: 3 }}>
                                    <Grid item xs={12} md={6}>
                                        <FormControl fullWidth>
                                            <InputLabel>Model for Generation</InputLabel>
                                            <Select
                                                value={selectedModel}
                                                onChange={(e) => setSelectedModel(e.target.value)}
                                                label="Model for Generation"
                                                required
                                            >
                                                {availableModels.map((model) => (
                                                    <MenuItem key={model?.name || 'unknown'} value={model?.name || ''}>
                                                        {model?.display_name || model?.name || 'Unknown Model'}
                                                    </MenuItem>
                                                ))}
                                            </Select>
                                        </FormControl>
                                    </Grid>
                                    <Grid item xs={12} md={6}>
                                        <FormControl fullWidth>
                                            <InputLabel>Questions per Document</InputLabel>
                                            <Select
                                                value={numQuestionsPerDoc}
                                                onChange={(e) => setNumQuestionsPerDoc(e.target.value)}
                                                label="Questions per Document"
                                            >
                                                <MenuItem value={1}>1 Question</MenuItem>
                                                <MenuItem value={2}>2 Questions</MenuItem>
                                                <MenuItem value={3}>3 Questions (Recommended)</MenuItem>
                                                <MenuItem value={4}>4 Questions</MenuItem>
                                                <MenuItem value={5}>5 Questions</MenuItem>
                                            </Select>
                                        </FormControl>
                                    </Grid>
                                </Grid>

                                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                                    Difficulty Levels
                                </Typography>
                                <Box sx={{ mb: 3 }}>
                                    {['easy', 'medium', 'hard'].map((level) => (
                                        <FormControlLabel
                                            key={level}
                                            control={
                                                <Checkbox
                                                    checked={difficultyLevels.includes(level)}
                                                    onChange={(e) => {
                                                        if (e.target.checked) {
                                                            setDifficultyLevels([...difficultyLevels, level]);
                                                        } else {
                                                            setDifficultyLevels(difficultyLevels.filter(l => l !== level));
                                                        }
                                                    }}
                                                />
                                            }
                                            label={level.charAt(0).toUpperCase() + level.slice(1)}
                                        />
                                    ))}
                                </Box>
                                
                                <Typography variant="subtitle1" gutterBottom>
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
                                
                                <List sx={{ maxHeight: 300, overflow: 'auto', border: '1px solid #ddd', borderRadius: 1 }}>
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
                                                        checked={selectedDocuments.includes(doc?.id || 'unknown')}
                                                        onChange={(e) => {
                                                            const docId = doc?.id || 'unknown';
                                                            if (e.target.checked) {
                                                                setSelectedDocuments([...selectedDocuments, docId]);
                                                            } else {
                                                                setSelectedDocuments(selectedDocuments.filter(id => id !== docId));
                                                            }
                                                        }}
                                                    />
                                                </ListItemSecondaryAction>
                                            </ListItem>
                                        ))
                                    )}
                                </List>
                            </Box>
                        </DialogContent>
                        <DialogActions>
                            <Button onClick={() => {
                                setCreateDatasetOpen(false);
                                // Reset form when canceling
                                setNewDatasetName('');
                                setNewDatasetDescription('');
                                setSelectedDocuments([]);
                                setSelectedModel('');
                                setNumQuestionsPerDoc(3);
                                setDifficultyLevels(['easy', 'medium', 'hard']);
                                setError('');
                            }}>Cancel</Button>
                            <Button 
                                onClick={handleCreateDataset} 
                                variant="contained"
                                disabled={!newDatasetName.trim() || selectedDocuments.length === 0 || !selectedModel}
                            >
                                Create Dataset
                            </Button>
                        </DialogActions>
                    </Dialog>

                    {/* Dataset Generation Progress Dialog */}
                    <Dialog 
                        open={showProgressDialog} 
                        maxWidth="sm" 
                        fullWidth
                        disableEscapeKeyDown
                    >
                        <DialogTitle>
                            {datasetGenerationProgress.status === 'starting' ? 'Starting Dataset Generation...' : 'Generating Dataset'}
                        </DialogTitle>
                        <DialogContent>
                            <Box sx={{ mb: 2 }}>
                                <Typography variant="body1" sx={{ mb: 1 }}>
                                    Status: {datasetGenerationProgress.status === 'starting' ? 'Initializing...' : datasetGenerationProgress.status}
                                </Typography>
                                
                                {datasetGenerationProgress.current_document && (
                                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                        Processing: {datasetGenerationProgress.current_document}
                                    </Typography>
                                )}
                                
                                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                    Progress: {datasetGenerationProgress.completed_documents} / {datasetGenerationProgress.total_documents} documents
                                </Typography>
                                
                                <LinearProgress 
                                    variant={datasetGenerationProgress.status === 'starting' ? 'indeterminate' : 'determinate'}
                                    value={datasetGenerationProgress.status === 'starting' ? 0 : datasetGenerationProgress.progress} 
                                    sx={{ mb: 1 }}
                                />
                                
                                <Typography variant="body2" color="text.secondary" align="center">
                                    {datasetGenerationProgress.status === 'starting' ? 'Please wait...' : `${datasetGenerationProgress.progress}%`}
                                </Typography>
                                
                                {datasetGenerationProgress.status === 'completed' && (
                                    <Alert severity="success" sx={{ mt: 2 }}>
                                        Dataset generated successfully! 
                                        {datasetGenerationProgress.question_count && 
                                            ` Generated ${datasetGenerationProgress.question_count} questions.`
                                        }
                                    </Alert>
                                )}
                                
                                {datasetGenerationProgress.status === 'error' && (
                                    <Alert severity="error" sx={{ mt: 2 }}>
                                        {datasetGenerationProgress.error || 'An error occurred during generation'}
                                    </Alert>
                                )}
                            </Box>
                        </DialogContent>
                        {datasetGenerationProgress.status === 'completed' || datasetGenerationProgress.status === 'error' ? (
                            <DialogActions>
                                <Button onClick={() => {
                                    setShowProgressDialog(false);
                                    setCurrentDatasetId(null);
                                }}>
                                    Close
                                </Button>
                            </DialogActions>
                        ) : null}
                    </Dialog>

                    {/* Test Case Creation Dialog */}
                    <Dialog 
                        open={createTestCaseOpen} 
                        onClose={() => setCreateTestCaseOpen(false)}
                        maxWidth="md"
                        fullWidth
                    >
                        <DialogTitle>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <PlayArrow color="primary" />
                                Create Test Case
                            </Box>
                        </DialogTitle>
                        <DialogContent>
                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, mt: 2 }}>
                                {/* Dataset Selector */}
                                <FormControl fullWidth>
                                    <InputLabel>Select Dataset</InputLabel>
                                    <Select
                                        value={selectedDatasetId}
                                        onChange={(e) => setSelectedDatasetId(e.target.value)}
                                        input={<OutlinedInput label="Select Dataset" />}
                                    >
                                        {datasets.filter(d => d?.status === "Ready").map((dataset) => (
                                            <MenuItem key={dataset?.id} value={dataset?.id}>
                                                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                                                    <Typography variant="body1">
                                                        {dataset?.name || 'Unknown Dataset'}
                                                    </Typography>
                                                    <Typography variant="caption" color="text.secondary">
                                                        {dataset?.description || 'No description'} • {dataset?.documentCount || 0} documents
                                                    </Typography>
                                                </Box>
                                            </MenuItem>
                                        ))}
                                    </Select>
                                </FormControl>

                                {/* Model Selector */}
                                <FormControl fullWidth>
                                    <InputLabel>Select Model</InputLabel>
                                    <Select
                                        value={selectedModelId}
                                        onChange={(e) => setSelectedModelId(e.target.value)}
                                        input={<OutlinedInput label="Select Model" />}
                                    >
                                        {availableModels.map((model) => (
                                            <MenuItem key={model.name} value={model.name}>
                                                <ListItemText 
                                                    primary={model.display_name || model.name}
                                                    secondary={`Size: ${model.size || 'Unknown'}`}
                                                />
                                            </MenuItem>
                                        ))}
                                    </Select>
                                </FormControl>

                                <Divider />

                                {/* Retrieval Configuration Toggle */}
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <Typography variant="h6">
                                        Retrieval Configuration
                                    </Typography>
                                    <Button
                                        variant="outlined"
                                        size="small"
                                        onClick={() => setShowAdvancedRetrieval(!showAdvancedRetrieval)}
                                    >
                                        {showAdvancedRetrieval ? 'Hide Advanced' : 'Show Advanced'}
                                    </Button>
                                </Box>

                                {/* Advanced Retrieval Settings */}
                                {showAdvancedRetrieval && (
                                    <Card variant="outlined" sx={{ p: 2 }}>
                                        <Grid container spacing={2}>
                                            {/* Similarity Threshold */}
                                            <Grid item xs={12} sm={6}>
                                                <Typography variant="subtitle2" gutterBottom>
                                                    Similarity Threshold: {testRetrievalConfig.similarity_threshold}
                                                </Typography>
                                                <Box sx={{ px: 1 }}>
                                                    <input
                                                        type="range"
                                                        min="0.0"
                                                        max="1.0"
                                                        step="0.1"
                                                        value={testRetrievalConfig.similarity_threshold}
                                                        onChange={(e) => handleTestRetrievalConfigChange('similarity_threshold', parseFloat(e.target.value))}
                                                        style={{ width: '100%' }}
                                                    />
                                                </Box>
                                                <Typography variant="caption" color="text.secondary">
                                                    Minimum similarity score for retrieving chunks
                                                </Typography>
                                            </Grid>

                                            {/* Keyword Similarity Weight */}
                                            <Grid item xs={12} sm={6}>
                                                <Typography variant="subtitle2" gutterBottom>
                                                    Keyword Weight: {testRetrievalConfig.keyword_similarity_weight}
                                                </Typography>
                                                <Box sx={{ px: 1 }}>
                                                    <input
                                                        type="range"
                                                        min="0.0"
                                                        max="1.0"
                                                        step="0.1"
                                                        value={testRetrievalConfig.keyword_similarity_weight}
                                                        onChange={(e) => handleTestRetrievalConfigChange('keyword_similarity_weight', parseFloat(e.target.value))}
                                                        style={{ width: '100%' }}
                                                    />
                                                </Box>
                                                <Typography variant="caption" color="text.secondary">
                                                    Only used with Hybrid search. 1.0 = keyword, 0.0 = semantic
                                                </Typography>
                                            </Grid>

                                            {/* Max Chunks */}
                                            <Grid item xs={12} sm={6}>
                                                <FormControl fullWidth size="small">
                                                    <InputLabel>Max Chunks</InputLabel>
                                                    <Select
                                                        value={testRetrievalConfig.max_chunks}
                                                        onChange={(e) => handleTestRetrievalConfigChange('max_chunks', parseInt(e.target.value))}
                                                        input={<OutlinedInput label="Max Chunks" />}
                                                    >
                                                        {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20].map((num) => (
                                                            <MenuItem key={num} value={num}>{num}</MenuItem>
                                                        ))}
                                                    </Select>
                                                </FormControl>
                                                <Typography variant="caption" color="text.secondary">
                                                    Maximum chunks to retrieve
                                                </Typography>
                                            </Grid>

                                            {/* Search Type */}
                                            <Grid item xs={12} sm={6}>
                                                <FormControl fullWidth size="small">
                                                    <InputLabel>Search Type</InputLabel>
                                                    <Select
                                                        value={testRetrievalConfig.search_type}
                                                        onChange={(e) => handleTestRetrievalConfigChange('search_type', e.target.value)}
                                                        input={<OutlinedInput label="Search Type" />}
                                                    >
                                                        <MenuItem value="similarity">Similarity Search</MenuItem>
                                                        <MenuItem value="mmr">Maximum Marginal Relevance (MMR)</MenuItem>
                                                        <MenuItem value="similarity_score_threshold">Similarity with Score Threshold</MenuItem>
                                                        <MenuItem value="hybrid">Hybrid (Semantic + Keyword)</MenuItem>
                                                    </Select>
                                                </FormControl>
                                                <Typography variant="caption" color="text.secondary">
                                                    Search algorithm for document retrieval. Hybrid combines semantic similarity with keyword matching.
                                                </Typography>
                                            </Grid>

                                            {/* Reranker Settings */}
                                            <Grid item xs={12}>
                                                <Box sx={{ mt: 2, p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                                                    <Typography variant="subtitle1" gutterBottom>
                                                        Reranker Configuration
                                                    </Typography>
                                                    
                                                    <FormControlLabel
                                                        control={
                                                            <Switch
                                                                checked={testRetrievalConfig.reranker_enabled}
                                                                onChange={(e) => handleTestRetrievalConfigChange('reranker_enabled', e.target.checked)}
                                                            />
                                                        }
                                                        label="Enable Reranker Model"
                                                    />

                                                    {testRetrievalConfig.reranker_enabled && (
                                                        <FormControl fullWidth size="small" sx={{ mt: 2 }}>
                                                            <InputLabel>Reranker Model</InputLabel>
                                                            <Select
                                                                value={testRetrievalConfig.reranker_model}
                                                                onChange={(e) => handleTestRetrievalConfigChange('reranker_model', e.target.value)}
                                                                input={<OutlinedInput label="Reranker Model" />}
                                                            >
                                                                {Array.isArray(rerankerModels) && rerankerModels.map((model) => (
                                                                    <MenuItem key={model?.name || 'unknown'} value={model?.name || ''}>
                                                                        {model?.display_name || model?.name || 'Unknown Model'}
                                                                    </MenuItem>
                                                                ))}
                                                            </Select>
                                                        </FormControl>
                                                    )}
                                                </Box>
                                            </Grid>

                                            {/* Auto Merging Settings */}
                                            <Grid item xs={12}>
                                                <Box sx={{ mt: 2, p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                                                    <Typography variant="subtitle1" gutterBottom>
                                                        Auto Merging Retrieval
                                                    </Typography>
                                                    
                                                    <FormControlLabel
                                                        control={
                                                            <Switch
                                                                checked={testRetrievalConfig.auto_merging_enabled}
                                                                onChange={(e) => handleTestRetrievalConfigChange('auto_merging_enabled', e.target.checked)}
                                                            />
                                                        }
                                                        label="Enable Auto Merging"
                                                    />

                                                    {testRetrievalConfig.auto_merging_enabled && (
                                                        <Box sx={{ mt: 2 }}>
                                                            <Typography variant="subtitle2" gutterBottom>
                                                                Merging Similarity Threshold: {testRetrievalConfig.auto_merging_similarity_threshold}
                                                            </Typography>
                                                            <Box sx={{ px: 1 }}>
                                                                <input
                                                                    type="range"
                                                                    min="0.5"
                                                                    max="1.0"
                                                                    step="0.05"
                                                                    value={testRetrievalConfig.auto_merging_similarity_threshold}
                                                                    onChange={(e) => handleTestRetrievalConfigChange('auto_merging_similarity_threshold', parseFloat(e.target.value))}
                                                                    style={{ width: '100%' }}
                                                                />
                                                            </Box>
                                                            <Typography variant="caption" color="text.secondary">
                                                                Higher values merge only very similar chunks
                                                            </Typography>
                                                        </Box>
                                                    )}
                                                </Box>
                                            </Grid>
                                        </Grid>
                                    </Card>
                                )}

                                {/* Configuration Summary */}
                                {selectedDatasetId && selectedModelId && (
                                    <Card variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
                                        <Typography variant="h6" gutterBottom color="primary">
                                            Test Configuration Summary
                                        </Typography>
                                        <Grid container spacing={2}>
                                            <Grid item xs={12} sm={6}>
                                                <Typography variant="subtitle2" color="text.secondary">
                                                    Dataset
                                                </Typography>
                                                <Typography variant="body2">
                                                    {datasets.find(d => d.id === selectedDatasetId)?.name || 'Unknown'}
                                                </Typography>
                                            </Grid>
                                            <Grid item xs={12} sm={6}>
                                                <Typography variant="subtitle2" color="text.secondary">
                                                    Model
                                                </Typography>
                                                <Typography variant="body2">
                                                    {(() => {
                                                        const model = availableModels.find(m => m.name === selectedModelId);
                                                        return model?.display_name || selectedModelId;
                                                    })()}
                                                </Typography>
                                            </Grid>
                                            <Grid item xs={12}>
                                                <Typography variant="subtitle2" color="text.secondary">
                                                    Retrieval Settings
                                                </Typography>
                                                <Typography variant="body2">
                                                    Similarity: {testRetrievalConfig.similarity_threshold}, 
                                                    Max Chunks: {testRetrievalConfig.max_chunks}, 
                                                    Search: {testRetrievalConfig.search_type}
                                                    {testRetrievalConfig.reranker_enabled && ', Reranker: Enabled'}
                                                    {testRetrievalConfig.auto_merging_enabled && ', Auto-merge: Enabled'}
                                                </Typography>
                                            </Grid>
                                        </Grid>
                                    </Card>
                                )}

                                {/* Error Display */}
                                {error && (
                                    <Alert severity="error">
                                        {error}
                                    </Alert>
                                )}
                            </Box>
                        </DialogContent>
                        <DialogActions>
                            <Button onClick={() => setCreateTestCaseOpen(false)}>
                                Cancel
                            </Button>
                            <Button
                                variant="contained"
                                onClick={handleCreateTestCase}
                                disabled={!selectedDatasetId || !selectedModelId || runningTestCase}
                                startIcon={runningTestCase ? <CircularProgress size={20} /> : <PlayArrow />}
                            >
                                {runningTestCase ? 'Running Test...' : 'Run Evaluation'}
                            </Button>
                        </DialogActions>
                    </Dialog>
                </Box>
            </Box>
        </ThemeProvider>
    );
};

export default EvaluationPage;