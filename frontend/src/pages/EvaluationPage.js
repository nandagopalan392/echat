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
import webSocketService, { CONNECTION_STATUS } from '../services/websocketService';
import useWebSocketConnection from '../hooks/useWebSocketConnection';

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
    
    // Test case details dialog state
    const [testCaseDetailsOpen, setTestCaseDetailsOpen] = useState(false);
    const [selectedTestCase, setSelectedTestCase] = useState(null);
    
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
    
    // Background evaluation state
    const [backgroundEvaluations, setBackgroundEvaluations] = useState(new Map());
    const [websocketConnections, setWebsocketConnections] = useState(new Map());
    const [evaluationProgress, setEvaluationProgress] = useState(new Map());
    
    // New WebSocket service state
    const [activeConnections, setActiveConnections] = useState(new Set());
    const [connectionStatuses, setConnectionStatuses] = useState(new Map());
    
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
        // Set loading state once at the beginning
        console.log('🚀 [DEBUG] EvaluationPage useEffect triggered with timeRange:', timeRange);
        setLoading(true);
        setError('');
        
        // Load only evaluation data on page load
        const loadEssentialData = async () => {
            try {
                console.log('📊 [DEBUG] Loading essential data on page load');
                // Only load evaluation-specific data that's visible by default
                await Promise.all([
                    loadEvaluationDataWithoutLoading(),
                    loadDatasets(),
                    loadEvaluationResults(),
                    loadRecentBackgroundEvaluations() // Load existing completed evaluations for the table
                ]);
                // Set loading to false after essential data is loaded
                setLoading(false);
                console.log('✅ [DEBUG] Essential data loaded successfully');
            } catch (error) {
                console.error('❌ [DEBUG] Error loading essential data:', error);
                setError('Failed to load evaluation data. Please try refreshing the page.');
                setLoading(false);
            }
        };

        // Load essential data only
        loadEssentialData();
        
        // Note: loadTestCaseResults() will be populated by loadRecentBackgroundEvaluations()
        // and then updated in real-time via WebSocket during active evaluations
        // Note: loadAvailableModels(), loadRerankerModels(), loadAvailableRetrievers(),
        // and loadAvailableDocuments() are loaded only when needed (on Create Test dialog open)
    }, [timeRange]);

    // Load models and reranker data only when Create Test dialog is opened
    useEffect(() => {
        if (createTestCaseOpen) {
            // Load data needed for creating tests
            Promise.all([
                loadAvailableDocuments(),
                loadAvailableModels(),
                loadAvailableRetrievers(),
                loadRerankerModels()
            ]).catch(error => {
                console.error('Error loading test creation data:', error);
                // Don't show error for this - user can still try to create test
            });
        }
    }, [createTestCaseOpen]);

    // Cleanup WebSocket connections on unmount
    useEffect(() => {
        return () => {
            // Cleanup old WebSocket connections
            websocketConnections.forEach((ws) => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.close();
                }
            });
            
            // Cleanup new WebSocket service connections
            webSocketService.disconnectAll();
        };
    }, [websocketConnections]);

    // Recalculate overview whenever testCaseResults changes
    useEffect(() => {
        if (testCaseResults && testCaseResults.length >= 0) {
            calculateOverviewFromRealData();
        }
    }, [testCaseResults]);

    // Refresh data when switching to Overview tab
    useEffect(() => {
        if (activeTab === 0) {
            console.log('📊 Switching to Overview tab, refreshing evaluation data...');
            // Reload recent evaluations to ensure we have the latest data
            loadRecentBackgroundEvaluations().then(() => {
                // After loading fresh data, recalculate overview
                calculateOverviewFromRealData();
            }).catch(error => {
                console.error('Error refreshing data for Overview tab:', error);
                // Still try to calculate with existing data
                calculateOverviewFromRealData();
            });
        }
    }, [activeTab]);

    // Load datasets
    const loadDatasets = async () => {
        console.log('📊 [DEBUG] loadDatasets() - START - Function called');
        try {
            console.log('📊 [DEBUG] loadDatasets() - Fetching datasets from API...');
            const response = await evaluationApi.getDatasets();
            console.log('📊 [DEBUG] loadDatasets() - API response:', response);
            console.log('📊 [DEBUG] loadDatasets() - Raw datasets count:', (response.datasets || []).length);
            
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
            
            console.log('📊 [DEBUG] loadDatasets() - Transformed datasets count:', transformedDatasets.length);
            console.log('📊 [DEBUG] loadDatasets() - Datasets with status:', transformedDatasets.map(d => ({id: d.id, name: d.name, status: d.status})));
            const completedCount = transformedDatasets.filter(d => d.status === 'completed').length;
            console.log(`✅ [DEBUG] loadDatasets() - Completed datasets: ${completedCount}/${transformedDatasets.length}`);
            
            setDatasets(transformedDatasets);
            console.log('✅ [DEBUG] loadDatasets() - State updated successfully');
        } catch (error) {
            console.error('❌ [DEBUG] Error loading datasets:', error);
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
            console.log('Loading available documents...');
            const response = await api.call('/api/documents');
            console.log('Documents API response:', response);
            setAvailableDocuments(response.documents || []);
            console.log('Set available documents:', response.documents || []);
            
            // Debug: Check document structure
            if (response.documents && response.documents.length > 0) {
                console.log('Sample document structure:', response.documents[0]);
                console.log('Document ID field:', response.documents[0]?.id);
                console.log('Document keys:', Object.keys(response.documents[0] || {}));
            }
        } catch (error) {
            console.error('Error loading documents:', error);
            // Fallback to some sample data for testing
            const sampleDocuments = [
                { id: 'doc1', title: 'Sample Document 1', filename: 'sample1.pdf' },
                { id: 'doc2', title: 'Sample Document 2', filename: 'sample2.pdf' },
                { id: 'doc3', title: 'Sample Document 3', filename: 'sample3.pdf' }
            ];
            setAvailableDocuments(sampleDocuments);
            console.log('Set fallback sample documents:', sampleDocuments);
        }
    };

    // Load available models
    const loadAvailableModels = async () => {
        try {
            console.log('Loading available models...');
            const response = await api.call('/api/models/available');
            console.log('Models API response:', response);
            setAvailableModels(response.llm_models || []);
            console.log('Set available models:', response.llm_models || []);
        } catch (error) {
            console.error('Error loading models:', error);
            // Fallback to some sample data for testing
            const sampleModels = [
                { name: 'llama3', category: 'llm', description: 'LLaMA 3 Model', source: 'local' },
                { name: 'deepseek-r1', category: 'llm', description: 'DeepSeek R1 Model', source: 'local' },
                { name: 'gemma3n:e2b', category: 'llm', description: 'Gemma 3N Model', source: 'local' }
            ];
            setAvailableModels(sampleModels);
            console.log('Set fallback sample models:', sampleModels);
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
        console.log('🔄 [DEBUG] loadEvaluationResults() called');
        try {
            console.log('🌐 [DEBUG] Making API call to evaluationApi.getResults()');
            const response = await evaluationApi.getResults();
            console.log('✅ [DEBUG] evaluationApi.getResults() response:', response);
            setEvaluationResults(response.results || []);
            setFilteredResults(response.results || []);
            console.log('✅ [DEBUG] Evaluation results state updated with', (response.results || []).length, 'items');
        } catch (error) {
            console.error('❌ [DEBUG] Error loading evaluation results:', error);
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
            console.log('⚠️ [DEBUG] API call failed, using mock evaluation results with', mockResults.length, 'items');
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
        console.log('🔄 [DEBUG] Tab changed from', activeTab, 'to', newValue);
        setActiveTab(newValue);
        
        // Load tab-specific data when switching to Evaluation Results tab
        if (newValue === 2) {
            console.log('📊 [DEBUG] Switching to Evaluation Results tab - loading data');
            // Load evaluation results and recent background evaluations
            Promise.all([
                loadEvaluationResults(),
                loadRecentBackgroundEvaluations()
            ]).catch(error => {
                console.error('❌ [DEBUG] Error loading evaluation tab data:', error);
            });
        }
    };

    // Test case click handler
    const handleTestCaseClick = (testCase) => {
        setSelectedTestCase(testCase);
        setTestCaseDetailsOpen(true);
    };

    // Create dataset dialog open handler - loads required data
    const handleCreateDatasetOpen = async () => {
        console.log('Opening Create Dataset dialog...');
        setCreateDatasetOpen(true);
        
        // Load models and documents when dialog opens
        try {
            console.log('Loading models and documents...');
            await Promise.all([
                loadAvailableModels(),
                loadAvailableDocuments()
            ]);
            console.log('Successfully loaded models and documents');
        } catch (error) {
            console.error('Error loading data for Create Dataset dialog:', error);
            setError('Failed to load models or documents');
        }
    };

    // Create dataset handler (updated for async background tasks - no progress dialog)
    const handleCreateDataset = async () => {
        try {
            if (!newDatasetName.trim()) {
                setError('Dataset name is required');
                return;
            }

            if (selectedDocuments.length === 0) {
                setError('Please select at least one document from the list below');
                return;
            }

            // Close the dialog immediately after validation
            setCreateDatasetOpen(false);
            
            // Add a small delay to ensure state updates are processed
            await new Promise(resolve => setTimeout(resolve, 100));

            if (!selectedModel) {
                setError('Please select a model for dataset generation');
                return;
            }

            // Ensure document_ids are strings (handle strings, numbers, or objects)
            const documentIds = selectedDocuments.map(doc => {
                if (typeof doc === 'string') return doc;
                if (typeof doc === 'number') return doc.toString();
                if (doc && typeof doc === 'object' && doc.id) return doc.id.toString();
                return null;
            }).filter(id => id); // Remove any null/undefined values

            console.log('Processed document IDs:', documentIds);
            console.log('Original selectedDocuments:', selectedDocuments);

            // Double-check we have valid document IDs after processing
            if (documentIds.length === 0) {
                setError('No valid document IDs found. Please reselect documents.');
                console.error('Document processing failed. selectedDocuments:', selectedDocuments);
                return;
            }

            // Ensure model_name is a string (extract name if object)
            const modelName = typeof selectedModel === 'string' ? selectedModel : selectedModel.name;

            const datasetConfig = {
                name: newDatasetName,
                description: newDatasetDescription,
                document_ids: documentIds,
                model_name: modelName,
                num_questions_per_doc: numQuestionsPerDoc,
                difficulty_levels: difficultyLevels,
                user_id: 'admin'
            };

            // Clear any previous errors
            setError('');

            // Start async dataset creation
            console.log('Sending async dataset creation request:', datasetConfig);
            console.log('Document IDs type check:', documentIds.map(id => typeof id));
            console.log('Model name type check:', typeof modelName);
            
            const response = await evaluationApi.createDataset(datasetConfig);
            console.log('Dataset creation task response:', response);

            if (response.task_id) {
                console.log('Started background dataset creation task:', response.task_id);
                
                // Create WebSocket connection to listen for dataset creation updates
                createWebSocketConnection(response.task_id);
                
                // Immediately refresh datasets list to show the new dataset with "running" status
                console.log('Refreshing datasets list to show new dataset');
                loadDatasets();
                
                // Reset form (dialog already closed)
                setNewDatasetName('');
                setNewDatasetDescription('');
                setSelectedDocuments([]);
                setSelectedModel('');
                setNumQuestionsPerDoc(3);
                setDifficultyLevels(['easy', 'medium', 'hard']);
                
                // Show success message briefly
                setError('');
                
                // Note: WebSocket will handle showing the running dataset status
                // The backend will publish updates via WebSocket as the dataset is created
                
            } else {
                setError('Failed to start dataset creation - no task ID returned');
            }
        } catch (error) {
            console.error('Error creating dataset:', error);
            setError('Failed to create dataset: ' + (error.message || 'Unknown error'));
        }
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

            // Get dataset to extract questions/test cases
            const selectedDataset = datasets.find(d => d.id === selectedDatasetId);
            if (!selectedDataset) {
                throw new Error('Selected dataset not found');
            }

            // TODO: For now, create a mock evaluation request. 
            // In a real implementation, you would fetch the dataset's test questions
            // and create evaluation requests for each question-answer pair.
            
            // Send dataset evaluation request to backend
            const datasetEvaluationData = {
                dataset_id: selectedDatasetId,
                dataset_name: selectedDataset.name,
                user_id: localStorage.getItem('username'),
                metadata: {
                    model_id: selectedModelId,
                    retrieval_config: testRetrievalConfig,
                    test_type: 'dataset_evaluation'
                }
            };

            // Start background evaluation with dataset
            const response = await evaluationApi.startDatasetEvaluation(datasetEvaluationData);
            
            if (!response.task_id) {
                throw new Error('No task ID returned from evaluation service');
            }

            console.log('Started background evaluation with task ID:', response.task_id);
            
            // Create test case entry with background task info
            const newTestCase = {
                id: response.task_id,
                task_id: response.task_id,
                dataset_name: selectedDataset.name || 'Unknown',
                models: [selectedModelId],
                retriever: 'Vector Similarity',
                status: 'running',
                created_at: new Date().toISOString(),
                retrieval_config: testRetrievalConfig,
                results: [] // Will be populated when evaluation completes
            };

            setTestCaseResults(prev => [newTestCase, ...prev]);
            
            // Create WebSocket connection for real-time updates
            createWebSocketConnection(response.task_id);
            
            // Initialize progress tracking
            setEvaluationProgress(prev => new Map(prev.set(response.task_id, {
                status: 'PENDING',
                progress: 0,
                message: 'Evaluation task submitted',
                timestamp: new Date().toISOString()
            })));
            
            // Close dialog and reset form
            setCreateTestCaseOpen(false);
            setSelectedDatasetId('');
            setSelectedModelId('');
            
            console.log('Test case created successfully with background evaluation');

        } catch (error) {
            console.error('Error creating test case with background evaluation:', error);
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

    // Load recent background evaluations
    const loadRecentBackgroundEvaluations = async () => {
        console.log('🔄 [DEBUG] loadRecentBackgroundEvaluations() called');
        try {
            console.log('🌐 [DEBUG] Making API call to evaluationApi.getRecentResults(50)');
            const response = await evaluationApi.getRecentResults(50);
            console.log('✅ [DEBUG] evaluationApi.getRecentResults() response:', response);
            const recentEvals = response.results || [];
            console.log('✅ [DEBUG] Processing', recentEvals.length, 'recent evaluations');
            
            // Convert to Map for easier lookup
            const evalMap = new Map();
            recentEvals.forEach(evaluation => {
                evalMap.set(evaluation.task_id, evaluation);
            });
            
            setBackgroundEvaluations(evalMap);
            console.log('✅ [DEBUG] Background evaluations state updated with', evalMap.size, 'items');
            
            // Convert evaluations to testCaseResults format for the table
            const testCaseData = recentEvals.map(evaluation => {
                const metadata = evaluation.metadata || {};
                const isRunning = evaluation.status === 'STARTED' || evaluation.status === 'PENDING';
                
                return {
                    id: evaluation.task_id,
                    task_id: evaluation.task_id,
                    dataset_name: metadata.dataset_name || `Dataset ${metadata.dataset_id || 'Unknown'}`,
                    models: [metadata.model_name || metadata.model_id || 'Unknown Model'],
                    status: isRunning ? 'running' : 'completed',
                    results: (evaluation.status === 'SUCCESS' && evaluation.results) ? [{
                        groundedness: evaluation.results.groundedness?.score || 0,
                        context_relevance: evaluation.results.context_relevance?.score || 0,
                        answer_quality: evaluation.results.answer_relevance?.score || 0,
                        avg_latency: evaluation.results.evaluation_time_seconds || 0
                    }] : [],
                    total_questions: metadata.total_questions || 1,
                    run_by: evaluation.user_id || 'system',
                    run_date: evaluation.completed_at ? 
                        new Date(evaluation.completed_at).toLocaleDateString('en-US', {
                            year: 'numeric',
                            month: 'short',
                            day: 'numeric'
                        }) : 
                        (evaluation.timestamp ? 
                            new Date(evaluation.timestamp).toLocaleDateString('en-US', {
                                year: 'numeric',
                                month: 'short',
                                day: 'numeric'
                            }) : 
                            'Unknown'
                        ),
                    started_at: evaluation.timestamp,
                    completed_at: evaluation.completed_at
                };
            });
            
            setTestCaseResults(testCaseData);
            console.log('✅ [DEBUG] TestCaseResults updated with', testCaseData.length, 'evaluations');
            
            // Restore WebSocket connections for running evaluations
            const runningEvaluations = recentEvals.filter(evaluation => 
                evaluation.status === 'STARTED' || evaluation.status === 'PENDING'
            );
            
            if (runningEvaluations.length > 0) {
                console.log('🔄 [DEBUG] Restoring WebSocket connections for', runningEvaluations.length, 'running evaluations');
                runningEvaluations.forEach(evaluation => {
                    const taskId = evaluation.task_id;
                    if (!activeConnections.has(taskId)) {
                        console.log('🔌 [DEBUG] Restoring WebSocket for task:', taskId);
                        createWebSocketConnection(taskId);
                        
                        // Also restore progress state for running evaluations
                        setEvaluationProgress(prev => new Map(prev.set(taskId, {
                            status: 'STARTED',
                            progress: 0.5, // Default to 50% if we don't know exact progress
                            message: 'Running...',
                            timestamp: new Date().toISOString(),
                            source: 'restored'
                        })));
                    }
                });
            }
            
            // Overview will be automatically calculated by useEffect watching testCaseResults
            
        } catch (error) {
            console.error('❌ [DEBUG] Error loading recent background evaluations:', error);
        }
    };

    // Create WebSocket connection for real-time updates using new WebSocket service
    const createWebSocketConnection = (taskId) => {
        // Check if we already have an active connection
        if (activeConnections.has(taskId)) {
            console.log(`🔄 Connection already exists for task ${taskId}`);
            return null;
        }

        console.log(`🔌 Creating new WebSocket connection for task ${taskId}`);

        // Add to active connections
        setActiveConnections(prev => new Set(prev).add(taskId));

        // Create connection using WebSocket service
        const connection = webSocketService.connect(taskId, {
            onMessage: (message) => {
                handleWebSocketMessage(taskId, message);
            },
            onError: (error) => {
                console.error(`❌ WebSocket error for task ${taskId}:`, error);
                handleWebSocketError(taskId, error);
            },
            onClose: (event) => {
                console.log(`🔒 WebSocket closed for task ${taskId}:`, event.code, event.reason);
                handleWebSocketClose(taskId, event);
            },
            onStatusChange: (status, oldStatus) => {
                console.log(`🔄 Connection status changed for task ${taskId}: ${oldStatus} → ${status}`);
                setConnectionStatuses(prev => new Map(prev).set(taskId, status));
            },
            pollCallback: async (taskId) => {
                // Custom polling logic for this task
                const status = await evaluationApi.getDetailedTaskStatus(taskId);
                
                // Convert polling response to WebSocket message format
                let progressValue = 0;
                
                if (status.progress !== undefined && status.progress !== null && !isNaN(status.progress)) {
                    // Use backend progress if it's a valid number
                    progressValue = status.progress > 1 ? status.progress / 100 : status.progress;
                } else {
                    // Calculate progress from status
                    progressValue = calculateProgressFromStatus(status);
                }
                
                const message = {
                    type: 'evaluation_update',
                    status: status.status,
                    data: {
                        ...status,
                        progress: progressValue,
                        message: getStatusMessage(status)
                    },
                    timestamp: new Date().toISOString(),
                    source: 'polling'
                };
                
                handleWebSocketMessage(taskId, message);
                return status;
            },
            enablePolling: true
        });

        return connection;
    };

    // Handle WebSocket messages with improved logic
    const handleWebSocketMessage = (taskId, message) => {
        console.log(`📨 WebSocket message for task ${taskId}:`, message);

        // Check if this is a dataset creation task
        const isDatasetTask = message.data?.task_type === 'dataset_creation' || 
                            message.data?.action === 'dataset_created' ||
                            message.message?.includes('dataset') ||
                            message.message?.includes('Dataset') ||
                            message.data?.name || // dataset has name
                            message.data?.question_count; // dataset has question_count

        if (message.type === 'evaluation_update') {
            // Update evaluation progress
            setEvaluationProgress(prev => new Map(prev.set(taskId, {
                status: message.status,
                progress: message.data?.progress || 0,
                message: message.data?.message || getStatusMessage(message.data) || '',
                timestamp: message.timestamp,
                source: message.source || 'websocket'
            })));

            // Update dataset generation progress dialog if this is a dataset task
            if (isDatasetTask) {
                setShowProgressDialog(true);
                setDatasetGenerationProgress({
                    status: message.status === 'SUCCESS' ? 'completed' : 
                           message.status === 'FAILURE' ? 'error' :
                           message.data?.message?.includes('Initializing') ? 'starting' : 'processing',
                    progress: Math.round((message.data?.progress || 0) * 100),
                    current_document: message.data?.current_document || '',
                    total_documents: message.data?.total_documents || 0,
                    completed_documents: message.data?.completed_documents || 0,
                    question_count: message.data?.question_count || 0,
                    error: message.status === 'FAILURE' ? (message.data?.error || 'Unknown error') : null
                });
            }

            // Handle task completion
            if (message.status === 'SUCCESS') {
                if (isDatasetTask) {
                    handleDatasetCreationSuccess(taskId, message);
                } else {
                    handleEvaluationSuccess(taskId, message);
                }
            } else if (message.status === 'FAILURE') {
                handleTaskFailure(taskId, message);
            } else if (isDatasetTask) {
                // For dataset tasks, refresh list on any status change
                console.log('📊 Dataset task update, refreshing datasets list');
                loadDatasets();
            }
        }
    };

    // Handle WebSocket errors
    const handleWebSocketError = (taskId, error) => {
        console.error(`❌ WebSocket error for task ${taskId}:`, error);
        // Error handling is managed by WebSocket service with auto-reconnection
        // Just update UI state if needed
    };

    // Handle WebSocket close
    const handleWebSocketClose = (taskId, event) => {
        console.log(`🔒 WebSocket closed for task ${taskId}`);
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
        console.log('✅ Dataset created successfully:', message);
        
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

        // Close progress dialog after showing success for 2 seconds
        setTimeout(() => {
            setShowProgressDialog(false);
            // Reset progress state
            setDatasetGenerationProgress({
                status: 'starting',
                progress: 0,
                current_document: '',
                total_documents: 0,
                completed_documents: 0,
                question_count: 0,
                error: null
            });
        }, 2000);

        // Disconnect WebSocket after short delay
        setTimeout(() => {
            webSocketService.disconnect(taskId);
        }, 2500);
    };

    // Handle evaluation success
    const handleEvaluationSuccess = (taskId, message) => {
        console.log('✅ Evaluation completed successfully:', message);

        // Update background evaluations
        setBackgroundEvaluations(prev => {
            const updated = new Map(prev);
            if (updated.has(taskId)) {
                updated.set(taskId, {
                    ...updated.get(taskId),
                    status: 'SUCCESS',
                    results: message.data
                });
            }
            return updated;
        });

        // Update test case results table
        setTestCaseResults(prev => 
            prev.map(testCase => 
                testCase.task_id === taskId 
                    ? { 
                        ...testCase, 
                        status: 'completed',
                        run_date: new Date().toLocaleDateString('en-US', {
                            year: 'numeric',
                            month: 'short',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit'
                        }),
                        completed_at: new Date().toISOString(),
                        results: [{
                            groundedness: message.data?.results?.groundedness?.score || 0,
                            context_relevance: message.data?.results?.context_relevance?.score || 0,
                            answer_quality: message.data?.results?.answer_relevance?.score || 0,
                            avg_latency: message.data?.evaluation_time || 0
                        }]
                    }
                    : testCase
            )
        );

        // Reload evaluation results from backend
        const reloadData = async () => {
            try {
                // Load evaluation results and recent background evaluations in parallel
                await Promise.all([
                    loadEvaluationResults(),
                    loadRecentBackgroundEvaluations(),
                    loadEvaluationDataWithoutLoading()
                ]);
                
            } catch (error) {
                console.error('Error reloading evaluation data:', error);
            }
        };
        
        // Execute the reload asynchronously without blocking
        reloadData();

        // Disconnect WebSocket after short delay
        setTimeout(() => {
            webSocketService.disconnect(taskId);
        }, 2000);
    };

    // Handle task failure
    const handleTaskFailure = (taskId, message) => {
        console.error('❌ Task failed:', message);

        // Update test case results to show failure
        setTestCaseResults(prev => 
            prev.map(testCase => 
                testCase.task_id === taskId 
                    ? { ...testCase, status: 'failed' }
                    : testCase
            )
        );

        // Close progress dialog after showing error for 3 seconds
        setTimeout(() => {
            setShowProgressDialog(false);
            // Reset progress state
            setDatasetGenerationProgress({
                status: 'starting',
                progress: 0,
                current_document: '',
                total_documents: 0,
                completed_documents: 0,
                question_count: 0,
                error: null
            });
        }, 3000);

        // Disconnect WebSocket after short delay
        setTimeout(() => {
            webSocketService.disconnect(taskId);
        }, 3500);
    };

    // Calculate progress from status data
    const calculateProgressFromStatus = (status) => {
        if (status.status === 'SUCCESS') return 1.0;
        if (status.status === 'FAILURE') return 0;
        if (status.status === 'PROGRESS') return (status.progress || 50) / 100;
        if (status.status === 'STARTED') return 0.1;
        return 0;
    };

    // Get user-friendly status message
    const getStatusMessage = (data) => {
        if (!data) return '';
        
        if (data.message) return data.message;
        
        // Generate message based on status and data
        if (data.status === 'STARTED') {
            if (data.task_type === 'dataset_creation') {
                return 'Creating dataset...';
            }
            return 'Starting evaluation...';
        }
        
        if (data.status === 'PROGRESS') {
            if (data.question_count) {
                return `Processing questions (${data.question_count} total)`;
            }
            return 'Processing...';
        }
        
        if (data.status === 'SUCCESS') {
            if (data.task_type === 'dataset_creation') {
                return `Dataset created with ${data.question_count || 0} questions`;
            }
            return 'Evaluation completed successfully';
        }
        
        return '';
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
            
            // Calculate overview data from real evaluation results instead of API
            calculateOverviewFromRealData();
            
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

    // Calculate overview statistics from real evaluation data
    const calculateOverviewFromRealData = () => {
        try {
            console.log('🧮 [DEBUG] Calculating overview from real data');
            console.log('🧮 [DEBUG] testCaseResults length:', testCaseResults.length);
            console.log('🧮 [DEBUG] testCaseResults:', testCaseResults);
            
            // Get completed evaluations from testCaseResults (same data that evaluation tab shows)
            const completedEvaluations = testCaseResults.filter(testCase => 
                testCase.status === 'completed' && testCase.results && testCase.results.length > 0
            );
            
            console.log('🧮 [DEBUG] completedEvaluations length:', completedEvaluations.length);
            
            if (completedEvaluations.length === 0) {
                // No evaluations yet, show empty state
                console.log('🧮 [DEBUG] No completed evaluations, showing empty state');
                setEvaluationData({
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
                });
                return;
            }

            // Calculate averages from real data
            const totalEvaluations = completedEvaluations.length;
            const avgGroundedness = completedEvaluations.reduce((sum, testCase) => 
                sum + (testCase.results[0].groundedness || 0), 0) / totalEvaluations;
            const avgContextRelevance = completedEvaluations.reduce((sum, testCase) => 
                sum + (testCase.results[0].context_relevance || 0), 0) / totalEvaluations;
            const avgAnswerQuality = completedEvaluations.reduce((sum, testCase) => 
                sum + (testCase.results[0].answer_quality || 0), 0) / totalEvaluations;
            const avgLatency = completedEvaluations.reduce((sum, testCase) => 
                sum + (testCase.results[0].avg_latency || 0), 0) / totalEvaluations;

            console.log('🧮 [DEBUG] Calculated averages:', {
                avgGroundedness, avgContextRelevance, avgAnswerQuality, avgLatency, totalEvaluations
            });

            // Create latency distribution
            const latencyBuckets = [
                { range: "0-1s", count: 0 },
                { range: "1-3s", count: 0 },
                { range: "3-5s", count: 0 },
                { range: "5-10s", count: 0 },
                { range: "10s+", count: 0 }
            ];
            
            completedEvaluations.forEach(testCase => {
                const latency = testCase.results[0].avg_latency || 0;
                if (latency < 1) latencyBuckets[0].count++;
                else if (latency < 3) latencyBuckets[1].count++;
                else if (latency < 5) latencyBuckets[2].count++;
                else if (latency < 10) latencyBuckets[3].count++;
                else latencyBuckets[4].count++;
            });

            // Create historical data (group by date)
            const historical = {};
            completedEvaluations.forEach(testCase => {
                const date = testCase.run_date || 'Unknown';
                
                if (!historical[date]) {
                    historical[date] = {
                        date,
                        groundedness: [],
                        contextRelevance: [],
                        answerQuality: [],
                        latency: [],
                        queries: 0
                    };
                }
                
                historical[date].groundedness.push(testCase.results[0].groundedness || 0);
                historical[date].contextRelevance.push(testCase.results[0].context_relevance || 0);
                historical[date].answerQuality.push(testCase.results[0].answer_quality || 0);
                historical[date].latency.push(testCase.results[0].avg_latency || 0);
                historical[date].queries++;
            });

            // Convert historical data to array with averages
            const historicalArray = Object.values(historical).map(day => ({
                date: day.date,
                groundedness: day.groundedness.reduce((a, b) => a + b, 0) / day.groundedness.length,
                contextRelevance: day.contextRelevance.reduce((a, b) => a + b, 0) / day.contextRelevance.length,
                answerQuality: day.answerQuality.reduce((a, b) => a + b, 0) / day.answerQuality.length,
                latency: day.latency.reduce((a, b) => a + b, 0) / day.latency.length,
                queries: day.queries
            })).sort((a, b) => new Date(a.date) - new Date(b.date));

            // Set the calculated overview data
            const overviewData = {
                overall: {
                    groundedness: avgGroundedness,
                    contextRelevance: avgContextRelevance,
                    answerQuality: avgAnswerQuality,
                    averageLatency: avgLatency,
                    totalQueries: totalEvaluations
                },
                historical: historicalArray,
                latencyDistribution: latencyBuckets,
                detailed: completedEvaluations.map(testCase => ({
                    query: 'Dataset evaluation',
                    groundedness: testCase.results[0].groundedness || 0,
                    contextRelevance: testCase.results[0].context_relevance || 0,
                    answerQuality: testCase.results[0].answer_quality || 0,
                    latency: testCase.results[0].avg_latency || 0,
                    timestamp: testCase.completed_at || testCase.started_at || new Date().toISOString(),
                    model: testCase.models?.[0] || 'Unknown',
                    dataset: testCase.dataset_name || 'Unknown'
                }))
            };
            
            console.log('🧮 [DEBUG] Setting overview data:', overviewData);
            setEvaluationData(overviewData);
            
        } catch (error) {
            console.error('❌ [DEBUG] Error calculating overview from real data:', error);
        }
    };

    // Version of loadEvaluationData without loading state management
    const loadEvaluationDataWithoutLoading = async () => {
        try {
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
                    query: item.question,
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
            throw error; // Re-throw so the caller can handle it
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
                                    onClick={handleCreateDatasetOpen}
                                >
                                    Create Dataset
                                </Button>
                            </Box>
                            
                            {error && (
                                <Alert severity="error" sx={{ mb: 2 }}>
                                    {error}
                                </Alert>
                            )}
                            
                            <Box sx={{ display: 'flex', gap: 2, height: 'calc(100vh - 300px)', minHeight: '500px' }}>
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
                                            maxHeight: '400px',
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
                                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                                        <Chip 
                                                                            label={dataset?.status || 'Unknown'}
                                                                            color={
                                                                                dataset?.status === "Ready" ? "success" : 
                                                                                dataset?.status === "Processing" ? "warning" :
                                                                                dataset?.status === "Error" ? "error" : "default"
                                                                            }
                                                                            size="small"
                                                                        />
                                                                        {dataset?.status === "Processing" && (
                                                                            <CircularProgress size={16} />
                                                                        )}
                                                                    </Box>
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
                                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                                <Chip 
                                                                    label={datasetDetails.status}
                                                                    color={
                                                                        datasetDetails.status === "Ready" ? "success" : 
                                                                        datasetDetails.status === "Processing" ? "warning" :
                                                                        datasetDetails.status === "Error" ? "error" : "default"
                                                                    }
                                                                    size="small"
                                                                />
                                                                {datasetDetails.status === "Processing" && (
                                                                    <CircularProgress size={16} />
                                                                )}
                                                            </Box>
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
                                                                                                    {item.question || 'No question available'}
                                                                                                </Typography>
                                                                                            </Box>
                                                                                        </Box>
                                                                                        
                                                                                        {item.answer && (
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
                                                                                                        {item.answer}
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
                                                                                <strong>Query:</strong> {item.question}
                                                                            </Typography>
                                                                            {item.answer && (
                                                                                <Typography variant="body2" paragraph>
                                                                                    <strong>Expected Response:</strong> {item.answer}
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
                                                    testCaseResults.map((testCase, index) => {
                                                        const progress = evaluationProgress.get(testCase.task_id);
                                                        const isRunning = testCase.status === 'running';
                                                        
                                                        return (
                                                            <TableRow key={testCase.id || testCase.task_id}>
                                                                <TableCell>
                                                                    <Button
                                                                        variant="text"
                                                                        onClick={() => handleTestCaseClick(testCase)}
                                                                        sx={{ textTransform: 'none' }}
                                                                    >
                                                                        #{testCaseResults.length - index}
                                                                    </Button>
                                                                </TableCell>
                                                                <TableCell>{testCase.dataset_name}</TableCell>
                                                                <TableCell>
                                                                    <Chip 
                                                                        label={testCase.models && testCase.models.length > 0 ? testCase.models[0] : 'N/A'} 
                                                                        size="small" 
                                                                        variant="outlined"
                                                                    />
                                                                </TableCell>
                                                                <TableCell>
                                                                    {isRunning ? (
                                                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                                            <CircularProgress size={16} />
                                                                            <Box>
                                                                                <Typography variant="body2">
                                                                                    Processing...
                                                                                </Typography>
                                                                            </Box>
                                                                        </Box>
                                                                    ) : testCase.results && testCase.results.length > 0 ? (
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
                                                                    {isRunning ? (
                                                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                                            <CircularProgress size={16} />
                                                                            <Typography variant="body2">
                                                                                Processing...
                                                                            </Typography>
                                                                        </Box>
                                                                    ) : testCase.results && testCase.results.length > 0 ? (
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
                                                                    {isRunning ? (
                                                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                                            <CircularProgress size={16} />
                                                                            <Typography variant="body2">
                                                                                Processing...
                                                                            </Typography>
                                                                        </Box>
                                                                    ) : testCase.results && testCase.results.length > 0 ? (
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
                                                                    {isRunning ? (
                                                                        <Typography variant="body2" color="text.secondary">
                                                                            -
                                                                        </Typography>
                                                                    ) : testCase.results && testCase.results.length > 0 ? (
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
                                                                    {testCase.run_date}
                                                                </TableCell>
                                                                <TableCell>
                                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                                        <Chip 
                                                                            label={
                                                                                isRunning 
                                                                                    ? "Running..."
                                                                                    : testCase.status
                                                                            }
                                                                            color={
                                                                                testCase.status === "completed" ? "success" : 
                                                                                testCase.status === "running" ? "warning" :
                                                                                testCase.status === "failed" ? "error" : "default"
                                                                            }
                                                                            size="small"
                                                                            icon={isRunning ? <CircularProgress size={12} /> : undefined}
                                                                        />
                                                                    </Box>
                                                                </TableCell>
                                                            </TableRow>
                                                        );
                                                    })
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
                                                            console.log('Checkbox clicked for document:', doc);
                                                            console.log('Document ID:', docId);
                                                            console.log('Checked:', e.target.checked);
                                                            console.log('Current selectedDocuments:', selectedDocuments);
                                                            
                                                            if (e.target.checked) {
                                                                const newSelection = [...selectedDocuments, docId];
                                                                console.log('Adding document, new selection:', newSelection);
                                                                setSelectedDocuments(newSelection);
                                                            } else {
                                                                const newSelection = selectedDocuments.filter(id => id !== docId);
                                                                console.log('Removing document, new selection:', newSelection);
                                                                setSelectedDocuments(newSelection);
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
                                {/* Info about background evaluation */}
                                <Alert severity="info" sx={{ mb: 2 }}>
                                    <Typography variant="body2">
                                        Evaluations run in the background using our async processing system. 
                                        You can close this dialog and monitor progress in real-time in the evaluations table.
                                    </Typography>
                                </Alert>

                                {/* Dataset Selector */}
                                <FormControl fullWidth>
                                    <InputLabel>Select Dataset</InputLabel>
                                    <Select
                                        value={selectedDatasetId}
                                        onChange={(e) => {
                                            console.log('Dataset selected:', e.target.value);
                                            setSelectedDatasetId(e.target.value);
                                        }}
                                        input={<OutlinedInput label="Select Dataset" />}
                                    >
                                        {(() => {
                                            console.log('🔍 Create Test Case - Total datasets:', datasets.length);
                                            console.log('🔍 Datasets array:', datasets);
                                            const completedDatasets = datasets.filter(d => d?.status === "completed");
                                            console.log('✅ Completed datasets:', completedDatasets.length);
                                            console.log('✅ Completed datasets array:', completedDatasets);
                                            return completedDatasets;
                                        })().map((dataset) => (
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
                                {runningTestCase ? 'Starting Evaluation...' : 'Start Background Evaluation'}
                            </Button>
                        </DialogActions>
                    </Dialog>

                    {/* Test Case Details Dialog */}
                    <Dialog 
                        open={testCaseDetailsOpen} 
                        onClose={() => setTestCaseDetailsOpen(false)}
                        maxWidth="md"
                        fullWidth
                    >
                        <DialogTitle>
                            Evaluation Details - Test #{testCaseResults.length - testCaseResults.findIndex(tc => tc.task_id === selectedTestCase?.task_id)}
                        </DialogTitle>
                        <DialogContent>
                            {selectedTestCase && (
                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                                    {/* Basic Information */}
                                    <Card>
                                        <CardContent>
                                            <Typography variant="h6" gutterBottom>Basic Information</Typography>
                                            <Grid container spacing={2}>
                                                <Grid item xs={6}>
                                                    <Typography variant="body2" color="text.secondary">Task ID</Typography>
                                                    <Typography variant="body1">{selectedTestCase.task_id}</Typography>
                                                </Grid>
                                                <Grid item xs={6}>
                                                    <Typography variant="body2" color="text.secondary">Dataset</Typography>
                                                    <Typography variant="body1">{selectedTestCase.dataset_name}</Typography>
                                                </Grid>
                                                <Grid item xs={6}>
                                                    <Typography variant="body2" color="text.secondary">Status</Typography>
                                                    <Chip 
                                                        label={selectedTestCase.status} 
                                                        color={selectedTestCase.status === 'completed' ? 'success' : selectedTestCase.status === 'running' ? 'warning' : 'error'}
                                                        size="small"
                                                    />
                                                </Grid>
                                                <Grid item xs={6}>
                                                    <Typography variant="body2" color="text.secondary">Run Date</Typography>
                                                    <Typography variant="body1">{selectedTestCase.run_date}</Typography>
                                                </Grid>
                                            </Grid>
                                        </CardContent>
                                    </Card>

                                    {/* Model and Configuration */}
                                    <Card>
                                        <CardContent>
                                            <Typography variant="h6" gutterBottom>Model Configuration</Typography>
                                            <Grid container spacing={2}>
                                                <Grid item xs={12}>
                                                    <Typography variant="body2" color="text.secondary">LLM Model</Typography>
                                                    <Typography variant="body1">{selectedTestCase.models?.[0] || 'N/A'}</Typography>
                                                </Grid>
                                                {/* Show retrieval settings if available */}
                                                {backgroundEvaluations.get(selectedTestCase.task_id)?.metadata && (
                                                    <>
                                                        <Grid item xs={6}>
                                                            <Typography variant="body2" color="text.secondary">Similarity Threshold</Typography>
                                                            <Typography variant="body1">
                                                                {backgroundEvaluations.get(selectedTestCase.task_id).metadata.retrieval_config?.similarity_threshold || 'N/A'}
                                                            </Typography>
                                                        </Grid>
                                                        <Grid item xs={6}>
                                                            <Typography variant="body2" color="text.secondary">Max Chunks</Typography>
                                                            <Typography variant="body1">
                                                                {backgroundEvaluations.get(selectedTestCase.task_id).metadata.retrieval_config?.max_chunks || 'N/A'}
                                                            </Typography>
                                                        </Grid>
                                                        <Grid item xs={6}>
                                                            <Typography variant="body2" color="text.secondary">Search Type</Typography>
                                                            <Typography variant="body1">
                                                                {backgroundEvaluations.get(selectedTestCase.task_id).metadata.retrieval_config?.search_type || 'N/A'}
                                                            </Typography>
                                                        </Grid>
                                                        <Grid item xs={6}>
                                                            <Typography variant="body2" color="text.secondary">Reranker Enabled</Typography>
                                                            <Typography variant="body1">
                                                                {backgroundEvaluations.get(selectedTestCase.task_id).metadata.retrieval_config?.reranker_enabled ? 'Yes' : 'No'}
                                                            </Typography>
                                                        </Grid>
                                                    </>
                                                )}
                                            </Grid>
                                        </CardContent>
                                    </Card>

                                    {/* Results */}
                                    {selectedTestCase.results && selectedTestCase.results.length > 0 && (
                                        <Card>
                                            <CardContent>
                                                <Typography variant="h6" gutterBottom>Evaluation Results</Typography>
                                                <Grid container spacing={2}>
                                                    <Grid item xs={3}>
                                                        <Typography variant="body2" color="text.secondary">Groundedness</Typography>
                                                        <Typography variant="h6" color="primary">
                                                            {(selectedTestCase.results[0].groundedness * 100).toFixed(1)}%
                                                        </Typography>
                                                    </Grid>
                                                    <Grid item xs={3}>
                                                        <Typography variant="body2" color="text.secondary">Context Relevance</Typography>
                                                        <Typography variant="h6" color="primary">
                                                            {(selectedTestCase.results[0].context_relevance * 100).toFixed(1)}%
                                                        </Typography>
                                                    </Grid>
                                                    <Grid item xs={3}>
                                                        <Typography variant="body2" color="text.secondary">Answer Quality</Typography>
                                                        <Typography variant="h6" color="primary">
                                                            {(selectedTestCase.results[0].answer_quality * 100).toFixed(1)}%
                                                        </Typography>
                                                    </Grid>
                                                    <Grid item xs={3}>
                                                        <Typography variant="body2" color="text.secondary">Avg Latency</Typography>
                                                        <Typography variant="h6" color="primary">
                                                            {selectedTestCase.results[0].avg_latency.toFixed(2)}s
                                                        </Typography>
                                                    </Grid>
                                                </Grid>
                                            </CardContent>
                                        </Card>
                                    )}
                                </Box>
                            )}
                        </DialogContent>
                        <DialogActions>
                            <Button onClick={() => setTestCaseDetailsOpen(false)}>
                                Close
                            </Button>
                        </DialogActions>
                    </Dialog>
                </Box>
            </Box>
        </ThemeProvider>
    );
};

export default EvaluationPage;