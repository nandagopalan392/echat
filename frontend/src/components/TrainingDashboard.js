import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Box,
    Grid,
    Card,
    CardContent,
    Typography,
    LinearProgress,
    Chip,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    Button,
    Tab,
    Tabs,
    Switch,
    FormControlLabel,
    Alert,
    CircularProgress
} from '@mui/material';
import {
    TrendingUp,
    Memory,
    Speed,
    Timer,
    Refresh
} from '@mui/icons-material';
import ReactECharts from 'echarts-for-react';
import { useTheme } from '@mui/material/styles';
import { api } from '../services/api';
import LossChart from './charts/LossChart';
import LearningRateChart from './charts/LearningRateChart';
import SystemChart from './charts/SystemChart';

/**
 * TrainingDashboard - displays real-time training metrics
 * 
 * Props:
 * - experimentId: The experiment ID to display
 * - trainingUpdate: WebSocket update passed from parent (FinetuningPage)
 * - onClose: Callback when dashboard is closed
 */
const TrainingDashboard = ({ experimentId, trainingUpdate, onClose }) => {
    const theme = useTheme();
    const [metrics, setMetrics] = useState(null);
    const [tabValue, setTabValue] = useState(0);
    const [autoRefresh, setAutoRefresh] = useState(true);
    const [connectionStatus, setConnectionStatus] = useState('connected');
    const [wsError, setWsError] = useState(null);
    const intervalRef = useRef(null);
    const lossHistoryRef = useRef([]); // Store loss history across renders

    // Fetch metrics function (HTTP fallback)
    const fetchMetrics = useCallback(async () => {
        try {
            const response = await api.getFineTuningMetrics(experimentId);
            setMetrics(response);
            return response;
        } catch (error) {
            console.error('Failed to fetch training metrics:', error);
            return null;
        }
    }, [experimentId]);

    // Process WebSocket update passed from parent
    const processTrainingUpdate = useCallback((message) => {
        console.log('📨 TrainingDashboard received update:', message);

        if (message.type === 'experiment_update') {
            // Extract loss data from latest_logs (which contains actual training logs with loss values)
            const latestLogs = message.latest_logs || [];
            
            // Build loss history from training logs
            latestLogs.forEach(log => {
                const logStep = log.step;
                const logLoss = log.loss || log.train_loss;
                const logEvalLoss = log.eval_loss;
                
                if (logStep !== undefined && logStep !== null) {
                    // Add train loss
                    if (logLoss !== undefined && logLoss !== null) {
                        const existingTrainStep = lossHistoryRef.current.find(h => h.step === logStep && h.type === 'train');
                        if (!existingTrainStep) {
                            lossHistoryRef.current.push({ step: logStep, loss: logLoss, type: 'train' });
                        }
                    }
                    // Add eval loss
                    if (logEvalLoss !== undefined && logEvalLoss !== null) {
                        const existingEvalStep = lossHistoryRef.current.find(h => h.step === logStep && h.type === 'eval');
                        if (!existingEvalStep) {
                            lossHistoryRef.current.push({ step: logStep, loss: logEvalLoss, type: 'eval' });
                        }
                    }
                }
            });
            
            // Keep last 500 data points
            if (lossHistoryRef.current.length > 500) {
                lossHistoryRef.current = lossHistoryRef.current.slice(-500);
            }
            
            // Calculate progress properly - use progress_info if available, otherwise estimate from epoch
            let epochProgress = 0;
            const progressInfo = message.progress_info || {};
            const currentEpoch = progressInfo.current_epoch || 0;
            const totalEpochs = progressInfo.total_epochs || 3;
            
            if (progressInfo.progress !== undefined && progressInfo.progress !== null && progressInfo.progress > 0) {
                // Use the backend-provided progress (0.0 to 1.0 scale)
                epochProgress = progressInfo.progress * 100;
            } else if (currentEpoch > 0 && totalEpochs > 0) {
                // Estimate from epoch info
                epochProgress = (currentEpoch / totalEpochs) * 100;
            }
            
            // Transform WebSocket data to metrics format
            const wsMetrics = {
                progress: {
                    epoch_progress: epochProgress,
                    current_epoch: currentEpoch,
                    total_epochs: totalEpochs,
                    elapsed_time: message.training_metrics?.elapsed_time,
                    eta: message.training_metrics?.eta,
                    samples_per_sec: message.training_metrics?.samples_per_sec
                },
                training_logs: latestLogs,
                system: message.training_metrics?.system || {},
                status: message.status,
                message: progressInfo.message
            };

            const lossHistory = lossHistoryRef.current;
            const trainLosses = lossHistory.filter(h => h.type === 'train').map(h => ({ step: h.step, value: h.loss }));
            const evalLosses = lossHistory.filter(h => h.type === 'eval').map(h => ({ step: h.step, value: h.loss }));

            // Merge with existing metrics to preserve history and build chart data
            setMetrics(prev => {
                const prevMetrics = prev || {};
                
                // Keep the best progress (don't go backwards)
                const prevProgress = prevMetrics.progress?.epoch_progress || 0;
                const newProgress = wsMetrics.progress.epoch_progress;
                const finalProgress = message.status === 'completed' ? 100 : Math.max(prevProgress, newProgress);
                
                return {
                    ...prevMetrics,
                    ...wsMetrics,
                    progress: {
                        ...wsMetrics.progress,
                        epoch_progress: finalProgress
                    },
                    // Preserve and update loss history
                    loss_history: lossHistory,
                    // Merge training logs, keeping history
                    training_logs: wsMetrics.training_logs.length > 0 
                        ? [...(prevMetrics.training_logs || []).filter(log => 
                            !wsMetrics.training_logs.some(newLog => 
                                newLog.step === log.step && newLog.epoch === log.epoch
                            )
                          ), ...wsMetrics.training_logs].sort((a, b) => 
                            (a.step || 0) - (b.step || 0)
                          )
                        : prevMetrics.training_logs || [],
                    // Update metrics with loss history formatted for LossChart
                    metrics: {
                        ...(prevMetrics.metrics || {}),
                        ...(message.metrics || {}),
                        // Format for LossChart component: [{step, value}, ...]
                        train_losses: trainLosses,
                        eval_losses: evalLosses
                    }
                };
            });
            
            setConnectionStatus('connected');
        } else if (message.type === 'completion') {
            console.log('🏁 Training completed:', message.final_status);
            // Final fetch to get complete data
            fetchMetrics();
        } else if (message.type === 'error') {
            console.error('❌ WebSocket error message:', message.message);
            setWsError(message.message);
        }
    }, [fetchMetrics]);

    // Process trainingUpdate prop when it changes (from parent WebSocket)
    useEffect(() => {
        if (trainingUpdate) {
            processTrainingUpdate(trainingUpdate);
        }
    }, [trainingUpdate, processTrainingUpdate]);

    // Initial fetch and optional polling fallback
    useEffect(() => {
        // Initial fetch to get current state
        fetchMetrics();

        // Set up polling as a fallback (slower rate since WebSocket is primary)
        if (autoRefresh) {
            intervalRef.current = setInterval(fetchMetrics, 5000); // Poll every 5 seconds as fallback
        }

        // Cleanup on unmount
        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
        };
    }, [autoRefresh, experimentId, fetchMetrics]);

    const onRefresh = () => {
        fetchMetrics();
    };

    const formatTimestamp = (timestamp) => {
        if (!timestamp) return 'N/A';
        try {
            if (typeof timestamp === 'number') {
                const date = new Date(timestamp > 1e12 ? timestamp : timestamp * 1000);
                return isNaN(date.getTime()) ? 'Invalid Date' : date.toLocaleString();
            }
            if (typeof timestamp === 'string') {
                const normalized = timestamp.includes('T') ? timestamp : timestamp.replace(' ', 'T');
                const date = new Date(normalized);
                return isNaN(date.getTime()) ? 'Invalid Date' : date.toLocaleString();
            }
            return 'Invalid Date';
        } catch (error) {
            return 'Invalid Date';
        }
    };

    const formatTime = (seconds) => {
        if (!seconds) return 'N/A';
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    const formatBytes = (bytes) => {
        if (!bytes) return 'N/A';
        const gb = bytes / (1024 ** 3);
        return `${gb.toFixed(1)} GB`;
    };

    if (!metrics) {
        return (
            <Box display="flex" justifyContent="center" alignItems="center" p={4}>
                <CircularProgress />
                <Typography variant="h6" sx={{ ml: 2 }}>Loading training data...</Typography>
            </Box>
        );
    }

    const progressData = metrics?.progress || {};
    const chartData = metrics?.metrics || {};
    const logsData = metrics?.training_logs || [];
    const systemData = metrics?.system || {};

    // Get connection status color and label (no icon)
    const getConnectionStatusDisplay = () => {
        switch (connectionStatus) {
            case 'connected':
                return { color: 'success', label: 'Live' };
            case 'connecting':
            case 'reconnecting':
                return { color: 'warning', label: 'Connecting...' };
            case 'polling':
                return { color: 'info', label: 'Polling' };
            default:
                return { color: 'default', label: 'Offline' };
        }
    };

    const statusDisplay = getConnectionStatusDisplay();

    return (
        <Box sx={{ width: '100%', height: '100%' }}>
            {/* WebSocket Error Alert */}
            {wsError && (
                <Alert severity="warning" sx={{ mb: 2 }} onClose={() => setWsError(null)}>
                    {wsError} - Using polling fallback
                </Alert>
            )}

            {/* Header with controls */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Typography variant="h5" fontWeight={600}>
                        Training Dashboard - Experiment {experimentId.slice(-8)}
                    </Typography>
                    <Chip
                        label={statusDisplay.label}
                        color={statusDisplay.color}
                        size="small"
                        variant="outlined"
                    />
                </Box>
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                    <FormControlLabel
                        control={
                            <Switch
                                checked={autoRefresh}
                                onChange={(e) => setAutoRefresh(e.target.checked)}
                            />
                        }
                        label="Auto Refresh"
                    />
                    <Button
                        variant="outlined"
                        startIcon={<Refresh />}
                        onClick={onRefresh}
                        size="small"
                    >
                        Refresh
                    </Button>
                    <Button
                        variant="contained"
                        onClick={onClose}
                        size="small"
                    >
                        Close
                    </Button>
                </Box>
            </Box>

            {/* Summary Cards */}
            <Grid container spacing={3} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={6} md={3}>
                    <Card>
                        <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                <Timer sx={{ color: 'primary.main', mr: 1 }} />
                                <Typography variant="h6">Progress</Typography>
                            </Box>
                            <Typography variant="h4" fontWeight={600} color="primary">
                                {progressData?.epoch_progress?.toFixed(1) || 0}%
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                Epoch {progressData?.current_epoch || 0} / {progressData?.total_epochs || 0}
                            </Typography>
                            <LinearProgress
                                variant="determinate"
                                value={progressData?.epoch_progress || 0}
                                sx={{ mt: 1 }}
                            />
                        </CardContent>
                    </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                    <Card>
                        <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                <Speed sx={{ color: 'success.main', mr: 1 }} />
                                <Typography variant="h6">Speed</Typography>
                            </Box>
                            <Typography variant="h4" fontWeight={600} color="success.main">
                                {progressData?.samples_per_sec?.toFixed(1) || 0}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                samples/sec
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                    <Card>
                        <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                <Timer sx={{ color: 'warning.main', mr: 1 }} />
                                <Typography variant="h6">Time</Typography>
                            </Box>
                            <Typography variant="h4" fontWeight={600} color="warning.main">
                                {formatTime(progressData?.elapsed_time)}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                ETA: {formatTime(progressData?.eta)}
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                    <Card>
                        <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                <Memory sx={{ color: 'error.main', mr: 1 }} />
                                <Typography variant="h6">GPU Memory</Typography>
                            </Box>
                            <Typography variant="h4" fontWeight={600} color="error.main">
                                {systemData?.gpu_metrics?.[0]?.memory_percent?.toFixed(1) || 0}%
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                {formatBytes(systemData?.gpu_metrics?.[0]?.memory_used * 1024 * 1024 * 1024)} / {formatBytes(systemData?.gpu_metrics?.[0]?.memory_total * 1024 * 1024 * 1024)}
                            </Typography>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>

            {/* Tabs for different views */}
            <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
                <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)}>
                    <Tab label="Training Loss" />
                    <Tab label="Learning Rate" />
                    <Tab label="System" />
                    <Tab label="Logs" />
                </Tabs>
            </Box>

            {/* Tab content */}
            <Box>
                {tabValue === 0 && (
                    <Box>
                        <Typography variant="h6" gutterBottom>Training Loss Over Time</Typography>
                        {chartData && <LossChart data={chartData} theme={theme} />}
                    </Box>
                )}
                {tabValue === 1 && (
                    <Box>
                        <Typography variant="h6" gutterBottom>Learning Rate Schedule</Typography>
                        {chartData && <LearningRateChart data={chartData} theme={theme} />}
                    </Box>
                )}
                {tabValue === 2 && (
                    <Box>
                        <Typography variant="h6" gutterBottom>System Metrics</Typography>
                        {systemData && typeof systemData === 'object' && Object.keys(systemData).length > 0 ? (
                            <SystemChart data={systemData} theme={theme} />
                        ) : (
                            <Typography variant="body2" color="text.secondary">No system metrics data available</Typography>
                        )}
                    </Box>
                )}
                {tabValue === 3 && (
                    <Box>
                        <Typography variant="h6" gutterBottom>Training Logs</Typography>
                        <TableContainer component={Paper} sx={{ maxHeight: 500 }}>
                            <Table stickyHeader>
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Timestamp</TableCell>
                                        <TableCell>Epoch</TableCell>
                                        <TableCell>Step</TableCell>
                                        <TableCell align="right">Train Loss</TableCell>
                                        <TableCell align="right">Eval Loss</TableCell>
                                        <TableCell align="right">Learning Rate</TableCell>
                                        <TableCell align="right">Accuracy</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {logsData.map((log, index) => (
                                        <TableRow key={`${log.step}-${log.epoch}-${index}`}>
                                            <TableCell>
                                                {formatTimestamp(log.timestamp)}
                                            </TableCell>
                                            <TableCell>{log.epoch ?? 'N/A'}</TableCell>
                                            <TableCell>{log.step ?? 'N/A'}</TableCell>
                                            <TableCell align="right">
                                                {typeof log.train_loss === 'number' ? log.train_loss.toFixed(4) :
                                                 typeof log.loss === 'number' ? log.loss.toFixed(4) : 'N/A'}
                                            </TableCell>
                                            <TableCell align="right">
                                                {typeof log.eval_loss === 'number' ? log.eval_loss.toFixed(4) : 'N/A'}
                                            </TableCell>
                                            <TableCell align="right">
                                                {typeof log.learning_rate === 'number' && log.learning_rate !== 0 ? log.learning_rate.toExponential(2) : 'N/A'}
                                            </TableCell>
                                            <TableCell align="right">
                                                {typeof log.accuracy === 'number' ? (log.accuracy * 100).toFixed(2) + '%' : 'N/A'}
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    </Box>
                )}
            </Box>
        </Box>
    );
};

export default TrainingDashboard;