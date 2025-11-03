import React, { useState, useEffect, useRef } from 'react';
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
    Pause,
    PlayArrow,
    Stop,
    Refresh
} from '@mui/icons-material';
import ReactECharts from 'echarts-for-react';
import { useTheme } from '@mui/material/styles';
import { api } from '../services/api';
import LossChart from './charts/LossChart';
import LearningRateChart from './charts/LearningRateChart';
import SystemChart from './charts/SystemChart';

const TrainingDashboard = ({ experimentId, onClose }) => {
    const theme = useTheme();
    const [metrics, setMetrics] = useState(null);
    const [tabValue, setTabValue] = useState(0);
    const [autoRefresh, setAutoRefresh] = useState(true);
    const intervalRef = useRef(null);

    // Fetch metrics function
    const fetchMetrics = async () => {
        try {
            const response = await api.getTrainingMetrics(experimentId);
            setMetrics(response);
        } catch (error) {
            console.error('Failed to fetch training metrics:', error);
        }
    };

    // Set up polling
    useEffect(() => {
        if (autoRefresh) {
            fetchMetrics(); // Initial fetch
            intervalRef.current = setInterval(fetchMetrics, 2000); // Poll every 2 seconds
        } else {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
        }

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, [autoRefresh, experimentId]);

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

    const progressData = metrics?.progress;
    const chartData = metrics?.metrics;
    const logsData = metrics?.training_logs || [];
    const systemData = metrics?.system;

    return (
        <Box sx={{ width: '100%', height: '100%' }}>
            {/* Header with controls */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Typography variant="h5" fontWeight={600}>
                    Training Dashboard - Experiment {experimentId.slice(-8)}
                </Typography>
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
                        {systemData && <SystemChart data={systemData} theme={theme} />}
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