import React, { useState, useEffect } from 'react';
import {
    Box,
    Grid,
    Card,
    CardContent,
    Typography,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Chip,
    Button,
    Alert,
    CircularProgress,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper
} from '@mui/material';
import { Compare, TrendingUp, Speed, Timer } from '@mui/icons-material';
import ReactECharts from 'echarts-for-react';
import { useTheme } from '@mui/material/styles';

const ExperimentComparison = ({ availableExperiments }) => {
    const theme = useTheme();
    const [selectedExperiments, setSelectedExperiments] = useState([]);
    const [comparisonData, setComparisonData] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleExperimentSelection = (experimentId) => {
        setSelectedExperiments(prev => {
            if (prev.includes(experimentId)) {
                return prev.filter(id => id !== experimentId);
            } else if (prev.length < 5) { // Limit to 5 experiments
                return [...prev, experimentId];
            }
            return prev;
        });
    };

    const fetchComparisonData = async () => {
        if (selectedExperiments.length === 0) return;
        
        setIsLoading(true);
        try {
            const promises = selectedExperiments.map(expId =>
                fetch(`/api/finetuning/experiments/${expId}/metrics`).then(res => res.json())
            );
            const results = await Promise.all(promises);
            
            // Combine data for comparison
            const combined = {
                experiments: selectedExperiments.map((expId, index) => ({
                    id: expId,
                    name: availableExperiments.find(e => e.id === expId)?.name || `Exp ${expId.slice(-8)}`,
                    data: results[index]
                }))
            };
            
            setComparisonData(combined);
        } catch (error) {
            console.error('Error fetching comparison data:', error);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchComparisonData();
    }, [selectedExperiments]);

    // Generate colors for each experiment
    const colors = [
        theme.palette.primary.main,
        theme.palette.secondary.main,
        theme.palette.success.main,
        theme.palette.warning.main,
        theme.palette.error.main
    ];

    // Loss comparison chart
    const lossComparisonOptions = {
        title: {
            text: 'Training Loss Comparison',
            left: 'center',
            textStyle: { color: theme.palette.text.primary, fontSize: 16, fontWeight: 'bold' }
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'cross' }
        },
        legend: {
            data: comparisonData?.experiments?.map(exp => exp.name) || [],
            textStyle: { color: theme.palette.text.primary }
        },
        grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
        xAxis: {
            type: 'value',
            name: 'Step',
            axisLabel: { color: theme.palette.text.secondary },
            axisLine: { lineStyle: { color: theme.palette.divider } }
        },
        yAxis: {
            type: 'value',
            name: 'Loss',
            axisLabel: { color: theme.palette.text.secondary },
            axisLine: { lineStyle: { color: theme.palette.divider } }
        },
        series: comparisonData?.experiments?.map((exp, index) => ({
            name: exp.name,
            type: 'line',
            data: exp.data?.metrics?.train_losses?.map(item => [item.step, item.value]) || [],
            itemStyle: { color: colors[index % colors.length] },
            smooth: true
        })) || []
    };

    // Learning rate comparison chart
    const learningRateComparisonOptions = {
        title: {
            text: 'Learning Rate Comparison',
            left: 'center',
            textStyle: { color: theme.palette.text.primary, fontSize: 16, fontWeight: 'bold' }
        },
        tooltip: { trigger: 'axis' },
        legend: {
            data: comparisonData?.experiments?.map(exp => exp.name) || [],
            textStyle: { color: theme.palette.text.primary }
        },
        grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
        xAxis: {
            type: 'value',
            name: 'Step',
            axisLabel: { color: theme.palette.text.secondary },
            axisLine: { lineStyle: { color: theme.palette.divider } }
        },
        yAxis: {
            type: 'value',
            name: 'Learning Rate',
            axisLabel: { color: theme.palette.text.secondary },
            axisLine: { lineStyle: { color: theme.palette.divider } }
        },
        series: comparisonData?.experiments?.map((exp, index) => ({
            name: exp.name,
            type: 'line',
            data: exp.data?.metrics?.learning_rates?.map(item => [item.step, item.value]) || [],
            itemStyle: { color: colors[index % colors.length] },
            smooth: true
        })) || []
    };

    const formatTime = (seconds) => {
        if (!seconds) return 'N/A';
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <Box sx={{ width: '100%' }}>
            <Typography variant="h5" fontWeight={600} sx={{ mb: 3 }}>
                <Compare sx={{ mr: 1, verticalAlign: 'middle' }} />
                Experiment Comparison
            </Typography>

            {/* Experiment Selection */}
            <Card sx={{ mb: 3 }}>
                <CardContent>
                    <Typography variant="h6" sx={{ mb: 2 }}>
                        Select Experiments to Compare (max 5)
                    </Typography>
                    <Grid container spacing={2}>
                        {availableExperiments.map((experiment) => (
                            <Grid item key={experiment.id}>
                                <Chip
                                    label={experiment.name || `Exp ${experiment.id.slice(-8)}`}
                                    onClick={() => handleExperimentSelection(experiment.id)}
                                    color={selectedExperiments.includes(experiment.id) ? "primary" : "default"}
                                    variant={selectedExperiments.includes(experiment.id) ? "filled" : "outlined"}
                                />
                            </Grid>
                        ))}
                    </Grid>
                    {selectedExperiments.length > 0 && (
                        <Box sx={{ mt: 2 }}>
                            <Typography variant="body2" color="text.secondary">
                                Selected: {selectedExperiments.length} experiments
                            </Typography>
                            <Button
                                variant="contained"
                                onClick={fetchComparisonData}
                                sx={{ mt: 1 }}
                                disabled={isLoading}
                            >
                                {isLoading ? <CircularProgress size={20} /> : 'Update Comparison'}
                            </Button>
                        </Box>
                    )}
                </CardContent>
            </Card>

            {selectedExperiments.length === 0 && (
                <Alert severity="info">
                    Select at least one experiment to view comparison charts.
                </Alert>
            )}

            {selectedExperiments.length > 0 && comparisonData && (
                <>
                    {/* Summary Table */}
                    <Card sx={{ mb: 3 }}>
                        <CardContent>
                            <Typography variant="h6" sx={{ mb: 2 }}>
                                Experiment Summary
                            </Typography>
                            <TableContainer component={Paper}>
                                <Table size="small">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Experiment</TableCell>
                                            <TableCell align="right">Progress</TableCell>
                                            <TableCell align="right">Final Loss</TableCell>
                                            <TableCell align="right">Best Accuracy</TableCell>
                                            <TableCell align="right">Training Time</TableCell>
                                            <TableCell align="right">Speed (samples/sec)</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {comparisonData.experiments.map((exp, index) => {
                                            const progress = exp.data?.progress || {};
                                            const metrics = exp.data?.metrics || {};
                                            const finalLoss = metrics.train_losses?.[metrics.train_losses.length - 1]?.value;
                                            const bestAccuracy = Math.max(...(metrics.accuracies?.map(a => a.value) || [0]));
                                            
                                            return (
                                                <TableRow key={exp.id}>
                                                    <TableCell>
                                                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                                            <Box
                                                                sx={{
                                                                    width: 12,
                                                                    height: 12,
                                                                    bgcolor: colors[index % colors.length],
                                                                    borderRadius: '50%',
                                                                    mr: 1
                                                                }}
                                                            />
                                                            {exp.name}
                                                        </Box>
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        {progress.epoch_progress?.toFixed(1) || 0}%
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        {finalLoss?.toFixed(4) || 'N/A'}
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        {bestAccuracy > 0 ? bestAccuracy.toFixed(4) : 'N/A'}
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        {formatTime(progress.elapsed_time)}
                                                    </TableCell>
                                                    <TableCell align="right">
                                                        {progress.samples_per_sec?.toFixed(1) || 'N/A'}
                                                    </TableCell>
                                                </TableRow>
                                            );
                                        })}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </CardContent>
                    </Card>

                    {/* Comparison Charts */}
                    <Grid container spacing={3}>
                        <Grid item xs={12} md={6}>
                            <Card>
                                <CardContent>
                                    <Box sx={{ height: 400 }}>
                                        <ReactECharts option={lossComparisonOptions} style={{ height: '100%' }} />
                                    </Box>
                                </CardContent>
                            </Card>
                        </Grid>
                        <Grid item xs={12} md={6}>
                            <Card>
                                <CardContent>
                                    <Box sx={{ height: 400 }}>
                                        <ReactECharts option={learningRateComparisonOptions} style={{ height: '100%' }} />
                                    </Box>
                                </CardContent>
                            </Card>
                        </Grid>
                    </Grid>
                </>
            )}
        </Box>
    );
};

export default ExperimentComparison;