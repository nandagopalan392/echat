import React, { useState, useEffect } from 'react';
import { Box, Typography, Card, CardContent, Grid, LinearProgress } from '@mui/material';

const SystemChart = ({ data, theme }) => {
  const [systemMetrics, setSystemMetrics] = useState({
    cpu: 0,
    memory: 0,
    gpu: 0,
    gpuMemory: 0
  });

  useEffect(() => {
    if (!data || typeof data !== 'object') {
      setSystemMetrics({ cpu: 0, memory: 0, gpu: 0, gpuMemory: 0 });
      return;
    }

    try {
      let metrics = { cpu: 0, memory: 0, gpu: 0, gpuMemory: 0 };

      // Handle current data
      if (data.current && typeof data.current === 'object') {
        const current = data.current;
        metrics.cpu = typeof current.cpu_percent === 'number' ? Math.min(100, Math.max(0, current.cpu_percent)) : 0;
        metrics.memory = typeof current.memory_percent === 'number' ? Math.min(100, Math.max(0, current.memory_percent)) : 0;
        
        if (current.gpu_metrics && Array.isArray(current.gpu_metrics) && current.gpu_metrics.length > 0) {
          const gpu = current.gpu_metrics[0];
          if (gpu && typeof gpu === 'object') {
            metrics.gpu = typeof gpu.utilization === 'number' ? Math.min(100, Math.max(0, gpu.utilization)) : 
                         typeof gpu.load === 'number' ? Math.min(100, Math.max(0, gpu.load)) : 0;
            metrics.gpuMemory = typeof gpu.memory_percent === 'number' ? Math.min(100, Math.max(0, gpu.memory_percent)) : 0;
          }
        }
      }

      // Handle history data (use latest entry)
      if (data.history && Array.isArray(data.history) && data.history.length > 0) {
        const latest = data.history[data.history.length - 1];
        if (latest && typeof latest === 'object') {
          metrics.cpu = typeof latest.cpu_percent === 'number' ? Math.min(100, Math.max(0, latest.cpu_percent)) : metrics.cpu;
          metrics.memory = typeof latest.memory_percent === 'number' ? Math.min(100, Math.max(0, latest.memory_percent)) : metrics.memory;
          
          if (latest.gpu_metrics && Array.isArray(latest.gpu_metrics) && latest.gpu_metrics.length > 0) {
            const gpu = latest.gpu_metrics[0];
            if (gpu && typeof gpu === 'object') {
              metrics.gpu = typeof gpu.utilization === 'number' ? Math.min(100, Math.max(0, gpu.utilization)) : 
                           typeof gpu.load === 'number' ? Math.min(100, Math.max(0, gpu.load)) : metrics.gpu;
              metrics.gpuMemory = typeof gpu.memory_percent === 'number' ? Math.min(100, Math.max(0, gpu.memory_percent)) : metrics.gpuMemory;
            }
          }
        }
      }

      setSystemMetrics(metrics);
    } catch (error) {
      console.error('Error processing system metrics:', error);
      setSystemMetrics({ cpu: 0, memory: 0, gpu: 0, gpuMemory: 0 });
    }
  }, [data]);

  const getColorForUsage = (usage) => {
    if (usage >= 90) return theme.palette.error.main;
    if (usage >= 70) return theme.palette.warning.main;
    return theme.palette.success.main;
  };

  const formatUsage = (value) => {
    if (typeof value !== 'number' || isNaN(value)) return '0.0';
    return value.toFixed(1);
  };

  return (
    <Box sx={{ p: 2 }}>
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                CPU Usage
              </Typography>
              <Typography variant="h4" sx={{ color: getColorForUsage(systemMetrics.cpu), mb: 1 }}>
                {formatUsage(systemMetrics.cpu)}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={systemMetrics.cpu}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: getColorForUsage(systemMetrics.cpu)
                  }
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Memory Usage
              </Typography>
              <Typography variant="h4" sx={{ color: getColorForUsage(systemMetrics.memory), mb: 1 }}>
                {formatUsage(systemMetrics.memory)}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={systemMetrics.memory}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: getColorForUsage(systemMetrics.memory)
                  }
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                GPU Usage
              </Typography>
              <Typography variant="h4" sx={{ color: getColorForUsage(systemMetrics.gpu), mb: 1 }}>
                {formatUsage(systemMetrics.gpu)}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={systemMetrics.gpu}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: getColorForUsage(systemMetrics.gpu)
                  }
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                GPU Memory
              </Typography>
              <Typography variant="h4" sx={{ color: getColorForUsage(systemMetrics.gpuMemory), mb: 1 }}>
                {formatUsage(systemMetrics.gpuMemory)}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={systemMetrics.gpuMemory}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: getColorForUsage(systemMetrics.gpuMemory)
                  }
                }}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SystemChart;
