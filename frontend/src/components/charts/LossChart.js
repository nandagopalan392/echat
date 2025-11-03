import React, { useMemo, useEffect, useRef } from 'react';
import ReactECharts from 'echarts-for-react';

const LossChart = React.memo(({ data, theme }) => {
  const chartRef = useRef(null);

  const baseOptions = useMemo(() => ({
    animation: false,
    tooltip: { 
      trigger: 'axis', 
      axisPointer: { type: 'cross' },
      formatter: (params) => {
        let tooltip = `<div style="font-size: 14px;">Step: ${params[0].value[0]}</div>`;
        params.forEach(param => {
          const value = typeof param.value[1] === 'number' ? param.value[1].toFixed(6) : 'N/A';
          tooltip += `<div style="color: ${param.color};">${param.seriesName}: ${value}</div>`;
        });
        return tooltip;
      }
    },
    legend: { 
      data: ['Training Loss', 'Validation Loss'], 
      bottom: 6, 
      textStyle: { color: theme.palette.text.primary } 
    },
    grid: { 
      left: '12%', 
      right: '10%', 
      bottom: 80, 
      top: 40, 
      containLabel: true 
    },
    xAxis: { 
      type: 'value', 
      name: 'Step', 
      nameLocation: 'middle',
      nameGap: 30,
      axisLabel: { 
        color: theme.palette.text.secondary,
        formatter: (value) => value.toLocaleString()
      }, 
      axisLine: { lineStyle: { color: theme.palette.divider } },
      splitLine: { show: true, lineStyle: { color: theme.palette.divider, type: 'dashed' } }
    },
    yAxis: { 
      type: 'value', 
      name: 'Loss', 
      nameLocation: 'middle',
      nameGap: 50,
      axisLabel: { 
        color: theme.palette.text.secondary,
        formatter: (value) => value.toFixed(4)
      }, 
      axisLine: { lineStyle: { color: theme.palette.divider } },
      splitLine: { show: true, lineStyle: { color: theme.palette.divider, type: 'dashed' } },
      scale: true
    },
    series: [
      { 
        name: 'Training Loss', 
        type: 'line', 
        data: [], 
        smooth: true, 
        showSymbol: false, 
        itemStyle: { color: theme.palette.primary.main },
        lineStyle: { width: 2 }
      },
      { 
        name: 'Validation Loss', 
        type: 'line', 
        data: [], 
        smooth: true, 
        showSymbol: false, 
        itemStyle: { color: theme.palette.secondary.main },
        lineStyle: { width: 2 }
      }
    ]
  }), [theme]);

  useEffect(() => {
    const inst = chartRef.current?.getEchartsInstance?.();
    if (!inst || !data) return;
    
    try {
      let trainLossData = [];
      let evalLossData = [];
      
      // Handle different data structures
      if (data.train_losses && Array.isArray(data.train_losses)) {
        trainLossData = data.train_losses
          .filter(d => d && typeof d.step === 'number' && typeof d.value === 'number')
          .map(d => [d.step, d.value]);
      }
      
      if (data.eval_losses && Array.isArray(data.eval_losses)) {
        evalLossData = data.eval_losses
          .filter(d => d && typeof d.step === 'number' && typeof d.value === 'number')
          .map(d => [d.step, d.value]);
      }
      
      // Also check for alternative data formats
      if (data.training_loss && Array.isArray(data.training_loss)) {
        trainLossData = data.training_loss
          .filter(d => d && typeof d.step === 'number' && typeof d.loss === 'number')
          .map(d => [d.step, d.loss]);
      }
      
      if (data.validation_loss && Array.isArray(data.validation_loss)) {
        evalLossData = data.validation_loss
          .filter(d => d && typeof d.step === 'number' && typeof d.loss === 'number')
          .map(d => [d.step, d.loss]);
      }
      
      // Handle simple arrays of loss values
      if (data.losses && Array.isArray(data.losses)) {
        trainLossData = data.losses
          .filter(loss => typeof loss === 'number')
          .map((loss, index) => [index + 1, loss]);
      }
      
      // Only update if we have meaningful data
      if (trainLossData.length > 0 || evalLossData.length > 0) {
        inst.setOption({ 
          series: [
            { data: trainLossData }, 
            { data: evalLossData }
          ] 
        }, { notMerge: false, lazyUpdate: true });
      }
    } catch (error) {
      console.error('Error updating LossChart:', error);
    }
  }, [data]);

  return (
    <ReactECharts 
      ref={chartRef} 
      option={baseOptions} 
      notMerge={false} 
      lazyUpdate 
      style={{ height: '100%' }} 
    />
  );
}, (prevProps, nextProps) => {
  // Custom comparison to prevent unnecessary re-renders
  if (prevProps.theme !== nextProps.theme) return false;
  if (!prevProps.data && !nextProps.data) return true;
  if (!prevProps.data || !nextProps.data) return false;
  
  const prevTrain = prevProps.data.train_losses || [];
  const nextTrain = nextProps.data.train_losses || [];
  const prevEval = prevProps.data.eval_losses || [];
  const nextEval = nextProps.data.eval_losses || [];
  
  return prevTrain.length === nextTrain.length && prevEval.length === nextEval.length;
});

export default LossChart;
