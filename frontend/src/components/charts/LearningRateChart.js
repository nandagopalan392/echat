import React, { useMemo, useEffect, useRef } from 'react';
import ReactECharts from 'echarts-for-react';

const LearningRateChart = React.memo(({ data, theme }) => {
  const chartRef = useRef(null);

  const baseOptions = useMemo(() => ({
    animation: false,
    tooltip: { 
      trigger: 'axis', 
      axisPointer: { type: 'cross' },
      formatter: (params) => {
        if (!params || params.length === 0) return '';
        const param = params[0];
        const value = typeof param.value[1] === 'number' ? param.value[1].toExponential(3) : 'N/A';
        return `<div style="font-size: 14px;">Step: ${param.value[0]}</div>
                <div style="color: ${param.color};">Learning Rate: ${value}</div>`;
      }
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
      type: 'log', 
      name: 'Learning Rate',
      nameLocation: 'middle',
      nameGap: 60,
      axisLabel: { 
        color: theme.palette.text.secondary,
        formatter: (value) => value.toExponential(1)
      }, 
      axisLine: { lineStyle: { color: theme.palette.divider } },
      splitLine: { show: true, lineStyle: { color: theme.palette.divider, type: 'dashed' } }
    },
    series: [
      { 
        name: 'Learning Rate', 
        type: 'line', 
        data: [], 
        smooth: true, 
        showSymbol: false, 
        itemStyle: { color: theme.palette.info.main },
        lineStyle: { width: 2 }
      }
    ]
  }), [theme]);

  useEffect(() => {
    const inst = chartRef.current?.getEchartsInstance?.();
    if (!inst || !data) return;
    
    try {
      let lrData = [];
      
      // Handle different data structures
      if (data.learning_rates && Array.isArray(data.learning_rates)) {
        lrData = data.learning_rates
          .filter(d => d && typeof d.step === 'number' && typeof d.value === 'number' && d.value > 0)
          .map(d => [d.step, d.value]);
      }
      
      // Alternative data format
      if (data.learning_rate && Array.isArray(data.learning_rate)) {
        lrData = data.learning_rate
          .filter(d => d && typeof d.step === 'number' && typeof d.lr === 'number' && d.lr > 0)
          .map(d => [d.step, d.lr]);
      }
      
      // Handle simple array of learning rates
      if (data.lr_schedule && Array.isArray(data.lr_schedule)) {
        lrData = data.lr_schedule
          .filter(lr => typeof lr === 'number' && lr > 0)
          .map((lr, index) => [index + 1, lr]);
      }
      
      // Only update if we have meaningful data
      if (lrData.length > 0) {
        inst.setOption({ 
          series: [{ data: lrData }] 
        }, { notMerge: false, lazyUpdate: true });
      }
    } catch (error) {
      console.error('Error updating LearningRateChart:', error);
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
  
  const prevLR = prevProps.data.learning_rates || [];
  const nextLR = nextProps.data.learning_rates || [];
  
  return prevLR.length === nextLR.length;
});

export default LearningRateChart;
