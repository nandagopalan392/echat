import React, { useMemo, useEffect, useRef } from 'react';
import ReactECharts from 'echarts-for-react';

const LearningRateChart = React.memo(({ data, theme }) => {
  const chartRef = useRef(null);

  const baseOptions = useMemo(() => ({
    animation: false,
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    grid: { left: '10%', right: '10%', bottom: 60, top: 10, containLabel: true },
    xAxis: { type: 'value', name: 'Step', axisLabel: { color: theme.palette.text.secondary }, axisLine: { lineStyle: { color: theme.palette.divider } } },
    yAxis: { type: 'value', name: 'Learning Rate', axisLabel: { color: theme.palette.text.secondary }, axisLine: { lineStyle: { color: theme.palette.divider } } },
    series: [
      { name: 'Learning Rate', type: 'line', data: [], smooth: true, showSymbol: false, itemStyle: { color: theme.palette.info.main } }
    ]
  }), [theme]);

  useEffect(() => {
    const inst = chartRef.current?.getEchartsInstance?.();
    if (!inst || !data) return;
    
    const lrData = (data.learning_rates || []).map(d => [d.step, d.value]);
    
    // Only update if we have meaningful data
    if (lrData.length > 0) {
      inst.setOption({ 
        series: [{ data: lrData }] 
      }, { notMerge: false, lazyUpdate: true });
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
