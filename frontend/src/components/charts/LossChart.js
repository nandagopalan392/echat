import React, { useMemo, useEffect, useRef } from 'react';
import ReactECharts from 'echarts-for-react';

const LossChart = React.memo(({ data, theme }) => {
  const chartRef = useRef(null);

  const baseOptions = useMemo(() => ({
    animation: false,
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    legend: { data: ['Training Loss', 'Validation Loss'], bottom: 6, textStyle: { color: theme.palette.text.primary } },
    grid: { left: '10%', right: '10%', bottom: 60, top: 10, containLabel: true },
    xAxis: { type: 'value', name: 'Step', axisLabel: { color: theme.palette.text.secondary }, axisLine: { lineStyle: { color: theme.palette.divider } } },
    yAxis: { type: 'value', name: 'Loss', axisLabel: { color: theme.palette.text.secondary }, axisLine: { lineStyle: { color: theme.palette.divider } } },
    series: [
      { name: 'Training Loss', type: 'line', data: [], smooth: true, showSymbol: false, itemStyle: { color: theme.palette.primary.main } },
      { name: 'Validation Loss', type: 'line', data: [], smooth: true, showSymbol: false, itemStyle: { color: theme.palette.secondary.main } }
    ]
  }), [theme]);

  useEffect(() => {
    const inst = chartRef.current?.getEchartsInstance?.();
    if (!inst || !data) return;
    
    const train = (data.train_losses || []).map(d => [d.step, d.value]);
    const evals = (data.eval_losses || []).map(d => [d.step, d.value]);
    
    // Only update if we have meaningful data
    if (train.length > 0 || evals.length > 0) {
      inst.setOption({ 
        series: [
          { data: train }, 
          { data: evals }
        ] 
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
  
  const prevTrain = prevProps.data.train_losses || [];
  const nextTrain = nextProps.data.train_losses || [];
  const prevEval = prevProps.data.eval_losses || [];
  const nextEval = nextProps.data.eval_losses || [];
  
  return prevTrain.length === nextTrain.length && prevEval.length === nextEval.length;
});

export default LossChart;
