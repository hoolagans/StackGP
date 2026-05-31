import React, { useEffect, useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, Square, ChevronRight } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import {
  getSessionState, startTraining, stopTraining, getTrainStatus,
  TrainConfig,
} from '../api/client';
import { Card, Button, Select, Input, StatBadge, EmptyState } from '../components/ui';
import toast from 'react-hot-toast';

const DEFAULT_CFG: TrainConfig = {
  generations: 100,
  pop_size: 300,
  max_complexity: 100,
  max_length: 10,
  mutation_rate: 79,
  crossover_rate: 11,
  spawn_rate: 10,
  time_limit: 300,
  ops_set: 'default',
  align: true,
  data_subsample: false,
  allow_early_termination: false,
  early_termination_threshold: 0.0,
};

const ModelPage: React.FC = () => {
  const navigate = useNavigate();
  const [hasData, setHasData] = useState(false);
  const [cfg, setCfg] = useState<TrainConfig>(DEFAULT_CFG);
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [total, setTotal] = useState(0);
  const [modelCount, setModelCount] = useState(0);
  const [log, setLog] = useState<{ gen?: number; fitness?: number; complexity?: number; error?: string }[]>([]);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    getSessionState().then(r => setHasData(r.data.has_processed_data)).catch(() => {});
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const startPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const r = await getTrainStatus();
        setStatus(r.data.status);
        setProgress(r.data.progress);
        setTotal(r.data.total);
        setLog(r.data.log);
        setModelCount(r.data.model_count);
        if (r.data.status === 'done' || r.data.status === 'error' || r.data.status === 'idle') {
          if (pollRef.current) clearInterval(pollRef.current);
          if (r.data.status === 'done') toast.success(`Training complete — ${r.data.model_count} models found`);
          if (r.data.status === 'error') toast.error('Training error');
        }
      } catch {}
    }, 1000);
  }, []);

  const handleStart = async () => {
    const rateSum = cfg.mutation_rate + cfg.crossover_rate + cfg.spawn_rate;
    if (rateSum !== 100) {
      toast.error(`Rates must sum to 100 (currently ${rateSum})`);
      return;
    }
    try {
      await startTraining(cfg);
      setStatus('running');
      setLog([]);
      setProgress(0);
      startPolling();
      toast('Training started…');
    } catch (e: any) {
      toast.error(e.response?.data?.detail ?? 'Failed to start');
    }
  };

  const handleStop = async () => {
    await stopTraining();
    if (pollRef.current) clearInterval(pollRef.current);
    setStatus('idle');
    toast('Training stopped');
  };

  const isRunning = status === 'running' || status === 'starting';
  const pct = total > 0 ? Math.round((progress / total) * 100) : 0;

  const chartData = log.filter(l => l.gen !== undefined).map(l => ({
    gen: l.gen,
    fitness: l.fitness != null ? parseFloat(l.fitness.toFixed(4)) : null,
    complexity: l.complexity,
  }));

  const updateCfg = (k: keyof TrainConfig, v: any) => setCfg(c => ({ ...c, [k]: v }));
  const numericInput = (label: string, key: keyof TrainConfig, min?: number, max?: number, step?: number) => (
    <Input
      label={label}
      type="number"
      min={min}
      max={max}
      step={step ?? 1}
      value={cfg[key] as number}
      onChange={e => updateCfg(key, parseFloat(e.target.value))}
    />
  );

  if (!hasData) {
    return (
      <div className="p-6">
        <EmptyState icon="🧬" title="No processed data" subtitle="Process your data first" />
        <div className="flex justify-center mt-4">
          <Button onClick={() => navigate('/process')}>Go to Process</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Symbolic Regression Modeling</h1>
          <p className="text-sm text-gray-500 mt-1">Configure and run the StackGP genetic programming evolution.</p>
        </div>
        {status === 'done' && modelCount > 0 && (
          <Button onClick={() => navigate('/analysis')}>
            View Analysis <ChevronRight size={16} />
          </Button>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Config panels */}
        <div className="lg:col-span-2 space-y-4">
          <Card title="Evolution Parameters">
            <div className="grid grid-cols-2 gap-4">
              {numericInput('Generations', 'generations', 10, 5000)}
              {numericInput('Population size', 'pop_size', 10, 2000)}
              {numericInput('Max complexity', 'max_complexity', 5, 500)}
              {numericInput('Max program length', 'max_length', 2, 50)}
              {numericInput('Time limit (s)', 'time_limit', 10, 3600)}
              <Select
                label="Operator set"
                value={cfg.ops_set}
                onChange={e => updateCfg('ops_set', e.target.value)}
              >
                <option value="default">Default (arithmetic)</option>
                <option value="all">All (incl. trig/log)</option>
                <option value="boolean">Boolean logic</option>
              </Select>
            </div>
          </Card>

          <Card title="Genetic Operators (must sum to 100)">
            <div className="grid grid-cols-3 gap-4">
              {numericInput('Mutation rate %', 'mutation_rate', 0, 100)}
              {numericInput('Crossover rate %', 'crossover_rate', 0, 100)}
              {numericInput('Spawn rate %', 'spawn_rate', 0, 100)}
            </div>
            {(() => {
              const s = cfg.mutation_rate + cfg.crossover_rate + cfg.spawn_rate;
              return s !== 100 ? (
                <p className="text-xs text-red-500 mt-2">⚠ Rates sum to {s}, must equal 100</p>
              ) : (
                <p className="text-xs text-green-600 mt-2">✓ Rates sum to 100</p>
              );
            })()}
          </Card>

          <Card title="Advanced Options">
            <div className="grid grid-cols-2 gap-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={cfg.align}
                  onChange={e => updateCfg('align', e.target.checked)}
                  className="rounded border-gray-300 text-brand-600"
                />
                <span className="text-sm text-gray-700">Align constants (recommended)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={cfg.data_subsample}
                  onChange={e => updateCfg('data_subsample', e.target.checked)}
                  className="rounded border-gray-300 text-brand-600"
                />
                <span className="text-sm text-gray-700">Data subsampling</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={cfg.allow_early_termination}
                  onChange={e => updateCfg('allow_early_termination', e.target.checked)}
                  className="rounded border-gray-300 text-brand-600"
                />
                <span className="text-sm text-gray-700">Allow early termination</span>
              </label>
              {cfg.allow_early_termination && (
                <Input
                  label="Early termination threshold"
                  type="number"
                  step={0.001}
                  value={cfg.early_termination_threshold}
                  onChange={e => updateCfg('early_termination_threshold', parseFloat(e.target.value))}
                />
              )}
            </div>
          </Card>
        </div>

        {/* Status panel */}
        <div className="space-y-4">
          <Card title="Training Control">
            <div className="space-y-4">
              {isRunning ? (
                <Button variant="danger" size="lg" className="w-full" onClick={handleStop} icon={<Square size={16} />}>
                  Stop Training
                </Button>
              ) : (
                <Button size="lg" className="w-full" onClick={handleStart} icon={<Play size={16} />}>
                  Start Training
                </Button>
              )}

              {/* Progress */}
              <div>
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span>Generation {progress} / {total || cfg.generations}</span>
                  <span>{pct}%</span>
                </div>
                <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${isRunning ? 'bg-brand-500 animate-pulse' : 'bg-brand-600'}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>

              {/* Status badges */}
              <div className="grid grid-cols-2 gap-2">
                <StatBadge label="Status" value={status} color="brand" />
                <StatBadge label="Models found" value={modelCount} color="brand" />
              </div>

              {/* Latest best */}
              {log.length > 0 && (() => {
                const last = [...log].reverse().find(l => l.fitness !== undefined);
                return last ? (
                  <div className="grid grid-cols-2 gap-2">
                    <StatBadge label="Best fitness" value={last.fitness != null ? last.fitness.toFixed(4) : '—'} color="brand" />
                    <StatBadge label="Complexity" value={last.complexity ?? '—'} color="brand" />
                  </div>
                ) : null;
              })()}
            </div>
          </Card>

          {/* Log */}
          <Card title="Training Log">
            <div className="font-mono text-[10px] bg-gray-900 text-green-400 rounded-lg p-3 h-48 overflow-y-auto space-y-0.5">
              {log.length === 0 ? (
                <div className="text-gray-500">Waiting for training to start…</div>
              ) : (
                [...log].reverse().slice(0, 50).map((entry, i) =>
                  entry.error ? (
                    <div key={i} className="text-red-400">ERROR: {entry.error}</div>
                  ) : (
                    <div key={i} className="text-green-400">
                      Gen {entry.gen}: fitness={entry.fitness?.toFixed(4) ?? '?'} complexity={entry.complexity ?? '?'}
                    </div>
                  )
                )
              )}
            </div>
          </Card>
        </div>
      </div>

      {/* Live chart */}
      {chartData.length > 1 && (
        <Card title="Training Progress">
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="gen" tick={{ fontSize: 10 }} label={{ value: 'Generation', position: 'insideBottom', offset: -3, fontSize: 11 }} />
              <YAxis yAxisId="fit" domain={[0, 1]} tick={{ fontSize: 10 }} label={{ value: 'Fitness', angle: -90, position: 'insideLeft', fontSize: 11 }} />
              <YAxis yAxisId="cplx" orientation="right" tick={{ fontSize: 10 }} label={{ value: 'Complexity', angle: 90, position: 'insideRight', fontSize: 11 }} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line yAxisId="fit" type="monotone" dataKey="fitness" stroke="#3b82f6" dot={false} name="Fitness (corr)" strokeWidth={2} />
              <Line yAxisId="cplx" type="monotone" dataKey="complexity" stroke="#f59e0b" dot={false} name="Complexity" strokeWidth={1.5} strokeDasharray="4 4" />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      )}
    </div>
  );
};

export default ModelPage;
