import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ChevronRight } from 'lucide-react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, ReferenceLine,
} from 'recharts';
import {
  getModels, getPareto, getResiduals, getVariableImportance, buildEnsemble,
  getSessionState, ModelInfo, ParetoPoint, getApiError,
} from '../api/client';
import { Card, Button, StatBadge, Badge, Spinner, EmptyState } from '../components/ui';
import toast from 'react-hot-toast';

const AnalysisPage: React.FC = () => {
  const navigate = useNavigate();
  const [fetched, setFetched] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [pareto, setPareto] = useState<ParetoPoint[]>([]);
  const [importance, setImportance] = useState<Record<string, number>>({});
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [residuals, setResiduals] = useState<{
    actual: (number | null)[];
    predicted: (number | null)[];
    residuals: (number | null)[];
    train_rmse: number;
    test_rmse: number | null;
    train_fitness: number;
    test_fitness: number | null;
  } | null>(null);
  const [buildingEnsemble, setBuildingEnsemble] = useState(false);
  const [ensembleSize, setEnsembleSize] = useState(10);
  const [ensembleBuilt, setEnsembleBuilt] = useState(false);
  const [modelCount, setModelCount] = useState<number | null>(null);

  // Derive loading: we are loading when modelCount is known, positive, but data isn't fetched yet.
  const loading = modelCount !== null && modelCount > 0 && !fetched;

  useEffect(() => {
    getSessionState().then(r => {
      setModelCount(r.data.model_count);
      setEnsembleBuilt(r.data.ensemble_size > 0);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (modelCount === null || modelCount === 0) return;
    Promise.all([getModels(100), getPareto(), getVariableImportance()])
      .then(([m, p, vi]) => {
        setModels(m.data.models);
        setPareto(p.data.pareto);
        setImportance(vi.data.importance);
        if (m.data.models.length > 0) setSelectedId(m.data.models[0].id);
      })
      .catch(() => toast.error('Failed to load models'))
      .finally(() => setFetched(true));
  }, [modelCount]);

  useEffect(() => {
    if (selectedId == null) return;
    getResiduals(selectedId).then(r => setResiduals(r.data)).catch(e => { console.error('Failed to load residuals', e); });
  }, [selectedId]);

  const handleBuildEnsemble = async () => {
    setBuildingEnsemble(true);
    try {
      const r = await buildEnsemble(ensembleSize);
      setEnsembleBuilt(true);
      toast.success(`Ensemble built with ${r.data.ensemble_size} models`);
    } catch (e) {
      toast.error(getApiError(e, 'Failed'));
    } finally {
      setBuildingEnsemble(false);
    }
  };

  if (modelCount === null || loading) {
    return <div className="flex justify-center items-center h-64"><Spinner /></div>;
  }

  if (modelCount === 0) {
    return (
      <div className="p-6">
        <EmptyState icon="📊" title="No models yet" subtitle="Train models first on the Model page" />
        <div className="flex justify-center mt-4">
          <Button onClick={() => navigate('/model')}>Go to Model</Button>
        </div>
      </div>
    );
  }

  const selectedModel = models.find(m => m.id === selectedId);

  // Pareto chart data
  const paretoChart = pareto.map(p => ({
    complexity: p.complexity,
    fitness: p.fitness != null ? parseFloat(p.fitness.toFixed(4)) : null,
    id: p.id,
  }));

  // Residual chart
  const residualChart = residuals
    ? residuals.actual.slice(0, 500).map((a, i) => ({
        actual: a,
        predicted: residuals.predicted[i],
        residual: residuals.residuals[i],
      })).filter(d => d.actual != null && d.predicted != null)
    : [];

  // Variable importance chart
  const impChart = Object.entries(importance)
    .sort(([, a], [, b]) => b - a)
    .map(([col, val]) => ({ col, val: parseFloat((val * 100).toFixed(1)) }));

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Model Analysis</h1>
          <p className="text-sm text-gray-500 mt-1">
            Explore the Pareto front, inspect expressions, analyze residuals, and build an ensemble.
          </p>
        </div>
        {ensembleBuilt && (
          <Button onClick={() => navigate('/predict')}>
            Predictions <ChevronRight size={16} />
          </Button>
        )}
      </div>

      {/* Pareto front */}
      <Card title="Pareto Front: Accuracy vs Complexity">
        <ResponsiveContainer width="100%" height={260}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="complexity" name="Complexity" tick={{ fontSize: 10 }}
              label={{ value: 'Complexity', position: 'insideBottom', offset: -3, fontSize: 11 }} />
            <YAxis dataKey="fitness" name="Fitness" domain={[0, 1]} tick={{ fontSize: 10 }}
              label={{ value: 'Fitness (1−R²)', angle: -90, position: 'insideLeft', fontSize: 11 }} />
            <Tooltip content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0].payload;
              const m = pareto.find(p => p.id === d.id);
              return (
                <div className="bg-white border border-gray-200 rounded-lg shadow p-3 text-xs max-w-xs">
                  <div className="font-semibold mb-1">Model #{d.id}</div>
                  <div>Fitness: {d.fitness}</div>
                  <div>Complexity: {d.complexity}</div>
                  <div className="mt-1 font-mono text-gray-600 break-all">{m?.expression}</div>
                </div>
              );
            }} />
            <Scatter
              data={paretoChart.map(d => ({ ...d, selected: d.id === selectedId }))}
              fill="#3b82f6"
              opacity={0.7}
              onClick={(d: unknown) => {
                if (typeof d === 'object' && d !== null && 'id' in d && typeof (d as { id: unknown }).id === 'number') {
                  setSelectedId((d as { id: number }).id);
                }
              }}
              cursor="pointer"
            />
          </ScatterChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-400 mt-1">Click a point to inspect that model below.</p>
      </Card>

      {/* Model table */}
      <Card title={`All Models (${models.length})`}>
        <div className="overflow-auto max-h-72">
          <table className="min-w-full text-xs">
            <thead className="bg-gray-50 sticky top-0">
              <tr>
                {['#', 'Expression', 'Fitness', 'RMSE', 'Complexity', ''].map(h => (
                  <th key={h} className="px-3 py-2 text-left font-semibold text-gray-600 border-b border-gray-200 whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map(m => (
                <tr
                  key={m.id}
                  onClick={() => setSelectedId(m.id)}
                  className={`cursor-pointer border-b border-gray-100 hover:bg-brand-50 transition-colors
                    ${m.id === selectedId ? 'bg-brand-100' : m.id % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}`}
                >
                  <td className="px-3 py-1.5 text-gray-500 font-mono">{m.id}</td>
                  <td className="px-3 py-1.5 font-mono text-gray-800 max-w-xs truncate" title={m.expression}>
                    {m.expression}
                  </td>
                  <td className="px-3 py-1.5 text-right">
                    <span className={`font-semibold ${(m.fitness ?? 1) < 0.1 ? 'text-green-600' : (m.fitness ?? 1) < 0.3 ? 'text-yellow-600' : 'text-red-500'}`}>
                      {m.fitness?.toFixed(4) ?? '—'}
                    </span>
                  </td>
                  <td className="px-3 py-1.5 text-right text-gray-600">{m.rmse?.toFixed(4) ?? '—'}</td>
                  <td className="px-3 py-1.5 text-right">
                    <Badge text={String(m.complexity ?? '?')} variant={
                      (m.complexity ?? 999) <= 20 ? 'green' : (m.complexity ?? 999) <= 50 ? 'yellow' : 'red'
                    } />
                  </td>
                  <td className="px-3 py-1.5">
                    {m.id === selectedId && <span className="text-brand-500">◀ selected</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Selected model detail */}
      {selectedModel && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card title={`Model #${selectedModel.id} — Expression`}>
            <div className="font-mono text-sm bg-gray-50 border border-gray-200 rounded-lg p-4 break-all leading-relaxed">
              {selectedModel.expression}
            </div>
            <div className="grid grid-cols-2 gap-3 mt-4">
              <StatBadge label="Fitness" value={selectedModel.fitness?.toFixed(4) ?? '—'} color="brand" />
              <StatBadge label="RMSE" value={selectedModel.rmse?.toFixed(4) ?? '—'} color="brand" />
              <StatBadge label="Complexity" value={selectedModel.complexity ?? '—'} color="brand" />
            </div>
            {residuals && (
              <div className="grid grid-cols-2 gap-3 mt-3">
                <StatBadge label="Train RMSE" value={residuals.train_rmse?.toFixed(4)} color="brand" />
                <StatBadge label="Test RMSE" value={residuals.test_rmse?.toFixed(4) ?? '—'} color="brand" />
                <StatBadge label="Train R" value={residuals.train_fitness?.toFixed(4)} color="brand" />
                <StatBadge label="Test R" value={residuals.test_fitness?.toFixed(4) ?? '—'} color="brand" />
              </div>
            )}
          </Card>

          {/* Prediction vs actual */}
          <Card title="Predicted vs Actual">
            {residualChart.length > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="actual" name="Actual" tick={{ fontSize: 10 }}
                    label={{ value: 'Actual', position: 'insideBottom', offset: -3, fontSize: 11 }} />
                  <YAxis dataKey="predicted" name="Predicted" tick={{ fontSize: 10 }}
                    label={{ value: 'Predicted', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                  <Tooltip />
                  <Scatter data={residualChart} fill="#3b82f6" opacity={0.5} r={3} />
                </ScatterChart>
              </ResponsiveContainer>
            ) : <div className="text-sm text-gray-400 italic py-8 text-center">Loading…</div>}
          </Card>

          {/* Residual plot */}
          {residualChart.length > 0 && (
            <Card title="Residuals">
              <ResponsiveContainer width="100%" height={200}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="actual" name="Actual" tick={{ fontSize: 10 }}
                    label={{ value: 'Actual', position: 'insideBottom', offset: -3, fontSize: 11 }} />
                  <YAxis dataKey="residual" name="Residual" tick={{ fontSize: 10 }}
                    label={{ value: 'Residual', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                  <ReferenceLine y={0} stroke="#ef4444" strokeDasharray="4 4" />
                  <Tooltip />
                  <Scatter data={residualChart} fill="#f59e0b" opacity={0.6} r={3} />
                </ScatterChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Variable importance */}
          {impChart.length > 0 && (
            <Card title="Variable Importance (avg operator usage)">
              <ResponsiveContainer width="100%" height={Math.max(160, impChart.length * 28)}>
                <BarChart data={impChart} layout="vertical" margin={{ left: 80 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 10 }} unit="%" />
                  <YAxis type="category" dataKey="col" tick={{ fontSize: 10 }} width={80} />
                  <Tooltip formatter={(v: number) => [`${v}%`, 'Importance']} />
                  <Bar dataKey="val" radius={4}>
                    {impChart.map((_, i) => (
                      <Cell key={i} fill={`hsl(${210 + i * 20}, 70%, 55%)`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}
        </div>
      )}

      {/* Ensemble builder */}
      <Card title="Ensemble Builder">
        <div className="flex items-end gap-4">
          <div>
            <label className="text-xs font-medium text-gray-600 mb-1 block">Number of clusters</label>
            <input
              type="number"
              min={2}
              max={50}
              value={ensembleSize}
              onChange={e => setEnsembleSize(parseInt(e.target.value))}
              className="w-24 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-400"
            />
          </div>
          <Button onClick={handleBuildEnsemble} loading={buildingEnsemble}>
            Build Ensemble
          </Button>
          {ensembleBuilt && (
            <span className="text-sm text-green-600 font-medium">✓ Ensemble ready</span>
          )}
        </div>
        <p className="text-xs text-gray-400 mt-2">
          Selects a diverse set of models using data-space clustering. Used for uncertainty estimation in predictions.
        </p>
      </Card>
    </div>
  );
};

export default AnalysisPage;
