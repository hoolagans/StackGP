import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ReferenceLine,
} from 'recharts';
import {
  getSessionState, getModels, getEnsemble, predictModel, predictEnsemble,
  findMaxUncertaintyPoint, ModelInfo,
} from '../api/client';
import { Card, Button, StatBadge, Badge, Spinner, EmptyState, Input } from '../components/ui';
import toast from 'react-hot-toast';

const PredictPage: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [featureCols, setFeatureCols] = useState<string[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [ensemble, setEnsemble] = useState<{ id: number; expression: string; complexity: number | null }[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<number>(0);
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [predictions, setPredictions] = useState<{ value: number | null; uncertainty?: number | null }[]>([]);
  const [batchText, setBatchText] = useState('');
  const [batchResults, setBatchResults] = useState<{ pred: (number | null)[]; unc: (number | null)[] } | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [maxUncPoint, setMaxUncPoint] = useState<number[] | null>(null);
  const [hasEnsemble, setHasEnsemble] = useState(false);

  useEffect(() => {
    const load = async () => {
      try {
        const s = await getSessionState();
        setFeatureCols(s.data.feature_cols);
        setHasEnsemble(s.data.ensemble_size > 0);

        const initial: Record<string, string> = {};
        s.data.feature_cols.forEach(f => { initial[f] = ''; });
        setInputs(initial);

        const [m, e] = await Promise.all([getModels(50), getEnsemble()]);
        setModels(m.data.models);
        setEnsemble(e.data.ensemble);
        if (m.data.models.length > 0) setSelectedModelId(m.data.models[0].id);
      } catch (e) { console.error('Failed to load prediction page data', e); }
      setLoading(false);
    };
    load();
  }, []);

  const handleSinglePredict = async () => {
    const row = featureCols.map(f => parseFloat(inputs[f] ?? '0'));
    if (row.some(isNaN)) { toast.error('Fill all feature values'); return; }
    setPredicting(true);
    try {
      if (hasEnsemble) {
        const r = await predictEnsemble([row]);
        setPredictions([{ value: r.data.predictions[0], uncertainty: r.data.uncertainty[0] }]);
      } else {
        const r = await predictModel(selectedModelId, [row]);
        setPredictions([{ value: r.data.predictions[0] }]);
      }
    } catch (e: any) {
      toast.error(e.response?.data?.detail ?? 'Prediction failed');
    } finally {
      setPredicting(false);
    }
  };

  const handleBatchPredict = async () => {
    const lines = batchText.trim().split('\n').filter(l => l.trim());
    const rows = lines.map(l => l.split(/[,\t]+/).map(v => parseFloat(v.trim())));
    const bad = rows.findIndex(r => r.some(isNaN) || r.length !== featureCols.length);
    if (bad >= 0) {
      toast.error(`Row ${bad + 1} has invalid values or wrong column count (expected ${featureCols.length})`);
      return;
    }
    setPredicting(true);
    try {
      if (hasEnsemble) {
        const r = await predictEnsemble(rows);
        setBatchResults({ pred: r.data.predictions, unc: r.data.uncertainty });
      } else {
        const r = await predictModel(selectedModelId, rows);
        setBatchResults({ pred: r.data.predictions, unc: r.data.predictions.map(() => null) });
      }
      toast.success(`Predicted ${rows.length} rows`);
    } catch (e: any) {
      toast.error(e.response?.data?.detail ?? 'Batch prediction failed');
    } finally {
      setPredicting(false);
    }
  };

  const handleMaxUncertainty = async () => {
    try {
      const r = await findMaxUncertaintyPoint();
      setMaxUncPoint(r.data.point);
      // Populate inputs
      const newInputs: Record<string, string> = {};
      featureCols.forEach((f, i) => { newInputs[f] = r.data.point[i]?.toFixed(4) ?? ''; });
      setInputs(newInputs);
      toast.success('Maximum uncertainty point found — inputs populated');
    } catch (e: any) {
      toast.error(e.response?.data?.detail ?? 'Failed');
    }
  };

  const downloadCSV = () => {
    if (!batchResults) return;
    const header = [...featureCols, 'prediction', 'uncertainty'].join(',');
    const lines = batchText.trim().split('\n').filter(Boolean).map((l, i) =>
      `${l},${batchResults.pred[i] ?? ''},${batchResults.unc[i] ?? ''}`
    );
    const blob = new Blob([[header, ...lines].join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'stackgp_predictions.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loading) {
    return <div className="flex justify-center items-center h-64"><Spinner /></div>;
  }

  if (models.length === 0) {
    return (
      <div className="p-6">
        <EmptyState icon="⚡" title="No models available" subtitle="Train models first on the Model page" />
        <div className="flex justify-center mt-4">
          <Button onClick={() => navigate('/model')}>Go to Model</Button>
        </div>
      </div>
    );
  }

  const batchChart = batchResults
    ? batchResults.pred.map((p, i) => ({ i: i + 1, pred: p, unc: batchResults.unc[i] }))
    : [];

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Model Predictions</h1>
        <p className="text-sm text-gray-500 mt-1">
          Apply your trained {hasEnsemble ? 'ensemble' : 'model'} to new data points.
          {hasEnsemble && ' Ensemble predictions include uncertainty estimates.'}
        </p>
      </div>

      {/* Model / Ensemble selector */}
      <Card title={hasEnsemble ? 'Ensemble Models' : 'Select Model'}>
        {hasEnsemble ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2 mb-2">
              <Badge text="Ensemble Active" variant="green" />
              <span className="text-xs text-gray-500">{ensemble.length} models</span>
            </div>
            <div className="max-h-40 overflow-y-auto divide-y divide-gray-100 border border-gray-200 rounded-lg">
              {ensemble.map(m => (
                <div key={m.id} className="flex items-center gap-3 px-3 py-2 text-xs">
                  <span className="text-gray-400 font-mono w-5">#{m.id}</span>
                  <span className="font-mono text-gray-700 flex-1 truncate" title={m.expression}>{m.expression}</span>
                  <Badge text={String(m.complexity ?? '?')} variant="gray" />
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div>
            <label className="text-xs font-medium text-gray-600 mb-1 block">Model</label>
            <select
              value={selectedModelId}
              onChange={e => setSelectedModelId(parseInt(e.target.value))}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-400"
            >
              {models.map(m => (
                <option key={m.id} value={m.id}>
                  #{m.id} — {m.expression} (fitness={m.fitness?.toFixed(3)}, cplx={m.complexity})
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-400 mt-1">
              Build an ensemble on the Analysis page for uncertainty estimates.
            </p>
          </div>
        )}
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Single point prediction */}
        <Card title="Single Point Prediction">
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              {featureCols.map(f => (
                <Input
                  key={f}
                  label={f}
                  type="number"
                  placeholder="0.0"
                  value={inputs[f] ?? ''}
                  onChange={e => setInputs(i => ({ ...i, [f]: e.target.value }))}
                />
              ))}
            </div>

            <div className="flex gap-2">
              <Button onClick={handleSinglePredict} loading={predicting} className="flex-1">
                Predict
              </Button>
              {hasEnsemble && (
                <Button variant="secondary" onClick={handleMaxUncertainty} title="Find point of maximum model uncertainty">
                  🎯 Max Uncertainty
                </Button>
              )}
            </div>

            {predictions.length > 0 && (
              <div className="p-4 bg-brand-50 border border-brand-200 rounded-xl space-y-2">
                <div className="text-xs font-medium text-brand-700 uppercase tracking-wide">Result</div>
                <div className="text-3xl font-bold text-brand-800">
                  {predictions[0].value?.toFixed(6) ?? 'N/A'}
                </div>
                {predictions[0].uncertainty != null && (
                  <div className="text-sm text-brand-600">
                    ± {predictions[0].uncertainty.toFixed(6)} uncertainty
                  </div>
                )}
              </div>
            )}

            {maxUncPoint && (
              <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-xs text-yellow-700">
                <strong>Max uncertainty point:</strong>{' '}
                {featureCols.map((f, i) => `${f}=${maxUncPoint[i]?.toFixed(4)}`).join(', ')}
              </div>
            )}
          </div>
        </Card>

        {/* Batch prediction */}
        <Card title="Batch Prediction">
          <div className="space-y-3">
            <div>
              <label className="text-xs font-medium text-gray-600 mb-1 block">
                CSV / tab-separated rows ({featureCols.join(', ')})
              </label>
              <textarea
                className="w-full h-36 rounded-lg border border-gray-300 px-3 py-2 text-xs font-mono focus:outline-none focus:ring-2 focus:ring-brand-400 resize-none"
                placeholder={`${featureCols.map(() => '0.0').join(',')}\n${featureCols.map(() => '1.0').join(',')}`}
                value={batchText}
                onChange={e => setBatchText(e.target.value)}
              />
            </div>
            <div className="flex gap-2">
              <Button onClick={handleBatchPredict} loading={predicting} className="flex-1">
                Predict Batch
              </Button>
              {batchResults && (
                <Button variant="secondary" onClick={downloadCSV}>
                  ⬇ Download CSV
                </Button>
              )}
            </div>
          </div>
        </Card>
      </div>

      {/* Batch results */}
      {batchResults && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card title="Batch Predictions Table">
            <div className="overflow-auto max-h-64">
              <table className="min-w-full text-xs">
                <thead className="bg-gray-50 sticky top-0">
                  <tr>
                    <th className="px-3 py-2 text-left border-b border-gray-200">#</th>
                    <th className="px-3 py-2 text-right border-b border-gray-200">Prediction</th>
                    {hasEnsemble && <th className="px-3 py-2 text-right border-b border-gray-200">Uncertainty</th>}
                  </tr>
                </thead>
                <tbody>
                  {batchResults.pred.map((p, i) => (
                    <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                      <td className="px-3 py-1.5 text-gray-500">{i + 1}</td>
                      <td className="px-3 py-1.5 text-right font-mono text-gray-800">
                        {p != null ? p.toFixed(6) : <span className="text-red-400">NaN</span>}
                      </td>
                      {hasEnsemble && (
                        <td className="px-3 py-1.5 text-right font-mono text-yellow-600">
                          {batchResults.unc[i] != null ? `±${batchResults.unc[i]!.toFixed(4)}` : '—'}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Prediction Chart">
            <ResponsiveContainer width="100%" height={230}>
              <BarChart data={batchChart}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="i" tick={{ fontSize: 10 }} label={{ value: 'Sample', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip formatter={(v: number, name: string) => [v?.toFixed(4), name === 'pred' ? 'Prediction' : 'Uncertainty']} />
                <Bar dataKey="pred" fill="#3b82f6" opacity={0.85} name="Prediction" />
                {hasEnsemble && <Bar dataKey="unc" fill="#f59e0b" opacity={0.7} name="Uncertainty" />}
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}
    </div>
  );
};

export default PredictPage;
