import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ChevronRight } from 'lucide-react';
import { processData, getSessionState, ProcessConfig, DataTable as DataTableType } from '../api/client';
import { Card, Button, Select, Input, StatBadge, EmptyState } from '../components/ui';
import DataTable from '../components/DataTable';
import toast from 'react-hot-toast';

const ProcessPage: React.FC = () => {
  const navigate = useNavigate();
  const [hasData, setHasData] = useState(false);
  const [targetCol, setTargetCol] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    total_rows: number; train_rows: number; test_rows: number;
    data: DataTableType;
  } | null>(null);

  const [cfg, setCfg] = useState<ProcessConfig>({
    fill_missing: 'drop',
    normalize: 'none',
    train_split: 0.8,
    random_seed: 42,
  });

  useEffect(() => {
    getSessionState().then(r => {
      setHasData(r.data.has_raw_data);
      setTargetCol(r.data.target_col ?? '');
    }).catch(() => {});
  }, []);

  const handleProcess = async () => {
    setLoading(true);
    try {
      const res = await processData(cfg);
      setResult(res.data);
      toast.success(`Processed: ${res.data.train_rows} train / ${res.data.test_rows} test`);
    } catch (e: any) {
      toast.error(e.response?.data?.detail ?? 'Processing failed');
    } finally {
      setLoading(false);
    }
  };

  if (!hasData) {
    return (
      <div className="p-6">
        <EmptyState icon="⚙️" title="No data loaded" subtitle="Go to Import to upload a dataset first" />
        <div className="flex justify-center mt-4">
          <Button onClick={() => navigate('/import')}>Go to Import</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Process Data</h1>
        <p className="text-sm text-gray-500 mt-1">
          Handle missing values, normalize features, and split into train/test sets.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Missing Value Handling">
          <Select
            label="Strategy"
            value={cfg.fill_missing}
            onChange={e => setCfg(c => ({ ...c, fill_missing: e.target.value as any }))}
          >
            <option value="drop">Drop rows with missing values</option>
            <option value="mean">Fill with column mean</option>
            <option value="median">Fill with column median</option>
            <option value="zero">Fill with zero</option>
          </Select>
          <p className="text-xs text-gray-400 mt-2">
            "Drop" is safest for symbolic regression. Fill strategies impute values and preserve sample size.
          </p>
        </Card>

        <Card title="Feature Normalization">
          <Select
            label="Method"
            value={cfg.normalize}
            onChange={e => setCfg(c => ({ ...c, normalize: e.target.value as any }))}
          >
            <option value="none">No normalization</option>
            <option value="minmax">Min-Max scaling [0, 1]</option>
            <option value="zscore">Z-score standardization (μ=0, σ=1)</option>
          </Select>
          <p className="text-xs text-gray-400 mt-2">
            Normalization applied to features only. Target <strong>{targetCol || 'Y'}</strong> is left unchanged.
          </p>
        </Card>

        <Card title="Train / Test Split">
          <div className="space-y-4">
            <div>
              <label className="text-xs font-medium text-gray-600 mb-2 block">
                Training set size: <strong>{Math.round(cfg.train_split * 100)}%</strong>
              </label>
              <input
                type="range" min={50} max={95} step={5}
                value={Math.round(cfg.train_split * 100)}
                onChange={e => setCfg(c => ({ ...c, train_split: parseInt(e.target.value) / 100 }))}
                className="w-full accent-brand-600"
              />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>50% train</span><span>95% train</span>
              </div>
            </div>
            <Input
              label="Random seed"
              type="number"
              value={cfg.random_seed}
              onChange={e => setCfg(c => ({ ...c, random_seed: parseInt(e.target.value) || 42 }))}
            />
          </div>
        </Card>

        <Card title="Processing Summary">
          {result ? (
            <div className="space-y-3">
              <div className="grid grid-cols-3 gap-3">
                <StatBadge label="Total rows" value={result.total_rows} color="brand" />
                <StatBadge label="Train rows" value={result.train_rows} color="brand" />
                <StatBadge label="Test rows" value={result.test_rows} color="brand" />
              </div>
              <div className="p-3 bg-green-50 border border-green-200 rounded-lg text-sm text-green-700">
                ✓ Data processed successfully
              </div>
            </div>
          ) : (
            <div className="text-sm text-gray-400 italic">
              Configure options above and click "Process Data".
            </div>
          )}
        </Card>
      </div>

      {/* Process button */}
      <div className="flex items-center gap-3">
        <Button size="lg" onClick={handleProcess} loading={loading}>
          ⚙️ Process Data
        </Button>
        {result && (
          <Button size="lg" variant="secondary" onClick={() => navigate('/explore')}>
            Next: Explore Data <ChevronRight size={16} />
          </Button>
        )}
      </div>

      {/* Data preview */}
      {result?.data && (
        <Card title="Processed Data Preview">
          <DataTable data={result.data} />
        </Card>
      )}
    </div>
  );
};

export default ProcessPage;
