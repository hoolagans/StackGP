import React, { useCallback, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, ChevronRight, X, Check } from 'lucide-react';
import { uploadFile, configureColumns, DataTable as DT } from '../api/client';
import { Card, Button, Select, Badge, EmptyState } from '../components/ui';
import DataTable from '../components/DataTable';
import toast from 'react-hot-toast';

const ImportPage: React.FC = () => {
  const [data, setData] = useState<DT | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [targetCol, setTargetCol] = useState('');
  const [featureCols, setFeatureCols] = useState<string[]>([]);
  const [configured, setConfigured] = useState(false);
  const navigate = useNavigate();

  const handleFile = async (file: File) => {
    if (!file) return;
    setLoading(true);
    try {
      const res = await uploadFile(file);
      setData(res.data);
      setConfigured(false);
      setFeatureCols([]);
      setTargetCol('');
      toast.success(`Loaded ${res.data.total_rows} rows × ${res.data.columns.length} columns`);
    } catch (e: any) {
      toast.error(e.response?.data?.detail ?? 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, []);

  const onFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const numericCols = data?.columns.filter(c => {
    const dt = data.dtypes[c] ?? '';
    return dt.startsWith('float') || dt.startsWith('int') || dt === 'number';
  }) ?? [];

  const toggleFeature = (col: string) => {
    setFeatureCols(prev =>
      prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]
    );
    if (configured) setConfigured(false);
  };

  const selectAllFeatures = () => {
    if (!data) return;
    setFeatureCols(numericCols.filter(c => c !== targetCol));
    setConfigured(false);
  };

  const handleConfigure = async () => {
    if (!targetCol || featureCols.length === 0) {
      toast.error('Select a target column and at least one feature');
      return;
    }
    try {
      await configureColumns({ target_col: targetCol, feature_cols: featureCols });
      setConfigured(true);
      toast.success('Columns configured');
    } catch (e: any) {
      toast.error(e.response?.data?.detail ?? 'Failed');
    }
  };

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Import Data</h1>
        <p className="text-sm text-gray-500 mt-1">Upload a CSV or Excel file, then configure feature and target columns.</p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`relative rounded-2xl border-2 border-dashed transition-colors cursor-pointer
          ${dragging ? 'border-brand-500 bg-brand-50' : 'border-gray-300 bg-white hover:border-brand-400 hover:bg-gray-50'}`}
      >
        <label className="flex flex-col items-center justify-center py-14 cursor-pointer">
          <Upload size={36} className={`mb-3 ${dragging ? 'text-brand-500' : 'text-gray-400'}`} />
          <div className="font-medium text-gray-700">Drop a file here, or <span className="text-brand-600 underline">browse</span></div>
          <div className="text-xs text-gray-400 mt-1">CSV, Excel (.xlsx / .xls) — any size</div>
          <input type="file" accept=".csv,.xlsx,.xls" className="hidden" onChange={onFileInput} />
        </label>
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/80 rounded-2xl">
            <div className="w-8 h-8 border-4 border-brand-200 border-t-brand-600 rounded-full animate-spin" />
          </div>
        )}
      </div>

      {data && (
        <>
          {/* Preview */}
          <Card title={`Data Preview — ${data.total_rows.toLocaleString()} rows × ${data.columns.length} cols`}>
            <DataTable data={data} highlightCols={[targetCol, ...featureCols]} />
          </Card>

          {/* Column configuration */}
          <Card title="Configure Columns">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Target */}
              <div>
                <label className="text-xs font-semibold text-gray-600 mb-2 block">Target Variable (Y)</label>
                <select
                  className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-400"
                  value={targetCol}
                  onChange={e => { setTargetCol(e.target.value); setConfigured(false); }}
                >
                  <option value="">— select target —</option>
                  {data.columns.map(c => (
                    <option key={c} value={c}>{c} ({data.dtypes[c]})</option>
                  ))}
                </select>
                {targetCol && (
                  <div className="mt-2 text-xs text-gray-500">
                    Selected: <Badge text={targetCol} variant="blue" />
                  </div>
                )}
              </div>

              {/* Features */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-xs font-semibold text-gray-600">Feature Columns (X)</label>
                  <button onClick={selectAllFeatures} className="text-xs text-brand-600 hover:underline">
                    Select all (excl. target)
                  </button>
                </div>
                <div className="max-h-48 overflow-y-auto border border-gray-200 rounded-lg divide-y divide-gray-100">
                  {data.columns.map(col => {
                    const isTarget = col === targetCol;
                    const isNonNumeric = !numericCols.includes(col);
                    const selected = featureCols.includes(col);
                    const disabled = isTarget || isNonNumeric;
                    return (
                      <label
                        key={col}
                        className={`flex items-center gap-3 px-3 py-2 cursor-pointer hover:bg-gray-50 select-none
                          ${disabled ? 'opacity-40 cursor-not-allowed' : ''}
                          ${selected ? 'bg-brand-50' : ''}`}
                      >
                        <input
                          type="checkbox"
                          checked={selected}
                          disabled={disabled}
                          onChange={() => toggleFeature(col)}
                          className="rounded border-gray-300 text-brand-600"
                        />
                        <span className="text-sm text-gray-700 flex-1">{col}</span>
                        <span className="text-xs text-gray-400">{data.dtypes[col]}</span>
                        {isTarget && <Badge text="target" variant="blue" />}
                        {!isTarget && isNonNumeric && <Badge text="non-numeric" variant="gray" />}
                      </label>
                    );
                  })}
                </div>
                <div className="text-xs text-gray-500 mt-1">{featureCols.length} features selected</div>
              </div>
            </div>

            <div className="mt-5 flex items-center gap-3">
              <Button onClick={handleConfigure} disabled={!targetCol || featureCols.length === 0}>
                <Check size={14} /> Apply Configuration
              </Button>
              {configured && (
                <span className="flex items-center gap-1 text-sm text-green-600 font-medium">
                  <Check size={14} /> Configured
                </span>
              )}
            </div>
          </Card>

          {configured && (
            <div className="flex justify-end">
              <Button size="lg" onClick={() => navigate('/process')}>
                Next: Process Data <ChevronRight size={16} />
              </Button>
            </div>
          )}
        </>
      )}

      {!data && !loading && (
        <EmptyState icon="📤" title="No data loaded yet" subtitle="Upload a CSV or Excel file to get started" />
      )}
    </div>
  );
};

export default ImportPage;
