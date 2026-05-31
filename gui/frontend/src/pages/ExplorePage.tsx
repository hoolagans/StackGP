import React, { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, LineChart, Line,
} from 'recharts';
import {
  getStats, getCorrelation, getColumnData, getScatter, getSessionState,
} from '../api/client';
import { Card, Button, Select, StatBadge, Spinner, EmptyState } from '../components/ui';
import toast from 'react-hot-toast';

// ---------- Correlation Heatmap ----------
const CORR_COLORS = [
  '#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf',
  '#e0f3f8', '#abd9e9', '#74add1', '#4575b4',
].reverse();

function corrColor(v: number | null): string {
  if (v == null) return '#e5e7eb';
  const t = (v + 1) / 2;
  const idx = Math.min(Math.floor(t * CORR_COLORS.length), CORR_COLORS.length - 1);
  return CORR_COLORS[idx];
}

const CorrHeatmap: React.FC<{ columns: string[]; matrix: (number | null)[][] }> = ({ columns, matrix }) => {
  const [hovered, setHovered] = useState<{ r: number; c: number } | null>(null);
  const cellSize = Math.max(24, Math.min(48, Math.floor(440 / columns.length)));

  return (
    <div className="overflow-auto">
      <div className="inline-block">
        {/* Column labels */}
        <div className="flex" style={{ paddingLeft: 90 }}>
          {columns.map((col, ci) => (
            <div
              key={ci}
              style={{ width: cellSize, minWidth: cellSize }}
              className="text-center overflow-hidden"
            >
              <span
                className="text-[9px] text-gray-500 block"
                style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', height: 56, maxHeight: 56 }}
              >
                {col}
              </span>
            </div>
          ))}
        </div>
        {matrix.map((row, ri) => (
          <div key={ri} className="flex items-center">
            <div className="text-[10px] text-gray-500 text-right pr-2" style={{ width: 88 }}>
              {columns[ri]}
            </div>
            {row.map((val, ci) => (
              <div
                key={ci}
                style={{ width: cellSize, height: cellSize, minWidth: cellSize, backgroundColor: corrColor(val) }}
                className="relative cursor-pointer border border-white"
                onMouseEnter={() => setHovered({ r: ri, c: ci })}
                onMouseLeave={() => setHovered(null)}
              >
                {hovered?.r === ri && hovered?.c === ci && val != null && (
                  <div className="absolute z-20 -top-8 left-1/2 -translate-x-1/2 bg-gray-800 text-white text-xs px-2 py-1 rounded whitespace-nowrap pointer-events-none">
                    {columns[ri]} × {columns[ci]}: {val.toFixed(3)}
                  </div>
                )}
              </div>
            ))}
          </div>
        ))}
        {/* Color scale */}
        <div className="flex items-center gap-2 mt-2 ml-[90px]">
          <span className="text-[9px] text-gray-400">−1</span>
          <div className="flex h-2 rounded overflow-hidden" style={{ width: cellSize * columns.length }}>
            {CORR_COLORS.map((c, i) => (
              <div key={i} style={{ flex: 1, backgroundColor: c }} />
            ))}
          </div>
          <span className="text-[9px] text-gray-400">+1</span>
        </div>
      </div>
    </div>
  );
};

// ---------- Main Page ----------
const ExplorePage: React.FC = () => {
  const navigate = useNavigate();
  const [hasData, setHasData] = useState(false);
  const [allCols, setAllCols] = useState<string[]>([]);
  const [targetCol, setTargetCol] = useState('');
  const [featureCols, setFeatureCols] = useState<string[]>([]);

  const [stats, setStats] = useState<Record<string, any> | null>(null);
  const [corr, setCorr] = useState<{ columns: string[]; matrix: (number | null)[][] } | null>(null);
  const [selectedCol, setSelectedCol] = useState('');
  const [colData, setColData] = useState<{ values: number[]; histogram: any } | null>(null);
  const [scatterX, setScatterX] = useState('');
  const [scatterY, setScatterY] = useState('');
  const [scatterData, setScatterData] = useState<{ x: number[]; y: number[] } | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getSessionState().then(r => {
      setHasData(r.data.has_processed_data);
      setTargetCol(r.data.target_col ?? '');
      setFeatureCols(r.data.feature_cols);
      const cols = [...r.data.feature_cols, r.data.target_col].filter(Boolean) as string[];
      setAllCols(cols);
      if (cols.length > 0) setSelectedCol(cols[0]);
      if (cols.length > 1) { setScatterX(cols[0]); setScatterY(r.data.target_col ?? cols[1]); }
    }).catch(e => { console.error('Failed to load session state', e); });
  }, []);

  useEffect(() => {
    if (!hasData) return;
    setLoading(true);
    Promise.all([getStats(), getCorrelation()])
      .then(([s, c]) => {
        setStats(s.data);
        setCorr(c.data);
      })
      .catch(() => toast.error('Failed to load stats'))
      .finally(() => setLoading(false));
  }, [hasData]);

  useEffect(() => {
    if (!selectedCol) return;
    getColumnData(selectedCol).then(r => setColData(r.data)).catch(() => {});
  }, [selectedCol]);

  const loadScatter = useCallback(() => {
    if (!scatterX || !scatterY) return;
    getScatter(scatterX, scatterY).then(r => setScatterData(r.data)).catch(() => {});
  }, [scatterX, scatterY]);

  useEffect(() => { loadScatter(); }, [loadScatter]);

  if (!hasData) {
    return (
      <div className="p-6">
        <EmptyState icon="🔍" title="No processed data" subtitle="Process your data first" />
        <div className="flex justify-center mt-4">
          <Button onClick={() => navigate('/process')}>Go to Process</Button>
        </div>
      </div>
    );
  }

  if (loading) {
    return <div className="flex justify-center items-center h-64"><Spinner /></div>;
  }

  const histData = colData?.histogram
    ? colData.histogram.bin_centers.map((x: number, i: number) => ({
        x: parseFloat(x.toFixed(4)),
        count: colData.histogram.counts[i],
      }))
    : [];

  const scatterPoints = scatterData
    ? scatterData.x.map((x, i) => ({ x, y: scatterData.y[i] }))
    : [];

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Explore Data</h1>
          <p className="text-sm text-gray-500 mt-1">Statistics, distributions, correlations, and relationships.</p>
        </div>
        <Button onClick={() => navigate('/model')}>Next: Model →</Button>
      </div>

      {/* Summary stats table */}
      {stats && (
        <Card title="Descriptive Statistics">
          <div className="overflow-x-auto">
            <table className="min-w-full text-xs">
              <thead>
                <tr className="bg-gray-50 border-b border-gray-200">
                  {['Column', 'Count', 'Mean', 'Std', 'Min', 'Q25', 'Median', 'Q75', 'Max', 'Missing', 'Skewness', 'Kurtosis'].map(h => (
                    <th key={h} className="px-3 py-2 text-left font-semibold text-gray-600 whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(stats).map(([col, s]: [string, any], i) => (
                  <tr
                    key={col}
                    onClick={() => setSelectedCol(col)}
                    className={`cursor-pointer border-b border-gray-100 hover:bg-brand-50 ${i % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}
                      ${col === selectedCol ? 'bg-brand-100' : ''}
                      ${col === targetCol ? 'font-semibold' : ''}`}
                  >
                    <td className="px-3 py-1.5 text-gray-800 whitespace-nowrap">
                      {col}
                      {col === targetCol && <span className="ml-1 text-brand-500 text-[10px]">▲ target</span>}
                    </td>
                    {[s.count, s.mean, s.std, s.min, s.q25, s.median, s.q75, s.max, s.missing, s.skewness, s.kurtosis].map((v, vi) => (
                      <td key={vi} className={`px-3 py-1.5 text-right text-gray-600 ${vi === 8 && v > 0 ? 'text-red-500' : ''}`}>
                        {typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : '—'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-400 mt-2">Click a row to see its distribution below.</p>
        </Card>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Histogram */}
        <Card
          title={`Distribution: ${selectedCol}`}
          action={
            <select
              value={selectedCol}
              onChange={e => setSelectedCol(e.target.value)}
              className="text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none"
            >
              {allCols.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          }
        >
          {histData.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={histData} barCategoryGap="2%">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="x" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip formatter={(v: number) => [v, 'Count']} />
                  <Bar dataKey="count" fill="#3b82f6" opacity={0.85} />
                </BarChart>
              </ResponsiveContainer>
              {stats?.[selectedCol] && (
                <div className="grid grid-cols-4 gap-2 mt-3">
                  {[
                    { label: 'Mean', value: stats[selectedCol].mean?.toFixed(3) },
                    { label: 'Std', value: stats[selectedCol].std?.toFixed(3) },
                    { label: 'Skew', value: stats[selectedCol].skewness?.toFixed(3) },
                    { label: 'Kurt', value: stats[selectedCol].kurtosis?.toFixed(3) },
                  ].map(({ label, value }) => (
                    <div key={label} className="text-center p-2 bg-gray-50 rounded-lg">
                      <div className="text-[10px] text-gray-500">{label}</div>
                      <div className="text-sm font-semibold text-gray-800">{value}</div>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="text-sm text-gray-400 italic py-8 text-center">Loading...</div>
          )}
        </Card>

        {/* Scatter plot */}
        <Card
          title="Scatter Plot"
          action={
            <div className="flex items-center gap-2">
              <select
                value={scatterX}
                onChange={e => setScatterX(e.target.value)}
                className="text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none"
              >
                {allCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
              <span className="text-xs text-gray-400">vs</span>
              <select
                value={scatterY}
                onChange={e => setScatterY(e.target.value)}
                className="text-xs border border-gray-200 rounded px-2 py-1 focus:outline-none"
              >
                {allCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          }
        >
          {scatterPoints.length > 0 ? (
            <ResponsiveContainer width="100%" height={230}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="x" name={scatterX} tick={{ fontSize: 10 }} label={{ value: scatterX, position: 'insideBottom', offset: -2, fontSize: 11 }} />
                <YAxis dataKey="y" name={scatterY} tick={{ fontSize: 10 }} label={{ value: scatterY, angle: -90, position: 'insideLeft', offset: 8, fontSize: 11 }} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter data={scatterPoints} fill="#3b82f6" opacity={0.6} r={3} />
              </ScatterChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-sm text-gray-400 italic py-8 text-center">Loading...</div>
          )}
        </Card>
      </div>

      {/* Correlation heatmap */}
      {corr && (
        <Card title="Correlation Matrix (Pearson)">
          <CorrHeatmap columns={corr.columns} matrix={corr.matrix} />
        </Card>
      )}

      {/* Target correlations */}
      {corr && targetCol && (() => {
        const ti = corr.columns.indexOf(targetCol);
        if (ti < 0) return null;
        const sorted = corr.columns
          .map((col, i) => ({ col, val: corr.matrix[ti][i] ?? 0 }))
          .filter(d => d.col !== targetCol)
          .sort((a, b) => Math.abs(b.val) - Math.abs(a.val));
        return (
          <Card title={`Feature Correlation with Target: ${targetCol}`}>
            <ResponsiveContainer width="100%" height={Math.max(200, sorted.length * 28)}>
              <BarChart data={sorted} layout="vertical" margin={{ left: 80, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" domain={[-1, 1]} tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="col" tick={{ fontSize: 10 }} width={80} />
                <Tooltip formatter={(v: number) => [v.toFixed(4), 'Correlation']} />
                <Bar dataKey="val" radius={2}>
                  {sorted.map((d, i) => (
                    <Cell key={i} fill={d.val >= 0 ? '#3b82f6' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        );
      })()}
    </div>
  );
};

export default ExplorePage;
