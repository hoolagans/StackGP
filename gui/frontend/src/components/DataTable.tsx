import React, { useState } from 'react';
import { DataTable as DT } from '../api/client';

interface Props {
  data: DT;
  maxRows?: number;
  onColumnClick?: (col: string) => void;
  highlightCols?: string[];
}

const DataTable: React.FC<Props> = ({ data, maxRows = 100, onColumnClick, highlightCols = [] }) => {
  const [sortCol, setSortCol] = useState<number | null>(null);
  const [sortAsc, setSortAsc] = useState(true);
  const [page, setPage] = useState(0);
  const pageSize = 20;

  const displayRows = React.useMemo(() => {
    let rows = data.rows.slice(0, maxRows);
    if (sortCol !== null) {
      rows = [...rows].sort((a, b) => {
        const va = a[sortCol], vb = b[sortCol];
        if (va == null) return 1;
        if (vb == null) return -1;
        return sortAsc ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
      });
    }
    return rows;
  }, [data.rows, maxRows, sortCol, sortAsc]);

  const pageRows = displayRows.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = Math.ceil(displayRows.length / pageSize);

  const handleSort = (i: number) => {
    if (sortCol === i) setSortAsc(!sortAsc);
    else { setSortCol(i); setSortAsc(true); }
    setPage(0);
  };

  return (
    <div className="overflow-hidden rounded-lg border border-gray-200">
      <div className="overflow-x-auto max-h-96">
        <table className="min-w-full text-xs">
          <thead className="bg-gray-50 sticky top-0 z-10">
            <tr>
              {data.columns.map((col, i) => (
                <th
                  key={col}
                  onClick={() => handleSort(i)}
                  className={`px-3 py-2.5 text-left font-semibold text-gray-600 cursor-pointer whitespace-nowrap border-b border-gray-200 hover:bg-gray-100 select-none
                    ${highlightCols.includes(col) ? 'bg-brand-50 text-brand-700' : ''}
                    ${onColumnClick ? 'hover:text-brand-600' : ''}`}
                >
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); onColumnClick?.(col); }}
                    className="text-left"
                  >
                    {col}
                    {sortCol === i && <span className="ml-1">{sortAsc ? '↑' : '↓'}</span>}
                  </button>
                  <div className="text-gray-400 font-normal text-[10px]">{data.dtypes[col]}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row, ri) => (
              <tr key={ri} className={ri % 2 === 0 ? 'bg-white' : 'bg-gray-50/60'}>
                {row.map((cell, ci) => (
                  <td key={ci} className="px-3 py-1.5 text-gray-700 whitespace-nowrap border-b border-gray-100">
                    {cell == null ? <span className="text-red-400 italic">null</span> : String(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-t border-gray-200 text-xs text-gray-500">
        <span>{data.total_rows.toLocaleString()} rows × {data.columns.length} cols</span>
        {totalPages > 1 && (
          <div className="flex items-center gap-2">
            <button
              className="px-2 py-0.5 rounded border border-gray-300 hover:bg-white disabled:opacity-40"
              onClick={() => setPage(p => Math.max(0, p - 1))}
              disabled={page === 0}
            >←</button>
            <span>{page + 1} / {totalPages}</span>
            <button
              className="px-2 py-0.5 rounded border border-gray-300 hover:bg-white disabled:opacity-40"
              onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
              disabled={page === totalPages - 1}
            >→</button>
          </div>
        )}
      </div>
    </div>
  );
};

export default DataTable;
