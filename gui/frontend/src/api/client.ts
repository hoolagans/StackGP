import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

export interface DataTable {
  columns: string[];
  dtypes: Record<string, string>;
  rows: (string | number | null)[][];
  total_rows: number;
  shape: [number, number];
}

export interface ColumnConfig {
  target_col: string;
  feature_cols: string[];
}

export interface ProcessConfig {
  fill_missing: 'drop' | 'mean' | 'median' | 'zero';
  normalize: 'none' | 'minmax' | 'zscore';
  train_split: number;
  random_seed: number;
}

export interface TrainConfig {
  generations: number;
  pop_size: number;
  max_complexity: number;
  max_length: number;
  mutation_rate: number;
  crossover_rate: number;
  spawn_rate: number;
  time_limit: number;
  ops_set: 'default' | 'all' | 'boolean';
  align: boolean;
  data_subsample: boolean;
  allow_early_termination: boolean;
  early_termination_threshold: number;
}

export interface ModelInfo {
  id: number;
  expression: string;
  rmse: number | null;
  fitness: number | null;
  complexity: number | null;
  metrics: (number | null)[];
  predictions: (number | null)[];
  residuals: number[];
}

export interface ParetoPoint {
  id: number;
  fitness: number | null;
  complexity: number | null;
  expression: string;
}

export interface TrainStatus {
  status: string;
  progress: number;
  total: number;
  log: { gen?: number; fitness?: number; complexity?: number; error?: string }[];
  model_count: number;
}

export interface SessionState {
  has_raw_data: boolean;
  has_processed_data: boolean;
  target_col: string | null;
  feature_cols: string[];
  train_rows: number;
  test_rows: number;
  model_count: number;
  ensemble_size: number;
  training_status: string;
}

// Import
export const uploadFile = (file: File) => {
  const form = new FormData();
  form.append('file', file);
  return api.post<DataTable>('/upload', form);
};
export const getRawData = () => api.get<DataTable>('/data/raw');
export const configureColumns = (cfg: ColumnConfig) => api.post('/data/configure', cfg);

// Process
export const processData = (cfg: ProcessConfig) => api.post<{ ok: boolean; total_rows: number; train_rows: number; test_rows: number; data: DataTable }>('/data/process', cfg);
export const getProcessedData = () => api.get<DataTable>('/data/processed');

// Explore
export const getStats = () => api.get<Record<string, {
  count: number; mean: number; std: number; min: number;
  q25: number; median: number; q75: number; max: number;
  missing: number; unique: number; skewness: number; kurtosis: number;
}>>('/explore/stats');
export const getCorrelation = () => api.get<{ columns: string[]; matrix: (number | null)[][] }>('/explore/correlation');
export const getColumnData = (col: string) => api.get<{ values: number[]; histogram: { counts: number[]; edges: number[]; bin_centers: number[] } }>(`/explore/column/${encodeURIComponent(col)}`);
export const getScatter = (x: string, y: string) => api.get<{ x: number[]; y: number[] }>(`/explore/scatter?x_col=${encodeURIComponent(x)}&y_col=${encodeURIComponent(y)}`);
export const getPairplot = () => api.get<{ columns: string[]; data: Record<string, number[]> }>('/explore/pairplot');

// Modeling
export const startTraining = (cfg: TrainConfig) => api.post('/train/start', cfg);
export const getTrainStatus = () => api.get<TrainStatus>('/train/status');
export const stopTraining = () => api.post('/train/stop');

// Analysis
export const getModels = (maxModels?: number) => api.get<{ models: ModelInfo[] }>(`/models?max_models=${maxModels ?? 50}`);
export const getModel = (id: number) => api.get<ModelInfo>(`/models/${id}`);
export const getPareto = () => api.get<{ pareto: ParetoPoint[] }>('/models/pareto');
export const getResiduals = (id: number) => api.get<{
  actual: (number | null)[]; predicted: (number | null)[]; residuals: (number | null)[];
  train_rmse: number; test_rmse: number | null; train_fitness: number; test_fitness: number | null;
}>(`/models/${id}/residuals`);
export const getVariableImportance = () => api.get<{ importance: Record<string, number> }>('/models/variable_importance');

// Ensemble
export const buildEnsemble = (num_clusters: number) => api.post('/ensemble/build', { num_clusters });
export const getEnsemble = () => api.get<{ ensemble: { id: number; expression: string; complexity: number | null }[] }>('/ensemble');

// Predict
export const predictModel = (id: number, rows: number[][]) => api.post<{ predictions: (number | null)[] }>(`/predict/model/${id}`, { rows });
export const predictEnsemble = (rows: number[][]) => api.post<{ predictions: (number | null)[]; uncertainty: (number | null)[] }>('/predict/ensemble', { rows });
export const findMaxUncertaintyPoint = () => api.post<{ point: number[] }>('/predict/uncertainty_point');

// Session
export const getSessionState = () => api.get<SessionState>('/session/state');
export const resetSession = () => api.delete('/session/reset');

export default api;
