import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Layout from './components/Layout';
import ErrorBoundary from './components/ErrorBoundary';
import ImportPage from './pages/ImportPage';
import ProcessPage from './pages/ProcessPage';
import ExplorePage from './pages/ExplorePage';
import ModelPage from './pages/ModelPage';
import AnalysisPage from './pages/AnalysisPage';
import PredictPage from './pages/PredictPage';

function App() {
  return (
    <BrowserRouter>
      <Toaster
        position="top-right"
        toastOptions={{
          className: 'text-sm',
          duration: 3000,
        }}
      />
      <ErrorBoundary>
        <Layout>
          <Routes>
            <Route path="/" element={<Navigate to="/import" replace />} />
            <Route path="/import" element={<ImportPage />} />
            <Route path="/process" element={<ProcessPage />} />
            <Route path="/explore" element={<ExplorePage />} />
            <Route path="/model" element={<ModelPage />} />
            <Route path="/analysis" element={<AnalysisPage />} />
            <Route path="/predict" element={<PredictPage />} />
          </Routes>
        </Layout>
      </ErrorBoundary>
    </BrowserRouter>
  );
}

export default App;
