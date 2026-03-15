import { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import MetricsDetail from './components/MetricsDetail';
import Inference from './components/Inference';
import './App.css';

const API_BASE = '/api';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [modelInfo, setModelInfo] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const [infoRes, evalRes] = await Promise.all([
          fetch(`${API_BASE}/model-info`),
          fetch(`${API_BASE}/evaluate`),
        ]);

        if (!infoRes.ok || !evalRes.ok) {
          throw new Error('Không thể kết nối đến API server');
        }

        const infoData = await infoRes.json();
        const evalData = await evalRes.json();

        setModelInfo(infoData);
        setMetrics(evalData);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>🧬 AI Pill Classifier</h1>
          <p className="subtitle">ResNet50 + GCN — Phân tích &amp; Nhận dạng viên thuốc</p>
        </div>
      </header>

      <nav className="app-nav">
        <button
          className={`nav-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          📊 Tổng quan
        </button>
        <button
          className={`nav-btn ${activeTab === 'metrics' ? 'active' : ''}`}
          onClick={() => setActiveTab('metrics')}
        >
          📈 Chi tiết Metrics
        </button>
        <button
          className={`nav-btn ${activeTab === 'inference' ? 'active' : ''}`}
          onClick={() => setActiveTab('inference')}
        >
          🔍 Nhận dạng ảnh
        </button>
      </nav>

      <main className="app-main">
        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Đang tải dữ liệu từ AI model...</p>
          </div>
        )}

        {error && (
          <div className="error-banner">
            <p>⚠️ {error}</p>
            <p className="error-hint">
              Hãy chắc chắn API server đang chạy: <code>python api.py</code>
            </p>
          </div>
        )}

        {!loading && !error && (
          <>
            {activeTab === 'dashboard' && (
              <Dashboard modelInfo={modelInfo} metrics={metrics} />
            )}
            {activeTab === 'metrics' && (
              <MetricsDetail metrics={metrics} />
            )}
            {activeTab === 'inference' && (
              <Inference apiBase={API_BASE} />
            )}
          </>
        )}
      </main>

      <footer className="app-footer">
        <p>Dự án Nhập môn Dữ liệu Học sâu — ResNet50 + GCN Pipeline</p>
      </footer>
    </div>
  );
}

export default App;
