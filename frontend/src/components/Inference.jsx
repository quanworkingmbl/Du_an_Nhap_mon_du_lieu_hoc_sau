import { useState, useRef } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts';

const COLORS = [
  '#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4',
  '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#84cc16',
];

function Inference({ apiBase }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
      setError(null);

      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setResult(null);
      setError(null);

      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);

      const res = await fetch(`${apiBase}/inference`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error('Lỗi từ server');
      }

      const data = await res.json();
      if (data.error) {
        throw new Error(data.error);
      }
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Dữ liệu cho biểu đồ xác suất
  const probData = result?.probabilities?.map((p, i) => ({
    name: p.class.length > 12 ? p.class.slice(0, 12) + '…' : p.class,
    probability: +(p.probability * 100).toFixed(1),
    color: COLORS[i % COLORS.length],
  })) || [];

  return (
    <div className="inference">
      <div className="card-row">
        {/* Upload Area */}
        <div className="card">
          <h3>📤 Tải ảnh viên thuốc</h3>

          <div
            className={`drop-zone ${preview ? 'has-image' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => fileInputRef.current?.click()}
          >
            {preview ? (
              <img src={preview} alt="Preview" className="preview-img" />
            ) : (
              <div className="drop-placeholder">
                <span className="drop-icon">🖼️</span>
                <p>Kéo thả ảnh vào đây</p>
                <p className="drop-hint">hoặc click để chọn file</p>
              </div>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />

          {selectedFile && (
            <p className="file-name">📎 {selectedFile.name}</p>
          )}

          <div className="inference-actions">
            <button
              className="btn-primary"
              onClick={handleSubmit}
              disabled={!selectedFile || loading}
            >
              {loading ? '⏳ Đang phân tích...' : '🔬 Phân tích ảnh'}
            </button>
            <button className="btn-secondary" onClick={handleReset}>
              🔄 Làm mới
            </button>
          </div>

          {error && (
            <div className="error-banner small">
              <p>❌ {error}</p>
            </div>
          )}
        </div>

        {/* Results */}
        <div className="card">
          <h3>🎯 Kết quả nhận dạng</h3>

          {!result && !loading && (
            <div className="no-result">
              <span className="no-result-icon">🔍</span>
              <p>Tải ảnh và nhấn &quot;Phân tích ảnh&quot; để xem kết quả</p>
            </div>
          )}

          {loading && (
            <div className="loading inline">
              <div className="spinner"></div>
              <p>Đang chạy AI phân tích...</p>
            </div>
          )}

          {result && (
            <div className="result-content">
              <div className="result-main">
                <div className="predicted-class">
                  <span className="result-label">Dự đoán:</span>
                  <span className="result-value">{result.predicted_class}</span>
                </div>
                <div className="confidence-meter">
                  <span className="result-label">Độ tin cậy:</span>
                  <div className="confidence-bar-wrapper">
                    <div
                      className="confidence-bar"
                      style={{
                        width: `${(result.confidence * 100).toFixed(1)}%`,
                        backgroundColor:
                          result.confidence > 0.8
                            ? '#22c55e'
                            : result.confidence > 0.5
                              ? '#f59e0b'
                              : '#ef4444',
                      }}
                    ></div>
                    <span className="confidence-text">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>

              {result.source === 'demo' && (
                <div className="demo-notice small">
                  ℹ️ Kết quả demo — Model chưa được load
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Probability Chart */}
      {result && probData.length > 0 && (
        <div className="card">
          <h3>📊 Xác suất từng lớp</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={probData} layout="vertical" margin={{ top: 10, right: 30, left: 80, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 12 }} />
              <YAxis dataKey="name" type="category" tick={{ fontSize: 12 }} />
              <Tooltip
                formatter={(value) => `${value}%`}
                contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0' }}
              />
              <Bar dataKey="probability" name="Xác suất" radius={[0, 4, 4, 0]}>
                {probData.map((entry, index) => (
                  <Cell key={index} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

export default Inference;
