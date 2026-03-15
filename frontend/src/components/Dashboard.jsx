import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend,
} from 'recharts';

const COLORS = [
  '#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4',
  '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#84cc16',
];

function Dashboard({ modelInfo, metrics }) {
  if (!modelInfo || !metrics) return null;

  const accuracy = metrics.accuracy;
  const perClass = metrics.per_class || [];

  // Dữ liệu cho biểu đồ F1
  const f1Data = perClass.map((c) => ({
    name: c.name.length > 12 ? c.name.slice(0, 12) + '…' : c.name,
    f1: +(c.f1 * 100).toFixed(1),
    precision: +(c.precision * 100).toFixed(1),
    recall: +(c.recall * 100).toFixed(1),
  }));

  // Dữ liệu phân phối mẫu
  const sampleData = perClass.map((c, i) => ({
    name: c.name,
    value: c.support,
    color: COLORS[i % COLORS.length],
  }));

  return (
    <div className="dashboard">
      {/* Accuracy Card */}
      <div className="stats-row">
        <div className="stat-card accent">
          <div className="stat-icon">🎯</div>
          <div className="stat-info">
            <span className="stat-value">{(accuracy * 100).toFixed(1)}%</span>
            <span className="stat-label">Accuracy tổng thể</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">🏷️</div>
          <div className="stat-info">
            <span className="stat-value">{modelInfo.dataset?.num_classes || 0}</span>
            <span className="stat-label">Số lớp phân loại</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">🖼️</div>
          <div className="stat-info">
            <span className="stat-value">{modelInfo.dataset?.num_samples || 0}</span>
            <span className="stat-label">Tổng số mẫu</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">🔗</div>
          <div className="stat-info">
            <span className="stat-value">{modelInfo.dataset?.num_edges || 0}</span>
            <span className="stat-label">Số cạnh đồ thị</span>
          </div>
        </div>
      </div>

      {/* Model Info */}
      <div className="card-row">
        <div className="card">
          <h3>🧠 Kiến trúc Model</h3>
          <div className="model-arch">
            <div className="arch-block cnn">
              <strong>CNN Feature Extractor</strong>
              <p>{modelInfo.cnn?.architecture}</p>
              <p>Output: {modelInfo.cnn?.feature_dim} chiều</p>
            </div>
            <div className="arch-arrow">→</div>
            <div className="arch-block graph">
              <strong>Graph Builder</strong>
              <p>Cosine Similarity</p>
              <p>Threshold: {modelInfo.training?.similarity_threshold}</p>
            </div>
            <div className="arch-arrow">→</div>
            <div className="arch-block gcn">
              <strong>GCN Classifier</strong>
              <p>{modelInfo.gcn?.layers} layers</p>
              <p>Hidden: {modelInfo.gcn?.hidden_dim}</p>
            </div>
          </div>
        </div>

        <div className="card">
          <h3>⚙️ Tham số huấn luyện</h3>
          <table className="params-table">
            <tbody>
              <tr><td>Optimizer</td><td>{modelInfo.training?.optimizer}</td></tr>
              <tr><td>Loss Function</td><td>{modelInfo.training?.loss_function}</td></tr>
              <tr><td>Learning Rate</td><td>{modelInfo.training?.learning_rate}</td></tr>
              <tr><td>Epochs</td><td>{modelInfo.training?.epochs}</td></tr>
              <tr><td>Batch Size</td><td>{modelInfo.training?.batch_size}</td></tr>
              <tr><td>GCN Dropout</td><td>{modelInfo.gcn?.dropout}</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Charts */}
      <div className="card-row">
        <div className="card chart-card">
          <h3>📊 F1-Score theo lớp</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={f1Data} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
              <Tooltip
                formatter={(value) => `${value}%`}
                contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0' }}
              />
              <Bar dataKey="f1" fill="#6366f1" radius={[4, 4, 0, 0]} name="F1-Score" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card chart-card">
          <h3>🥧 Phân phối mẫu</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={sampleData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={100}
                label={({ name, percent }) =>
                  `${name.length > 10 ? name.slice(0, 10) + '…' : name} (${(percent * 100).toFixed(0)}%)`
                }
              >
                {sampleData.map((entry, index) => (
                  <Cell key={index} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Source indicator */}
      {metrics.source === 'demo' && (
        <div className="demo-notice">
          ℹ️ Đang hiển thị dữ liệu demo. Hãy chạy <code>python train.py</code> rồi
          khởi động lại API để xem kết quả thật.
        </div>
      )}
    </div>
  );
}

export default Dashboard;
