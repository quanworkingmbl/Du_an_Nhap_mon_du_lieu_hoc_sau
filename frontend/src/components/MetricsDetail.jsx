import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts';

const COLORS = [
  '#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4',
  '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#84cc16',
];

function MetricsDetail({ metrics }) {
  if (!metrics) return null;

  const perClass = metrics.per_class || [];
  const cm = metrics.confusion_matrix || [];
  const classNames = metrics.class_names || [];

  // Dữ liệu cho biểu đồ so sánh
  const comparisonData = perClass.map((c) => ({
    name: c.name.length > 12 ? c.name.slice(0, 12) + '…' : c.name,
    Precision: +(c.precision * 100).toFixed(1),
    Recall: +(c.recall * 100).toFixed(1),
    'F1-Score': +(c.f1 * 100).toFixed(1),
  }));

  // Tìm lớp tốt nhất và kém nhất
  const bestClass = perClass.reduce(
    (best, c) => (c.f1 > best.f1 ? c : best),
    perClass[0] || { name: '-', f1: 0 }
  );
  const worstClass = perClass.reduce(
    (worst, c) => (c.f1 < worst.f1 ? c : worst),
    perClass[0] || { name: '-', f1: 1 }
  );

  // Radar chart data
  const radarData = perClass.map((c) => ({
    subject: c.name.length > 10 ? c.name.slice(0, 10) + '…' : c.name,
    precision: +(c.precision * 100).toFixed(1),
    recall: +(c.recall * 100).toFixed(1),
    f1: +(c.f1 * 100).toFixed(1),
  }));

  return (
    <div className="metrics-detail">
      {/* Quick Summary */}
      <div className="stats-row">
        <div className="stat-card good">
          <div className="stat-icon">🏆</div>
          <div className="stat-info">
            <span className="stat-value">{bestClass.name}</span>
            <span className="stat-label">Lớp tốt nhất (F1: {(bestClass.f1 * 100).toFixed(1)}%)</span>
          </div>
        </div>
        <div className="stat-card warn">
          <div className="stat-icon">⚠️</div>
          <div className="stat-info">
            <span className="stat-value">{worstClass.name}</span>
            <span className="stat-label">Lớp cần cải thiện (F1: {(worstClass.f1 * 100).toFixed(1)}%)</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">📊</div>
          <div className="stat-info">
            <span className="stat-value">
              {(perClass.reduce((sum, c) => sum + c.f1, 0) / perClass.length * 100).toFixed(1)}%
            </span>
            <span className="stat-label">F1-Score trung bình</span>
          </div>
        </div>
      </div>

      {/* Comparison Chart */}
      <div className="card-row">
        <div className="card chart-card wide">
          <h3>📊 So sánh Precision / Recall / F1 theo lớp</h3>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={comparisonData} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
              <Tooltip
                formatter={(value) => `${value}%`}
                contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0' }}
              />
              <Legend />
              <Bar dataKey="Precision" fill="#6366f1" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Recall" fill="#22c55e" radius={[4, 4, 0, 0]} />
              <Bar dataKey="F1-Score" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Radar + Table */}
      <div className="card-row">
        {radarData.length <= 10 && (
          <div className="card chart-card">
            <h3>🕸️ Radar Chart — Hiệu suất đa chiều</h3>
            <ResponsiveContainer width="100%" height={350}>
              <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
                <PolarGrid stroke="#e2e8f0" />
                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                <Radar name="Precision" dataKey="precision" stroke="#6366f1" fill="#6366f1" fillOpacity={0.2} />
                <Radar name="Recall" dataKey="recall" stroke="#22c55e" fill="#22c55e" fillOpacity={0.2} />
                <Radar name="F1" dataKey="f1" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.2} />
                <Legend />
                <Tooltip formatter={(value) => `${value}%`} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        )}

        <div className="card">
          <h3>📋 Bảng chi tiết từng lớp</h3>
          <div className="table-wrapper">
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Lớp</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-Score</th>
                  <th>Số mẫu</th>
                </tr>
              </thead>
              <tbody>
                {perClass.map((c, i) => (
                  <tr key={i}>
                    <td>
                      <span className="class-dot" style={{ backgroundColor: COLORS[i % COLORS.length] }}></span>
                      {c.name}
                    </td>
                    <td>{(c.precision * 100).toFixed(1)}%</td>
                    <td>{(c.recall * 100).toFixed(1)}%</td>
                    <td>
                      <span className={`f1-badge ${c.f1 >= 0.9 ? 'good' : c.f1 >= 0.7 ? 'ok' : 'bad'}`}>
                        {(c.f1 * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td>{c.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Confusion Matrix */}
      {cm.length > 0 && (
        <div className="card">
          <h3>🔢 Confusion Matrix</h3>
          <p className="card-desc">
            Hàng = nhãn thực tế, Cột = nhãn dự đoán. Đường chéo chính là số dự đoán đúng.
          </p>
          <div className="table-wrapper">
            <table className="confusion-matrix">
              <thead>
                <tr>
                  <th></th>
                  {classNames.map((name, i) => (
                    <th key={i} title={name}>
                      {name.length > 8 ? name.slice(0, 8) + '…' : name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cm.map((row, i) => {
                  const maxVal = Math.max(...cm.flat());
                  return (
                    <tr key={i}>
                      <td className="cm-label" title={classNames[i]}>
                        {classNames[i]?.length > 8
                          ? classNames[i].slice(0, 8) + '…'
                          : classNames[i]}
                      </td>
                      {row.map((val, j) => {
                        const isDiagonal = i === j;
                        const intensity = maxVal > 0 ? val / maxVal : 0;
                        const bgColor = isDiagonal
                          ? `rgba(34, 197, 94, ${0.15 + intensity * 0.6})`
                          : val > 0
                            ? `rgba(239, 68, 68, ${0.05 + intensity * 0.4})`
                            : 'transparent';

                        return (
                          <td
                            key={j}
                            className={`cm-cell ${isDiagonal ? 'diagonal' : ''}`}
                            style={{ backgroundColor: bgColor }}
                          >
                            {val}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {metrics.source === 'demo' && (
        <div className="demo-notice">
          ℹ️ Đang hiển thị dữ liệu demo. Hãy chạy <code>python train.py</code> rồi
          khởi động lại API để xem kết quả thật.
        </div>
      )}
    </div>
  );
}

export default MetricsDetail;
