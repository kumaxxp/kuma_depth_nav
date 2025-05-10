"""
HTMLテンプレートを提供するモジュール
"""

def get_index_html():
    """インデックスページのHTMLを返す"""
    return """
    <html>
    <head>
        <title>Fast Camera Streaming</title>
        <style>            body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
            .container { display: flex; flex-wrap: wrap; gap: 15px; }
            .video-box { background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); position: relative; }
            h2 { margin-top: 0; color: #333; }
            .stats { margin-top: 20px; padding: 10px; background: #e8f5e9; border-radius: 5px; }
            #stats-container { font-family: monospace; }
            .calibration-status { margin-top: 20px; padding: 10px; background: #e3f2fd; border-radius: 5px; }
            .calibration-info { font-family: monospace; }
            .status-active { color: #4caf50; font-weight: bold; }
            .status-inactive { color: #f44336; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Fast Depth Processing System</h1>
        
        <div class="container">
            <div class="video-box">
                <h2>Camera Stream</h2>
                <img src="/video" alt="Camera Stream" />
            </div>
            <div class="video-box">
                <h2>Depth Map</h2>
                <img src="/depth_video" alt="Depth Map" />
            </div>
            <div class="video-box">
                <h2>Depth Grid</h2>
                <img src="/depth_grid" alt="Depth Grid" />
            </div>
        </div>        <div class="stats">
            <h3>Performance Stats</h3>
            <div id="stats-container">Loading stats...</div>
        </div>
        
        <div class="calibration-status">
            <h3>Camera Calibration</h3>
            <div id="calibration-status">Loading calibration status...</div>
        </div>
        
        <script>            // 2秒ごとに統計情報を更新
            async function updateStats() {
                try {
                    const response = await fetch('/stats');
                    const stats = await response.json();
                    const container = document.getElementById('stats-container');
                    
                    let html = '<table>';
                    html += '<tr><th>Stream</th><th>FPS</th><th>Latency (ms)</th></tr>';
                    html += `<tr><td>Camera</td><td>${stats.fps.camera}</td><td>${stats.latency.camera}</td></tr>`;
                    html += `<tr><td>Depth</td><td>${stats.fps.depth}</td><td>-</td></tr>`;
                    html += `<tr><td>Grid</td><td>${stats.fps.grid}</td><td>-</td></tr>`;
                    html += `<tr><td>Inference</td><td>${stats.fps.inference}</td><td>${stats.latency.inference}</td></tr>`;
                    html += `<tr><td>Visualization</td><td>-</td><td>${stats.latency.visualization}</td></tr>`;
                    html += `<tr><td>Total Delay</td><td>-</td><td>${stats.latency.total_delay}</td></tr>`;
                    html += '</table>';
                    
                    container.innerHTML = html;
                } catch (e) {
                    console.error('Failed to fetch stats:', e);
                }
            }
            
            // キャリブレーション状態を更新
            async function updateCalibrationStatus() {
                try {
                    const response = await fetch('/calibration_status');
                    const status = await response.json();
                    const container = document.getElementById('calibration-status');
                    
                    if (status.calibrated) {
                        let html = '<div class="calibration-info">';
                        html += `<p>キャリブレーション状態: <span class="status-active">有効</span></p>`;
                        html += `<p>適用状態: <span class="${status.applied ? 'status-active' : 'status-inactive'}">${status.applied ? '適用中' : '未適用'}</span></p>`;
                        html += `<p>RMS誤差: ${status.rms_error.toFixed(4)}</p>`;
                        if (status.calibration_time) {
                            html += `<p>キャリブレーション日時: ${status.calibration_time}</p>`;
                        }
                        html += '</div>';
                        container.innerHTML = html;
                    } else {
                        container.innerHTML = '<p>キャリブレーションは実行されていません</p>';
                    }
                } catch (e) {
                    console.error('Failed to fetch calibration status:', e);
                }
            }
            
            // 定期的に更新
            setInterval(updateStats, 2000);
            setInterval(updateCalibrationStatus, 5000);
            
            // 初期更新
            updateStats();
            updateCalibrationStatus();
        </script>
    </body>
    </html>
    """
