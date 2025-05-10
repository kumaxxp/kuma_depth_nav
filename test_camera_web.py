#!/usr/bin/env python3
"""
Webベースのカメラとキャリブレーションのテストスクリプト
"""
import cv2
import numpy as np
import os
import sys
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

# アプリケーションのルートディレクトリをPythonパスに追加
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 必要なモジュールをインポート
from app.calibration.calibration import calibration
from app.camera.capture import camera

# FastAPIアプリケーションを作成
app = FastAPI(title="Camera Calibration Test")

@app.get("/", response_class=HTMLResponse)
async def index():
    """インデックスページを返す"""
    return get_index_html()

@app.get("/video")
async def video():
    """カメラビデオストリームを返す"""
    return StreamingResponse(camera.get_camera_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/toggle_calibration")
async def toggle_calibration():
    """キャリブレーションのON/OFFを切り替える"""
    if camera.use_calibration:
        camera.disable_calibration()
        return {"status": "Calibration OFF"}
    else:
        if calibration.camera_matrix is not None and calibration.dist_coeffs is not None:
            camera.set_calibration(calibration)
            return {"status": "Calibration ON"}
        else:
            return {"status": "Calibration data not available", "error": True}

@app.get("/calibration_status")
async def get_calibration_status():
    """キャリブレーション状態を取得する"""
    calib_file = "calibration_data/calibration.json"
    calib_exists = os.path.exists(calib_file)
    calibration_loaded = calibration.camera_matrix is not None and calibration.dist_coeffs is not None
    
    return {
        "calibration_file_exists": calib_exists,
        "calibration_data_loaded": calibration_loaded,
        "calibration_applied": camera.use_calibration,
        "rms_error": calibration.rms_error if hasattr(calibration, "rms_error") else None
    }

def get_index_html():
    """HTMLテンプレートを返す"""
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>カメラキャリブレーションテスト</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background: #f0f0f0;
            }
            h1, h2 {
                color: #2c3e50;
            }
            .container {
                display: flex;
                flex-direction: column;
                gap: 20px;
                max-width: 800px;
                margin: 0 auto;
            }
            .card {
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .controls {
                margin-top: 15px;
            }
            button {
                background: #3498db;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 5px;
                cursor: pointer;
                transition: background 0.3s;
            }
            button:hover {
                background: #2980b9;
            }
            .status-panel {
                background: #ecf0f1;
                padding: 10px;
                border-radius: 5px;
                margin-top: 15px;
            }
            .status-item {
                margin-bottom: 5px;
            }
            .status-active {
                color: #27ae60;
                font-weight: bold;
            }
            .status-inactive {
                color: #e74c3c;
                font-weight: bold;
            }
            img {
                width: 100%;
                border-radius: 5px;
            }
            .links {
                margin-top: 20px;
                text-align: center;
            }
            .links a {
                color: #3498db;
                margin: 0 10px;
                text-decoration: none;
            }
            .links a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>カメラキャリブレーションテスト</h1>
                <p>カメラ映像を表示し、キャリブレーション機能をテストするシンプルなインターフェースです。</p>
                
                <div class="card">
                    <h2>カメラ映像</h2>
                    <img src="/video" alt="カメラ映像" />
                </div>
                
                <div class="card">
                    <h2>キャリブレーション状態</h2>
                    <div class="status-panel" id="status-panel">
                        <p>読み込み中...</p>
                    </div>
                    
                    <div class="controls">
                        <button id="toggle-btn">キャリブレーション ON/OFF</button>
                    </div>
                </div>
                
                <div class="links">
                    <a href="http://localhost:8000/">メインアプリへ</a>
                    <a href="http://localhost:8000/calibration">キャリブレーション設定へ</a>
                </div>
            </div>
        </div>
        
        <script>
            const toggleBtn = document.getElementById('toggle-btn');
            const statusPanel = document.getElementById('status-panel');
            
            // キャリブレーション状態を更新
            async function updateStatus() {
                try {
                    const response = await fetch('/calibration_status');
                    const status = await response.json();
                    
                    let html = '<div class="status-info">';
                    html += `<p class="status-item">キャリブレーションファイル: 
                           <span class="${status.calibration_file_exists ? 'status-active' : 'status-inactive'}">
                           ${status.calibration_file_exists ? '存在します' : '見つかりません'}</span></p>`;
                    
                    html += `<p class="status-item">キャリブレーションデータ: 
                           <span class="${status.calibration_data_loaded ? 'status-active' : 'status-inactive'}">
                           ${status.calibration_data_loaded ? '読み込み済み' : '未読み込み'}</span></p>`;
                    
                    html += `<p class="status-item">キャリブレーション適用: 
                           <span class="${status.calibration_applied ? 'status-active' : 'status-inactive'}">
                           ${status.calibration_applied ? 'ON' : 'OFF'}</span></p>`;
                    
                    if (status.rms_error !== null) {
                        html += `<p class="status-item">RMS誤差: ${status.rms_error.toFixed(4)}</p>`;
                    }
                    
                    html += '</div>';
                    statusPanel.innerHTML = html;
                    
                } catch (e) {
                    console.error('Failed to fetch status:', e);
                    statusPanel.innerHTML = '<p class="status-inactive">エラー: ステータス取得に失敗しました</p>';
                }
            }
            
            // キャリブレーションの切り替え
            toggleBtn.addEventListener('click', async () => {
                try {
                    const response = await fetch('/toggle_calibration');
                    const result = await response.json();
                    
                    if (result.error) {
                        alert('エラー: ' + result.status);
                    }
                    
                    // 状態を更新
                    updateStatus();
                    
                } catch (e) {
                    console.error('Failed to toggle calibration:', e);
                    alert('キャリブレーション切り替え中にエラーが発生しました');
                }
            });
            
            // 初期状態を取得
            updateStatus();
            
            // 定期的に状態を更新
            setInterval(updateStatus, 5000);
        </script>
    </body>
    </html>
    """

def main():
    """アプリケーションを起動"""
    print("カメラとキャリブレーションのWebテストを開始")
    
    # キャリブレーションデータをロード
    calib_file = "calibration_data/calibration.json"
    if os.path.exists(calib_file):
        if calibration.load_calibration(calib_file):
            print(f"キャリブレーションデータを読み込みました: {calib_file}")
        else:
            print("キャリブレーションデータを読み込めませんでした")
    else:
        print("キャリブレーションデータが見つかりません")
        
    print("\nWebサーバーを起動します...")
    print("ブラウザで http://localhost:8080 にアクセスしてください")
    print("終了するには Ctrl+C を押してください\n")
    
    # FastAPIアプリケーションを起動
    uvicorn.run(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nユーザーにより中断されました")
        camera.release()
    except Exception as e:
        import traceback
        print(f"エラー: {str(e)}")
        print(traceback.format_exc())
        camera.release()
