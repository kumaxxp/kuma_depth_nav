"""
キャリブレーション機能のWebルートを管理するモジュール
"""
import os
import cv2
import time
import numpy as np
import io
from fastapi import APIRouter, HTTPException, Response, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from ..calibration.calibration import calibration
from ..calibration.app_ui import calibration_app
from ..camera.capture import camera

# ルーターを作成
calibration_router = APIRouter(prefix="/calibration")

@calibration_router.get("/", response_class=HTMLResponse)
async def calibration_ui():
    """キャリブレーションUIのHTMLを返す"""
    return get_calibration_html()

@calibration_router.get("/status")
async def get_status():
    """キャリブレーションの状態を取得"""
    status = {
        "is_capturing": calibration_app.is_capturing,
        "is_calibrating": calibration_app.is_calibrating,
        "calibration_success": calibration_app.calibration_success,
        "image_count": len(calibration_app.captured_images),
        "status_text": calibration_app.status_text,
        "has_comparison_view": calibration_app.comparison_view is not None,
    }
    
    # カメラキャリブレーション状態を追加
    status["camera_calibration_applied"] = camera.use_calibration
    
    # キャリブレーションデータの存在を確認
    calib_file = "calibration_data/calibration.json"
    status["calibration_file_exists"] = os.path.exists(calib_file)
    
    # キャリブレーションメトリクスを追加
    if calibration.rms_error is not None:
        status["rms_error"] = calibration.rms_error
    
    return status

@calibration_router.post("/capture")
async def capture_images(background_tasks: BackgroundTasks, num_images: int = 10, delay_seconds: int = 2):
    """キャリブレーション用の画像をキャプチャ
    
    Args:
        num_images: キャプチャする画像数
        delay_seconds: キャプチャ間の遅延（秒）
    """
    if calibration_app.is_capturing:
        raise HTTPException(status_code=400, detail="既にキャプチャ中です")
    
    # バックグラウンドタスクとしてキャプチャを実行
    background_tasks.add_task(calibration_app.capture_images, num_images, delay_seconds)
    
    return {"message": f"画像キャプチャを開始しました。{num_images}枚、間隔{delay_seconds}秒"}

@calibration_router.post("/run")
async def run_calibration(background_tasks: BackgroundTasks):
    """キャリブレーションを実行"""
    if calibration_app.is_calibrating:
        raise HTTPException(status_code=400, detail="既にキャリブレーション中です")
    
    if len(calibration_app.captured_images) < 5:
        raise HTTPException(status_code=400, detail="キャリブレーションには少なくとも5枚の画像が必要です")
    
    # バックグラウンドタスクとしてキャリブレーションを実行
    background_tasks.add_task(calibration_app.run_calibration)
    
    return {"message": "キャリブレーションを開始しました"}

@calibration_router.post("/load")
async def load_images(background_tasks: BackgroundTasks):
    """フォルダから画像を読み込む"""
    background_tasks.add_task(calibration_app.load_images_from_folder)
    return {"message": "画像の読み込みを開始しました"}

@calibration_router.post("/apply")
async def apply_calibration():
    """キャリブレーションをカメラに適用"""
    if not calibration_app.calibration_success:
        raise HTTPException(status_code=400, detail="キャリブレーションが成功していません")
    
    result = calibration_app.apply_calibration_to_camera()
    
    return {
        "success": result,
        "message": "キャリブレーションを適用しました" if result else "キャリブレーションの適用に失敗しました"
    }

@calibration_router.get("/comparison")
async def get_comparison():
    """キャリブレーション前後の比較画像を取得"""
    try:
        if calibration_app.comparison_view is None:
            # 比較画像がない場合はダミー画像を生成
            dummy_img = np.zeros((240, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_img, "No Comparison Image Available", (120, 120),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', dummy_img)
        else:
            ret, buffer = cv2.imencode('.jpg', calibration_app.comparison_view)
            
        if not ret:
            raise HTTPException(status_code=500, detail="画像のエンコードに失敗しました")
        
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
    except Exception as e:
        # エラー時もダミー画像を返す
        dummy_img = np.zeros((240, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_img, f"Error: {str(e)}", (10, 120),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', dummy_img)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

@calibration_router.get("/live_view")
async def get_live_view():
    """キャリブレーションのライブビューストリーム（元の画像と補正後の画像の並べて表示）"""
    def generate():
        while True:
            try:
                # カメラフレームを取得
                frame, _ = camera.get_frame()
                if frame is None:
                    # ダミー画像を生成
                    frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(frame, "Camera Not Available", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    time.sleep(0.1)
                
                # 元画像をコピー
                original_frame = frame.copy()
                
                try:
                    # キャリブレーションが有効な場合、補正された画像を取得
                    if calibration.camera_matrix is not None and calibration.dist_coeffs is not None:
                        corrected_frame = calibration.undistort_image(frame)
                    else:
                        # キャリブレーションがない場合は同じ画像を使用
                        corrected_frame = frame.copy()
                        cv2.putText(corrected_frame, "No Calibration", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except Exception as e:
                    print(f"補正エラー: {e}")
                    corrected_frame = frame.copy()
                    cv2.putText(corrected_frame, "Calibration Error", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 2つの画像を並べて表示
                h, w = frame.shape[:2]
                comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
                comparison[:, :w] = original_frame
                comparison[:, w:] = corrected_frame
                
                # テキスト追加
                cv2.putText(comparison, "元画像", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(comparison, "補正後", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # エンコーディング
                ret, buffer = cv2.imencode('.jpg', comparison, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if not ret:
                    time.sleep(0.01)
                    continue
                
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.05)  # 20FPS
                
            except Exception as e:
                print(f"ライブビューエラー: {e}")
                time.sleep(0.1)
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

def get_calibration_html():
    """キャリブレーションUIのHTMLを返す"""
    return """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>カメラキャリブレーション</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background: #f0f0f0;
                color: #333;
            }
            h1, h2 {
                color: #2c3e50;
            }
            .container {
                display: flex;
                flex-direction: column;
                gap: 20px;
                max-width: 1200px;
                margin: 0 auto;
            }
            .card {
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .controls {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin-bottom: 15px;
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
            button:disabled {
                background: #95a5a6;
                cursor: not-allowed;
            }
            button:hover:not(:disabled) {
                background: #2980b9;
            }
            .status-panel {
                background: #ecf0f1;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 15px;
            }
            .status-item {
                margin-bottom: 5px;
            }
            .status-success {
                color: #27ae60;
                font-weight: bold;
            }
            .status-warning {
                color: #f39c12;
                font-weight: bold;
            }
            .status-error {
                color: #e74c3c;
                font-weight: bold;
            }
            .image-preview {
                width: 100%;
                max-width: 1000px;
                margin-top: 10px;
                border-radius: 5px;
            }
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(0, 0, 0, 0.3);
                border-radius: 50%;
                border-top-color: #3498db;
                animation: spin 1s ease-in-out infinite;
                vertical-align: middle;
                margin-right: 10px;
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            .hidden {
                display: none;
            }
            .input-group {
                margin-bottom: 10px;
            }
            label {
                display: inline-block;
                width: 200px;
                margin-right: 10px;
            }
            input {
                padding: 8px;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
            a.back-button {
                display: inline-block;
                margin-bottom: 20px;
                color: #3498db;
                text-decoration: none;
            }
            a.back-button:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="back-button">← メインページに戻る</a>
            
            <h1>カメラキャリブレーション</h1>
            
            <div class="card">
                <h2>ライブビュー</h2>
                <p>左: 元画像、右: キャリブレーション適用後</p>
                <img src="/calibration/live_view" alt="ライブビュー" class="image-preview" />
            </div>
            
            <div class="card">
                <h2>キャリブレーション状態</h2>
                <div class="status-panel" id="status-panel">
                    <div class="status-item">ステータス: <span id="status-text">読み込み中...</span></div>
                    <div class="status-item">キャプチャした画像: <span id="image-count">0</span>枚</div>
                    <div class="status-item">キャリブレーション: <span id="calibration-status">未実行</span></div>
                    <div class="status-item">RMS誤差: <span id="rms-error">-</span></div>
                    <div class="status-item">カメラ適用状態: <span id="camera-applied">未適用</span></div>
                </div>
            </div>
            
            <div class="card">
                <h2>キャリブレーション手順</h2>
                <ol>
                    <li>「画像キャプチャ」ボタンをクリックし、チェッカーボードを様々な角度で撮影</li>
                    <li>十分な枚数（最低5枚）キャプチャしたら「キャリブレーション実行」ボタンをクリック</li>
                    <li>キャリブレーションが成功したら「カメラに適用」ボタンをクリック</li>
                </ol>
                
                <div class="controls">
                    <div class="input-group">
                        <label for="num-images">キャプチャ枚数:</label>
                        <input type="number" id="num-images" min="5" max="30" value="10">
                    </div>
                    <div class="input-group">
                        <label for="delay-seconds">キャプチャ間隔（秒）:</label>
                        <input type="number" id="delay-seconds" min="1" max="10" value="2">
                    </div>
                </div>
                
                <div class="controls">
                    <button id="capture-btn">画像キャプチャ</button>
                    <button id="load-btn">画像読み込み</button>
                    <button id="calibration-btn">キャリブレーション実行</button>
                    <button id="apply-btn">カメラに適用</button>
                </div>
            </div>
            
            <div class="card" id="comparison-card">
                <h2>キャリブレーション結果</h2>
                <p>キャリブレーション前と後の比較画像</p>
                <div id="comparison-container">
                    <p>キャリブレーションを実行すると、ここに結果が表示されます</p>
                </div>
            </div>
        </div>
        
        <script>
            // DOM要素の取得
            const statusText = document.getElementById('status-text');
            const imageCount = document.getElementById('image-count');
            const calibrationStatus = document.getElementById('calibration-status');
            const rmsError = document.getElementById('rms-error');
            const cameraApplied = document.getElementById('camera-applied');
            const captureBtn = document.getElementById('capture-btn');
            const loadBtn = document.getElementById('load-btn');
            const calibrationBtn = document.getElementById('calibration-btn');
            const applyBtn = document.getElementById('apply-btn');
            const numImages = document.getElementById('num-images');
            const delaySeconds = document.getElementById('delay-seconds');
            const comparisonContainer = document.getElementById('comparison-container');
            const comparisonCard = document.getElementById('comparison-card');
            
            // ステータス更新関数
            async function updateStatus() {
                try {
                    const response = await fetch('/calibration/status');
                    const data = await response.json();
                    
                    statusText.textContent = data.status_text;
                    imageCount.textContent = data.image_count;
                    calibrationStatus.textContent = data.calibration_success ? '成功' : '未実行';
                    calibrationStatus.className = data.calibration_success ? 'status-success' : 'status-warning';
                    
                    if (data.rms_error !== undefined) {
                        rmsError.textContent = data.rms_error.toFixed(4);
                    } else {
                        rmsError.textContent = '-';
                    }
                    
                    cameraApplied.textContent = data.camera_calibration_applied ? '適用中' : '未適用';
                    cameraApplied.className = data.camera_calibration_applied ? 'status-success' : 'status-warning';
                    
                    // ボタンの有効/無効状態を設定
                    captureBtn.disabled = data.is_capturing;
                    loadBtn.disabled = data.is_capturing;
                    calibrationBtn.disabled = data.is_calibrating || data.image_count < 5;
                    applyBtn.disabled = !data.calibration_success;
                    
                    // 比較画像を表示
                    if (data.has_comparison_view && !comparisonContainer.querySelector('img')) {
                        const img = document.createElement('img');
                        img.src = `/calibration/comparison?t=${new Date().getTime()}`;  // キャッシュ回避
                        img.alt = 'キャリブレーション比較';
                        img.className = 'image-preview';
                        comparisonContainer.innerHTML = '';
                        comparisonContainer.appendChild(img);
                    }
                    
                } catch (error) {
                    console.error('ステータス更新エラー:', error);
                }
            }
            
            // APIリクエスト関数
            async function sendApiRequest(url, method = 'GET', data = null) {
                try {
                    const options = {
                        method: method,
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    };
                    
                    if (data) {
                        options.body = JSON.stringify(data);
                    }
                    
                    const response = await fetch(url, options);
                    return await response.json();
                } catch (error) {
                    console.error('APIリクエストエラー:', error);
                    return { success: false, message: 'エラーが発生しました' };
                }
            }
            
            // ボタンクリックイベント
            captureBtn.addEventListener('click', async () => {
                const count = parseInt(numImages.value);
                const delay = parseInt(delaySeconds.value);
                
                captureBtn.disabled = true;
                statusText.innerHTML = '<span class="loading"></span> 画像キャプチャ中...';
                
                await sendApiRequest(`/calibration/capture?num_images=${count}&delay_seconds=${delay}`, 'POST');
                updateStatus();
            });
            
            loadBtn.addEventListener('click', async () => {
                loadBtn.disabled = true;
                statusText.innerHTML = '<span class="loading"></span> 画像読み込み中...';
                
                await sendApiRequest('/calibration/load', 'POST');
                updateStatus();
            });
            
            calibrationBtn.addEventListener('click', async () => {
                calibrationBtn.disabled = true;
                statusText.innerHTML = '<span class="loading"></span> キャリブレーション処理中...';
                
                await sendApiRequest('/calibration/run', 'POST');
                updateStatus();
                
                // キャリブレーション完了後、結果が表示されるまで少し待つ
                setTimeout(() => {
                    if (comparisonContainer.querySelector('img')) {
                        comparisonContainer.querySelector('img').src = `/calibration/comparison?t=${new Date().getTime()}`;
                    }
                }, 3000);
            });
            
            applyBtn.addEventListener('click', async () => {
                applyBtn.disabled = true;
                statusText.innerHTML = '<span class="loading"></span> キャリブレーション適用中...';
                
                const result = await sendApiRequest('/calibration/apply', 'POST');
                alert(result.message);
                updateStatus();
            });
            
            // 定期的にステータスを更新
            updateStatus();
            setInterval(updateStatus, 2000);
        </script>
    </body>
    </html>
    """

# 必要なインポートを遅延実行（循環インポートを防ぐ）
import cv2
import time
import numpy as np
import io
