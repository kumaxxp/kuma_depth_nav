from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import cv2
import numpy as np
import time
import threading
import queue
import os
import io
import json
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

# カメラキャリブレーションクラスをインポート
from calibration.camera_calibration import CameraCalibration

# FastAPIアプリケーションのライフスパン管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # アプリ起動時の処理
    global is_running, frame_thread
    
    is_running = True
    frame_thread = threading.Thread(target=camera_capture_thread, daemon=True)
    frame_thread.start()
    print("カメラキャプチャスレッドを開始しました")
    
    yield  # アプリケーション実行中
    
    # アプリ終了時の処理
    is_running = False
    if frame_thread:
        frame_thread.join(timeout=1.0)
    print("カメラキャプチャスレッドを停止しました")

app = FastAPI(lifespan=lifespan)

# グローバル変数
is_running = False
frame_thread = None
frame_queue = queue.Queue(maxsize=10)
calibration_frames = []
calibration_instance = CameraCalibration()
calibration_status = {
    "is_calibrating": False,
    "frames_captured": 0,
    "required_frames": 15,
    "message": "キャリブレーションの準備ができました"
}
camera = None

def initialize_camera(camera_id=0, width=640, height=480):
    """カメラを初期化する関数"""
    global camera
    
    if camera is not None and camera.isOpened():
        camera.release()
    
    camera = cv2.VideoCapture(camera_id)
    if not camera.isOpened():
        print(f"カメラID {camera_id} を開けませんでした")
        return False
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return True

def camera_capture_thread():
    """カメラからフレームを取得し続けるスレッド"""
    global is_running, frame_queue, camera
    
    # カメラの初期化
    if not initialize_camera():
        print("カメラの初期化に失敗しました。スレッドを終了します。")
        return
    
    frame_count = 0
    while is_running:
        try:
            success, frame = camera.read()
            if not success:
                print("フレームの取得に失敗しました")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # キューにフレームを追加（古いフレームは捨てる）
            if frame_queue.full():
                frame_queue.get_nowait()
            
            frame_queue.put_nowait(frame)
            
            time.sleep(0.03)  # 約30FPS
            
        except Exception as e:
            print(f"カメラキャプチャエラー: {e}")
            time.sleep(0.1)
    
    # カメラを解放
    if camera.isOpened():
        camera.release()
    print("カメラを解放しました")

@app.get("/", response_class=HTMLResponse)
async def root():
    """メインページのHTMLを返す"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>カメラキャリブレーション</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1, h2, h3 {
                color: #333;
            }
            .camera-view {
                margin: 20px 0;
                text-align: center;
            }
            .camera-view img {
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            .controls {
                display: flex;
                justify-content: space-between;
                margin: 20px 0;
            }
            button {
                padding: 10px 15px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background: #45a049;
            }
            button:disabled {
                background: #cccccc;
                cursor: not-allowed;
            }
            .status {
                padding: 10px;
                margin: 10px 0;
                border-left: 4px solid #2196F3;
                background: #e3f2fd;
            }
            .thumbnails {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin: 20px 0;
            }
            .thumbnail {
                width: 120px;
                height: 90px;
                object-fit: cover;
                border: 1px solid #ddd;
            }
            .results {
                white-space: pre-wrap;
                background: #f8f8f8;
                padding: 15px;
                border-radius: 4px;
                overflow-x: auto;
                font-family: monospace;
            }
            .instructions {
                background: #fff3e0;
                padding: 15px;
                border-left: 4px solid #ff9800;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>カメラキャリブレーション</h1>
            
            <div class="instructions">
                <h3>使い方:</h3>
                <ol>
                    <li>チェスボード(9x6)パターンを様々な角度でカメラに見せます</li>
                    <li>「画像キャプチャ」ボタンを押してチェスボードが見えている画像を15枚程度撮影します</li>
                    <li>十分な枚数をキャプチャしたら「キャリブレーション実行」ボタンを押します</li>
                    <li>キャリブレーション結果を確認して「保存」ボタンで設定を保存します</li>
                </ol>
            </div>
            
            <div class="camera-view">
                <h2>カメラ映像</h2>
                <img src="/video" width="640" height="480" alt="カメラ映像">
            </div>
            
            <div class="status" id="status">
                ステータス: キャリブレーションの準備ができました
            </div>
            
            <div class="controls">
                <button id="captureBtn">画像キャプチャ</button>
                <button id="clearBtn">クリア</button>
                <button id="calibrateBtn" disabled>キャリブレーション実行</button>
                <button id="saveBtn" disabled>保存</button>
                <button id="loadBtn">読み込み</button>
                <button id="undistortBtn" disabled>歪み補正表示</button>
            </div>
            
            <h3>キャプチャ画像 (<span id="frameCount">0</span>/15)</h3>
            <div class="thumbnails" id="thumbnails"></div>
            
            <h3>キャリブレーション結果</h3>
            <div class="results" id="results">
                結果はまだありません
            </div>
        </div>
        
        <script>
            // ステータスの更新
            function updateStatus() {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status').textContent = 'ステータス: ' + data.message;
                        document.getElementById('frameCount').textContent = data.frames_captured;
                        
                        // ボタンの有効/無効状態を更新
                        const calibrateBtn = document.getElementById('calibrateBtn');
                        const saveBtn = document.getElementById('saveBtn');
                        const undistortBtn = document.getElementById('undistortBtn');
                        
                        calibrateBtn.disabled = data.frames_captured < 5 || data.is_calibrating;
                        saveBtn.disabled = !data.has_results;
                        undistortBtn.disabled = !data.has_results;
                        
                        if (data.has_results) {
                            document.getElementById('results').textContent = JSON.stringify(data.results, null, 2);
                        }
                    });
            }
            
            // 定期的にステータスを更新
            setInterval(updateStatus, 1000);
            
            // 画像キャプチャボタン
            document.getElementById('captureBtn').addEventListener('click', () => {
                fetch('/capture', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            // サムネイルを追加
                            const thumbnails = document.getElementById('thumbnails');
                            const img = document.createElement('img');
                            img.src = `/frame/${data.frame_id}?${new Date().getTime()}`;
                            img.className = 'thumbnail';
                            img.alt = `Frame ${data.frame_id}`;
                            thumbnails.appendChild(img);
                            
                            updateStatus();
                        } else {
                            alert('キャプチャ失敗: ' + data.message);
                        }
                    });
            });
            
            // クリアボタン
            document.getElementById('clearBtn').addEventListener('click', () => {
                fetch('/clear', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            document.getElementById('thumbnails').innerHTML = '';
                            document.getElementById('results').textContent = '結果はまだありません';
                            updateStatus();
                        }
                    });
            });
            
            // キャリブレーション実行ボタン
            document.getElementById('calibrateBtn').addEventListener('click', () => {
                fetch('/calibrate', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            document.getElementById('results').textContent = JSON.stringify(data.results, null, 2);
                            alert('キャリブレーション成功! RMS誤差: ' + data.results.rms_error);
                        } else {
                            alert('キャリブレーション失敗: ' + data.message);
                        }
                        updateStatus();
                    });
            });
            
            // 保存ボタン
            document.getElementById('saveBtn').addEventListener('click', () => {
                fetch('/save', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('キャリブレーション設定を保存しました');
                        } else {
                            alert('保存失敗: ' + data.message);
                        }
                    });
            });
            
            // 読み込みボタン
            document.getElementById('loadBtn').addEventListener('click', () => {
                fetch('/load', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            document.getElementById('results').textContent = JSON.stringify(data.results, null, 2);
                            alert('キャリブレーション設定を読み込みました');
                        } else {
                            alert('読み込み失敗: ' + data.message);
                        }
                        updateStatus();
                    });
            });
            
            // 歪み補正表示ボタン
            document.getElementById('undistortBtn').addEventListener('click', () => {
                // 別ウィンドウで歪み補正映像を表示
                window.open('/undistorted', '_blank');
            });
            
            // 初期状態を更新
            updateStatus();
        </script>
    </body>
    </html>
    """
    
    return html_content

@app.get("/video")
async def video_stream():
    """カメラストリームを提供するエンドポイント"""
    return StreamingResponse(generate_video_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/undistorted")
async def undistorted_stream():
    """歪み補正したカメラストリームを提供するHTMLページ"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>歪み補正映像</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                text-align: center;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
            }
            img {
                max-width: 100%;
                border: 1px solid #ddd;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>歪み補正映像</h2>
            <img src="/undistorted_video" alt="歪み補正映像">
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/undistorted_video")
async def undistorted_video_stream():
    """歪み補正したカメラストリームを提供するエンドポイント"""
    return StreamingResponse(generate_undistorted_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
async def get_status():
    """現在のキャリブレーションステータスを返す"""
    global calibration_status, calibration_instance
    
    status_copy = calibration_status.copy()
    status_copy["has_results"] = calibration_instance.camera_matrix is not None
    
    if calibration_instance.camera_matrix is not None:
        status_copy["results"] = {
            "camera_matrix": calibration_instance.camera_matrix.tolist(),
            "dist_coeffs": calibration_instance.dist_coeffs.tolist(),
            "rms_error": calibration_instance.rms_error,
            "frame_size": calibration_instance.frame_size
        }
    
    return status_copy

@app.post("/capture")
async def capture_frame():
    """現在のカメラフレームをキャプチャ"""
    global calibration_frames, calibration_status
    
    try:
        if frame_queue.empty():
            return {"success": False, "message": "カメラフレームがありません"}
        
        # 最新のフレームをコピー
        frame = frame_queue.queue[0].copy()
        
        # チェスボードの検出を試みる
        ret, frame_with_corners, _ = calibration_instance.detect_chessboard(frame)
        
        if ret:
            # 成功したらフレームを保存
            frame_id = len(calibration_frames)
            calibration_frames.append((frame_id, frame))
            
            # ステータス更新
            calibration_status["frames_captured"] = len(calibration_frames)
            calibration_status["message"] = f"フレーム {frame_id} キャプチャ成功 (チェスボード検出)"
            
            return {"success": True, "frame_id": frame_id, "has_chessboard": True}
        else:
            return {"success": False, "message": "チェスボードが検出できませんでした"}
    
    except Exception as e:
        print(f"キャプチャエラー: {e}")
        return {"success": False, "message": str(e)}

@app.get("/frame/{frame_id}")
async def get_frame(frame_id: int):
    """キャプチャしたフレームを取得する"""
    global calibration_frames
    
    try:
        # 指定IDのフレームを検索
        for fid, frame in calibration_frames:
            if fid == frame_id:
                # チェスボードを描画
                ret, vis_frame, _ = calibration_instance.detect_chessboard(frame)
                
                # JPEG画像にエンコード
                ret, buffer = cv2.imencode('.jpg', vis_frame)
                io_buf = io.BytesIO(buffer.tobytes())
                
                return StreamingResponse(io_buf, media_type="image/jpeg")
        
        # 見つからない場合はエラー画像を返す
        error_img = np.zeros((200, 300, 3), dtype=np.uint8)
        error_img[:] = (0, 0, 255)  # 赤い画像
        cv2.putText(error_img, f"Frame {frame_id} not found", (30, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', error_img)
        io_buf = io.BytesIO(buffer.tobytes())
        
        return StreamingResponse(io_buf, media_type="image/jpeg")
        
    except Exception as e:
        print(f"フレーム取得エラー: {e}")
        
        # エラー画像を返す
        error_img = np.zeros((200, 300, 3), dtype=np.uint8)
        error_img[:] = (0, 0, 255)  # 赤い画像
        cv2.putText(error_img, "Error", (100, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', error_img)
        io_buf = io.BytesIO(buffer.tobytes())
        
        return StreamingResponse(io_buf, media_type="image/jpeg")

@app.post("/clear")
async def clear_frames():
    """キャプチャしたフレームをクリアする"""
    global calibration_frames, calibration_status
    
    calibration_frames = []
    calibration_status["frames_captured"] = 0
    calibration_status["message"] = "キャプチャした画像をクリアしました"
    
    return {"success": True}

@app.post("/calibrate")
async def run_calibration():
    """キャリブレーションを実行する"""
    global calibration_frames, calibration_status, calibration_instance
    
    try:
        # 最低5フレームが必要
        if len(calibration_frames) < 5:
            return {
                "success": False, 
                "message": f"最低5フレームが必要です (現在: {len(calibration_frames)})"
            }
        
        # キャリブレーションステータス更新
        calibration_status["is_calibrating"] = True
        calibration_status["message"] = "キャリブレーション実行中..."
        
        # フレームのみを取り出す
        frames = [frame for _, frame in calibration_frames]
        
        # キャリブレーション実行
        success, rms_error, camera_matrix, dist_coeffs = calibration_instance.calibrate(frames)
        
        if success:
            calibration_status["is_calibrating"] = False
            calibration_status["message"] = f"キャリブレーション完了 (RMS誤差: {rms_error:.4f})"
            
            # 結果を返す
            return {
                "success": True,
                "results": {
                    "camera_matrix": camera_matrix.tolist(),
                    "dist_coeffs": dist_coeffs.tolist(),
                    "rms_error": float(rms_error),
                    "frame_size": calibration_instance.frame_size
                }
            }
        else:
            calibration_status["is_calibrating"] = False
            calibration_status["message"] = "キャリブレーション失敗"
            return {"success": False, "message": "キャリブレーションに失敗しました"}
            
    except Exception as e:
        print(f"キャリブレーションエラー: {e}")
        import traceback
        traceback.print_exc()
        
        calibration_status["is_calibrating"] = False
        calibration_status["message"] = f"エラー: {str(e)}"
        
        return {"success": False, "message": str(e)}

@app.post("/save")
async def save_calibration():
    """キャリブレーション結果を保存"""
    global calibration_instance
    
    try:
        if calibration_instance.camera_matrix is None:
            return {"success": False, "message": "キャリブレーション結果がありません"}
        
        # 設定ディレクトリ作成
        os.makedirs("calibration_data", exist_ok=True)
        
        # JSON形式で保存
        filepath = "calibration_data/calibration.json"
        success = calibration_instance.save_calibration(filepath)
        
        if success:
            return {"success": True, "filepath": filepath}
        else:
            return {"success": False, "message": "保存中にエラーが発生しました"}
            
    except Exception as e:
        print(f"保存エラー: {e}")
        return {"success": False, "message": str(e)}

@app.post("/load")
async def load_calibration():
    """保存されたキャリブレーション結果を読み込む"""
    global calibration_instance, calibration_status
    
    try:
        filepath = "calibration_data/calibration.json"
        
        # JSONファイルが無い場合はPickleを試す
        if not os.path.exists(filepath):
            alternate_filepath = "calibration_data/calibration.pkl"
            if not os.path.exists(alternate_filepath):
                return {"success": False, "message": "キャリブレーションファイルが見つかりません"}
            filepath = alternate_filepath
        
        # 設定を読み込む
        success = calibration_instance.load_calibration(filepath)
        
        if success:
            # ステータス更新
            calibration_status["message"] = "キャリブレーション設定を読み込みました"
            
            return {
                "success": True,
                "results": {
                    "camera_matrix": calibration_instance.camera_matrix.tolist(),
                    "dist_coeffs": calibration_instance.dist_coeffs.tolist(),
                    "rms_error": calibration_instance.rms_error,
                    "frame_size": calibration_instance.frame_size
                }
            }
        else:
            return {"success": False, "message": "読み込み失敗"}
            
    except Exception as e:
        print(f"読み込みエラー: {e}")
        return {"success": False, "message": str(e)}

def generate_video_frames():
    """カメラフレームを生成するジェネレータ"""
    while True:
        if frame_queue.empty():
            # 空のフレームを生成（黒画像）
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not available", (100, 240), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # キューから最新フレームを取得（取り出さず参照）
            frame = frame_queue.queue[0].copy()

            # チェスボード検出を試みる（エッジで）
            ret, vis_frame, _ = calibration_instance.detect_chessboard(frame)
            if ret:
                frame = vis_frame
                
                # チェスボード検出成功のメッセージを追加
                cv2.putText(frame, "Chessboard Detected", (20, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # 検出失敗のメッセージを追加
                cv2.putText(frame, "No Chessboard", (20, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # JPEG画像にエンコード
        ret, buffer = cv2.imencode('.jpg', frame)
        
        # HTTP形式でフレームを送信
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # フレームレート調整
        time.sleep(0.033)  # 約30fps

def generate_undistorted_frames():
    """歪み補正したカメラフレームを生成するジェネレータ"""
    global calibration_instance
    
    while True:
        if frame_queue.empty():
            # 空のフレームを生成（黒画像）
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not available", (100, 240), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # キューから最新フレームを取得
            frame = frame_queue.queue[0].copy()
            
            # キャリブレーション結果があれば歪み補正を適用
            if calibration_instance.camera_matrix is not None:
                frame = calibration_instance.undistort_image(frame)
                
                # 補正済みの表示を追加
                cv2.putText(frame, "Undistorted", (20, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # キャリブレーション結果がない場合
                cv2.putText(frame, "No calibration data", (20, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # JPEG画像にエンコード
        ret, buffer = cv2.imencode('.jpg', frame)
        
        # HTTP形式でフレームを送信
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # フレームレート調整
        time.sleep(0.033)  # 約30fps

# サーバー起動コード（直接実行時のみ）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
