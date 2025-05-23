import cv2
import time
import numpy as np
import threading
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from collections import deque
from contextlib import asynccontextmanager

# Depth Anything用
from depth_processor import DepthProcessor, create_depth_visualization, create_depth_grid_visualization

# FastAPIのライフサイクル管理を最新のasynccontextmanagerに変更
@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    # 起動時の処理
    print("アプリケーション起動: カメラとスレッドを初期化します")
    # スレッドは既にグローバルで開始されているので、ここでは何もしない
    
    yield  # アプリケーション実行中
    
    # 終了時の処理
    print("アプリケーション終了: リソースを解放します")
    try:
        # カメラのクリーンアップ
        if cap is not None:
            cap.release()
            print("カメラリソースを解放しました")
    except Exception as e:
        print(f"終了処理中のエラー: {e}")

# FastAPIアプリケーションをlifespanコンテキストマネージャーで初期化
app = FastAPI(lifespan=lifespan)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# カメラバッファ設定とエラー処理を追加
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファサイズを最小に
cap.set(cv2.CAP_PROP_FPS, 30)        # カメラのFPS設定

# カメラの接続状態を確認
if not cap.isOpened():
    print("エラー: カメラに接続できません")
    import sys
    sys.exit(1)
else:
    print(f"カメラ接続成功: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

depth_processor = DepthProcessor()

# 共有メモリを拡張
latest_depth_map = None
latest_camera_frame = None  # 最新のカメラフレームを保存
frame_timestamp = 0  # フレームのタイムスタンプ
depth_map_lock = threading.Lock()
last_inference_time = 0
INFERENCE_INTERVAL = 0.08  # 0.1→0.08秒に短縮（約12.5FPS）

# カメラキャプチャ専用スレッド（修正）
def camera_capture_thread():
    global latest_camera_frame, frame_timestamp
    consecutive_errors = 0
    max_errors = 5
    
    while True:
        try:
            ret, frame = cap.read()
            if ret:
                with depth_map_lock:
                    latest_camera_frame = frame.copy()
                    frame_timestamp = time.time()
                consecutive_errors = 0  # エラーカウンタをリセット
            else:
                consecutive_errors += 1
                print(f"カメラ読み取りエラー ({consecutive_errors}/{max_errors})")
                
                if consecutive_errors >= max_errors:
                    print("カメラをリセットします...")
                    cap.release()
                    time.sleep(1.0)
                    cap.open(0)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    consecutive_errors = 0
        except Exception as e:
            print(f"カメラ例外: {e}")
            time.sleep(0.5)
            
        time.sleep(0.05)  # 20FPSを維持

# 処理時間計測用
camera_times = deque(maxlen=1000)
inference_times = deque(maxlen=1000)
visualization_times = deque(maxlen=1000)
encoding_times = deque(maxlen=1000)

# パフォーマンス統計用の変数を追加
fps_stats = {
    "camera": deque(maxlen=30),
    "depth": deque(maxlen=30),
    "grid": deque(maxlen=30),
    "inference": deque(maxlen=30)
}
last_frame_times = {
    "camera": 0,
    "depth": 0,
    "grid": 0,
}

def log_processing_times():
    """5秒ごとに平均、最大、最小の処理時間をログ出力"""
    while True:
        time.sleep(5)
        if camera_times:
            print(f"[Camera] Avg: {np.mean(camera_times):.4f}s, Max: {np.max(camera_times):.4f}s, Min: {np.min(camera_times):.4f}s")
        if inference_times:
            print(f"[Inference] Avg: {np.mean(inference_times):.4f}s, Max: {np.max(inference_times):.4f}s, Min: {np.min(inference_times):.4f}s")
        if visualization_times:
            print(f"[Visualization] Avg: {np.mean(visualization_times):.4f}s, Max: {np.max(visualization_times):.4f}s, Min: {np.min(visualization_times):.4f}s")
        if encoding_times:
            print(f"[Encoding] Avg: {np.mean(encoding_times):.4f}s, Max: {np.max(encoding_times):.4f}s, Min: {np.min(encoding_times):.4f}s")

threading.Thread(target=log_processing_times, daemon=True).start()

# 推論専用の関数を追加
def inference_thread():
    global latest_depth_map, last_inference_time
    while True:
        current_time = time.time()
        # 前回の推論から一定時間経過した場合のみ推論実行
        if current_time - last_inference_time > INFERENCE_INTERVAL:
            # 共有メモリからカメラフレームを取得
            with depth_map_lock:
                if latest_camera_frame is None:
                    time.sleep(0.01)
                    continue
                frame = latest_camera_frame.copy()
                capture_time = frame_timestamp
            
            # 推論用にリサイズ
            small = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
            start_time = time.perf_counter()
            depth_map, _ = depth_processor.predict(small)
            inference_time = time.perf_counter() - start_time
            inference_times.append(inference_time)
            
            # FPS計算
            now = time.time()
            if last_inference_time > 0:
                fps = 1.0 / (now - last_inference_time)
                fps_stats["inference"].append(fps)
            
            # ロックを取得して共有メモリを更新
            with depth_map_lock:
                latest_depth_map = depth_map
                last_inference_time = now
            
            # 遅延を計算して表示
            delay = now - capture_time
            print(f"[Thread] Inference completed in {inference_time:.4f}s, Delay: {delay*1000:.1f}ms")
        else:
            # 推論間隔が来るまで少し待機
            time.sleep(0.01)  # 0.05→0.01に短縮して応答性を向上

# カメラスレッド起動
threading.Thread(target=camera_capture_thread, daemon=True).start()

# 推論スレッド起動
threading.Thread(target=inference_thread, daemon=True).start()

def get_depth_stream():
    while True:
        # 共有メモリからカメラフレームと深度マップを取得
        with depth_map_lock:
            if latest_depth_map is None or latest_camera_frame is None:
                time.sleep(0.01)
                continue
            current_depth_map = latest_depth_map.copy()
            current_frame = latest_camera_frame.copy()  # 現在のフレームも取得

        # 深度マップの可視化
        start_time = time.perf_counter()
        vis = create_depth_visualization(current_depth_map, (128, 128))
        vis = cv2.resize(vis, (320, 240), interpolation=cv2.INTER_NEAREST)
        visualization_times.append(time.perf_counter() - start_time)

        # FPS計算とテキスト表示
        now = time.time()
        if last_frame_times["depth"] > 0:
            fps = 1.0 / (now - last_frame_times["depth"])
            fps_stats["depth"].append(fps)
        last_frame_times["depth"] = now

        # FPS表示と遅延表示
        if len(fps_stats["depth"]) > 0:
            avg_fps = sum(fps_stats["depth"]) / len(fps_stats["depth"])
            cv2.putText(vis, f"FPS: {avg_fps:.1f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 最新の遅延情報を表示
            with depth_map_lock:
                delay = (time.time() - frame_timestamp) * 1000
            cv2.putText(vis, f"Delay: {delay:.1f}ms", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # JPEG エンコード
        start_time = time.perf_counter()
        ret, buffer = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        encoding_times.append(time.perf_counter() - start_time)
        if not ret:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.015)  # 約66FPSに向上（0.02→0.015に変更）

def get_depth_grid_stream():
    while True:
        # 共有メモリからカメラフレームと深度マップを取得
        with depth_map_lock:
            if latest_depth_map is None or latest_camera_frame is None:
                time.sleep(0.01)
                continue
            current_depth_map = latest_depth_map.copy()
            current_frame = latest_camera_frame.copy()  # 現在のフレームも取得

        # グリッドの可視化
        start_time = time.perf_counter()
        grid_img = create_depth_grid_visualization(current_depth_map, grid_size=(10, 10), cell_size=18)
        if grid_img is None or len(grid_img.shape) < 2:
            grid_img = 128 * np.ones((240, 320, 3), dtype=np.uint8)
        elif len(grid_img.shape) == 2 or (len(grid_img.shape) == 3 and grid_img.shape[2] == 1):
            grid_img = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)
        grid_img = cv2.resize(grid_img, (320, 240), interpolation=cv2.INTER_NEAREST)
        visualization_times.append(time.perf_counter() - start_time)

        # FPS計算とテキスト表示
        now = time.time()
        if last_frame_times["grid"] > 0:
            fps = 1.0 / (now - last_frame_times["grid"])
            fps_stats["grid"].append(fps)
        last_frame_times["grid"] = now

        # FPS表示と遅延表示
        if len(fps_stats["grid"]) > 0:
            avg_fps = sum(fps_stats["grid"]) / len(fps_stats["grid"])
            cv2.putText(grid_img, f"FPS: {avg_fps:.1f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 最新の遅延情報を表示
            with depth_map_lock:
                delay = (time.time() - frame_timestamp) * 1000
            cv2.putText(grid_img, f"Delay: {delay:.1f}ms", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # JPEG エンコード
        start_time = time.perf_counter()
        ret, buffer = cv2.imencode('.jpg', grid_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        encoding_times.append(time.perf_counter() - start_time)
        if not ret:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.015)  # 約66FPSに向上（0.02→0.015に変更）

@app.get("/stats")
async def get_stats():
    """統計情報を取得するAPIエンドポイント"""
    # 共有メモリからフレームタイムスタンプを取得して遅延を計算
    with depth_map_lock:
        current_delay = (time.time() - frame_timestamp) * 1000 if frame_timestamp > 0 else 0
        
    # 中央値を計算するヘルパー関数
    def median(values):
        if not values:
            return 0
        values_list = list(values)
        values_list.sort()
        return values_list[len(values_list) // 2]
        
    stats = {
        "fps": {
            # 平均値の代わりに中央値を使用
            "camera": round(median(fps_stats["camera"]), 1) if fps_stats["camera"] else 0,
            "depth": round(median(fps_stats["depth"]), 1) if fps_stats["depth"] else 0,
            "grid": round(median(fps_stats["grid"]), 1) if fps_stats["grid"] else 0,
            "inference": round(median(fps_stats["inference"]), 1) if fps_stats["inference"] else 0,
        },
        "latency": {
            "camera": round(np.mean(camera_times) * 1000, 1) if camera_times else 0,
            "inference": round(np.mean(inference_times) * 1000, 1) if inference_times else 0,
            "visualization": round(np.mean(visualization_times) * 1000, 1) if visualization_times else 0,
            "encoding": round(np.mean(encoding_times) * 1000, 1) if encoding_times else 0,
            "total_delay": round(current_delay, 1)  # 現在の総遅延を追加
        }
    }
    return stats

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head>
        <title>Fast Camera Streaming</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
            .container { display: flex; flex-wrap: wrap; gap: 15px; }
            .video-box { background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); position: relative; }
            h2 { margin-top: 0; color: #333; }
            .stats { margin-top: 20px; padding: 10px; background: #e8f5e9; border-radius: 5px; }
            #stats-container { font-family: monospace; }
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
        </div>
        <div class="stats">
            <h3>Performance Stats</h3>
            <div id="stats-container">Loading stats...</div>
        </div>
        
        <script>
            // 2秒ごとに統計情報を更新
            setInterval(async () => {
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
            }, 2000);
        </script>
    </body>
    </html>
    """

@app.get("/video")
async def video():
    return StreamingResponse(get_camera_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/depth_video")
async def depth_video():
    return StreamingResponse(get_depth_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/depth_grid")
async def depth_grid():
    return StreamingResponse(get_depth_grid_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

def get_camera_stream():
    # ストリーム開始時のタイムスタンプをリセット
    last_frame_times["camera"] = 0
    fps_stats["camera"].clear()  # FPS統計をクリア
    first_frame = True  # 最初のフレームかどうかを追跡
    
    while True:
        # 共有メモリからカメラフレームを取得
        with depth_map_lock:
            if latest_camera_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_camera_frame.copy()
        
        # FPS計算 - 改善版
        now = time.time()
        if first_frame:
            # 最初のフレームはFPS計算をスキップ、タイムスタンプのみ記録
            first_frame = False
        elif (now - last_frame_times["camera"]) < 0.5:  # 0.5秒以内の正常な間隔
            # 正常な間隔の場合のみFPSを計算
            fps = 1.0 / (now - last_frame_times["camera"])
            # 異常値フィルタリング (FPSが200を超える値はエラーと見なす)
            if fps < 200:  
                fps_stats["camera"].append(fps)
        
        # 現在時刻を常に記録
        last_frame_times["camera"] = now

        # 画面上にFPS表示
        if len(fps_stats["camera"]) > 0:
            # 中央値を使用 (平均値より外れ値の影響を受けにくい)
            camera_fps_values = list(fps_stats["camera"])
            camera_fps_values.sort()
            median_fps = camera_fps_values[len(camera_fps_values) // 2]
            cv2.putText(frame, f"FPS: {median_fps:.1f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # エンコーディング
        start_time = time.perf_counter()
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        encoding_times.append(time.perf_counter() - start_time)
        if not ret:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)  # 20FPSに制限（0.02→0.05に変更）

# 例外ハンドリングを強化
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    error_msg = f"予期せぬエラーが発生しました: {str(exc)}"
    print(f"[エラー] {error_msg}")
    import traceback
    print(traceback.format_exc())
    
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"detail": error_msg}
    )

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8888)
    except KeyboardInterrupt:
        print("Ctrl+Cが押されました。アプリケーションを終了します。")
    except Exception as e:
        print(f"予期せぬエラーでアプリケーションが終了しました: {e}")
        import traceback
        print(traceback.format_exc())