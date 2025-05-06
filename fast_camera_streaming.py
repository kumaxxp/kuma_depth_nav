from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import time
import threading
import queue
import os
from contextlib import asynccontextmanager

# カスタムモジュールのインポート
from depth_processor import (
    DepthProcessor, create_depth_visualization, create_default_depth_image,
    create_depth_grid_visualization, convert_to_absolute_depth,  # この行を修正
    depth_to_point_cloud, create_top_down_occupancy_grid, visualize_occupancy_grid
)

# ユーティリティのインポート
from utils import load_config, setup_logger, optimize_linux_performance

# 設定の読み込み
config = load_config()
camera_config = config["camera"]
depth_config = config["depth"] 
server_config = config["server"]
logging_config = config["logging"]

# ロガーの設定
logger = setup_logger(
    "kuma_depth_nav", 
    logging_config.get("level", "INFO"),
    logging_config.get("file")
)

# Linux最適化を実行
try:
    optimize_linux_performance()
except NameError:
    logger.warning("optimize_linux_performance 関数が見つかりません。Linux最適化はスキップします。")

# グローバル変数
frame_queue = queue.Queue(maxsize=2)  # 最新のフレームだけを保持するキュー
depth_image_queue = queue.Queue(maxsize=1)  # 深度画像キュー
depth_data_queue = queue.Queue(maxsize=1)  # 深度データキュー
topview_image_queue = queue.Queue(maxsize=1)  # トップビュー画像キュー
depth_grid_queue = queue.Queue(maxsize=1)  # 深度グリッド画像キュー

# axengine をインポート (機能チェック用)
try:
    import axengine as axe
    HAS_AXENGINE = True
    logger.info("axengine successfully imported")
except ImportError:
    HAS_AXENGINE = False
    logger.warning("axengine is not installed. Running in basic mode without depth estimation.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフスパン管理"""
    global process_thread, is_running
    
    # 起動時処理
    process_thread = threading.Thread(target=depth_processing_thread, daemon=True)
    process_thread.start()
    logger.info("Started depth processing thread")
    
    yield  # アプリケーション実行中
    
    # 終了時処理
    is_running = False
    if process_thread:
        process_thread.join(timeout=2.0)
    logger.info("Stopped depth processing thread")

app = FastAPI(lifespan=lifespan)

# グローバル変数
process_thread = None
is_running = True
depth_processor = None  # 深度プロセッサ

@app.get("/", response_class=HTMLResponse)
async def root():
    """HTMLページを提供します"""
    # シンプルなHTMLテンプレート
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kuma Depth Navigation</title>
        <style>
            body { 
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f0f0f0;
            }
            .container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(640px, 1fr));
                gap: 20px;
                width: 100%;
            }
            .video-container {
                margin-bottom: 20px;
                background: white;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            h2 {
                color: #333;
                margin-top: 0;
            }
            img {
                width: 100%;
                border-radius: 5px;
            }
            .status {
                margin-top: 10px;
                padding: 10px;
                background-color: #e6f7ff;
                border-left: 4px solid #1890ff;
            }
            .warning {
                background-color: #fff7e6;
                border-left: 4px solid #fa8c16;
                padding: 10px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <h1>Kuma Depth Navigation System</h1>
        <div class="status" id="statusBar">
            System Status: Running
        </div>
        
        <div class="container">
            <div class="video-container">
                <h2>Camera Feed</h2>
                <img src="/video" alt="Camera Feed">
            </div>
            
            <div class="video-container">
                <h2>Depth Map</h2>
                <img src="/depth_video" alt="Depth Map">
            </div>
            
            <div class="video-container">
                <h2>Depth Grid</h2>
                <img src="/depth_grid" alt="Depth Grid">
            </div>
            
            <div class="video-container">
                <h2>Top View</h2>
                <img src="/topview" alt="Top View">
            </div>
        </div>
        
        <script>
            // 画像の読み込み状態をモニタリング
            const images = document.querySelectorAll('img');
            const statusBar = document.getElementById('statusBar');
            
            // 画像の読み込みエラーを監視
            images.forEach(img => {
                img.onerror = function() {
                    this.src = '/static/error.jpg';  // エラー時の画像
                    this.style.border = '2px solid red';
                    statusBar.innerHTML = 'System Status: Error loading some components';
                    statusBar.style.backgroundColor = '#fff1f0';
                    statusBar.style.borderLeft = '4px solid #f5222d';
                };
                
                // 定期的に画像をリロード（負荷軽減のため10秒に1回）
                setInterval(() => {
                    const currentSrc = img.src;
                    if (!img.currentSrc || img.currentSrc.includes('error.jpg')) return;
                    img.src = currentSrc.split('?')[0] + '?' + new Date().getTime();
                }, 10000);  // 10秒ごとにリロード
            });
        </script>
    </body>
    </html>
    """
    
    return html_content

def initialize_camera(index=0, width=640, height=480):
    """カメラを初期化します"""
    cam = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cam

def get_video_stream():
    """ビデオストリームを生成します"""
    camera = initialize_camera()

    try:
        while True:
            if not camera.isOpened():
                logger.warning("Camera not open. Retrying...")
                camera.release()
                time.sleep(0.5)
                camera = initialize_camera()
                continue

            success, frame = camera.read()
            if not success or frame is None:
                print("[WARN] Failed to read frame. Skipping...")
                continue
                
            # フレームをキューに追加（古いフレームは捨てる）
            try:
                if frame_queue.full():
                    # キューがいっぱいなら古いフレームを取り出して捨てる
                    frame_queue.get_nowait()
                frame_queue.put_nowait(frame)
            except:
                pass  # キューの操作でエラーが発生しても無視

            # JPEGエンコード
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("[WARN] JPEG encode failed.")
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.001)  # スリープ時間を短縮

    finally:
        camera.release()

def get_depth_stream():
    """深度画像ストリーム生成関数"""
    while True:
        try:
            # キューが空の場合はデフォルト深度画像を使用
            if depth_image_queue.empty():
                frame = create_default_depth_image()
                logger.debug("Using default depth image")
            else:
                frame = depth_image_queue.get_nowait()
            
            # JPEG にエンコード
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            # HTTP レスポンス形式に変換
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
            time.sleep(0.05)  # ストリームのスロットリング
                
        except Exception as e:
            logger.error(f"Error in depth stream: {e}")
            time.sleep(0.1)  # エラー時は少し長めのウェイト

def get_topview_stream():
    """トップビュー画像ストリームを生成します"""
    try:
        # デフォルトのトップビュー画像（空の占有グリッド）を準備
        default_grid = np.zeros((100, 100), dtype=np.uint8)  # 空の占有グリッド
        default_topview = visualize_occupancy_grid(default_grid)
        
        # 最後に表示した有効な画像を保持
        last_valid_topview = default_topview.copy()
        
        while True:
            try:
                # トップビュー画像があればそれを使用（キューから取り出さずに参照）
                if not topview_image_queue.empty():
                    current_topview = topview_image_queue.queue[0]  # キューから取り出さずに参照
                    # 有効な画像なら最後の有効画像として保存
                    if current_topview is not None and current_topview.shape[0] > 0:
                        last_valid_topview = current_topview  # .copy()が不要な場合も多い
                
                # 前回の有効なトップビュー画像を使用
                topview_image = last_valid_topview
                    
                # JPEGエンコード
                ret, buffer = cv2.imencode('.jpg', topview_image)
                if not ret:
                    logger.warning("JPEG encode failed for topview image.")
                    continue
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.05)  # トップビュー更新は低いフレームレートでOK
                
            except Exception as e:
                print(f"[ERROR] Error in topview stream: {e}")
                time.sleep(0.1)
    
    except Exception as e:
        print(f"[ERROR] Fatal error in topview stream: {e}")

def depth_processing_thread():
    """深度推論を行うスレッド"""
    global is_running, depth_processor
    logger.info("Depth processing thread started")
    
    # 深度処理クラスの初期化
    depth_processor = DepthProcessor()
    
    if not depth_processor.is_available():
        logger.warning("Using dummy depth data (model initialization failed)")
    else:
        logger.info("Depth model initialized successfully")
    
    frame_count = 0
    
    # デバッグ用: 最初のフレームの可視化テスト
    debug_first_frame = True
    
    while is_running:
        try:
            # キューからフレームを取得
            frame = frame_queue.get(timeout=1.0)
            
            # 深度推論実行
            depth_map, inference_time = depth_processor.predict(frame)
            
            # 深度データが有効であることを確認
            if depth_map is None or depth_map.size == 0:
                logger.warning("Empty depth map received. Skipping...")
                continue
                
            # 最初のフレームをファイルに保存（デバッグ用）
            if debug_first_frame:
                try:
                    # 深度マップ統計
                    min_val = np.min(depth_map)
                    max_val = np.max(depth_map)
                    mean_val = np.mean(depth_map)
                    logger.info(f"First depth map - Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}")
                    
                    # 可視化して保存
                    first_vis = create_depth_visualization(depth_map, frame.shape)
                    cv2.imwrite("first_depth_frame.jpg", first_vis)
                    logger.info("First depth frame visualization saved to: first_depth_frame.jpg")
                    
                    # 形状情報
                    logger.info(f"Frame shape: {frame.shape}")
                    logger.info(f"Depth map shape: {depth_map.shape}")
                    
                    debug_first_frame = False
                except Exception as e:
                    logger.error(f"Error saving debug frame: {e}")
            
            # 深度マップを可視化
            logger.debug(f"Creating depth visualization for frame {frame_count}")
            try:
                colored_depth = create_depth_visualization(depth_map, frame.shape)
                
                # 可視化に成功したら深度画像をキューに追加
                if colored_depth is not None and colored_depth.shape[0] > 0:
                    try:
                        if depth_image_queue.full():
                            depth_image_queue.get_nowait()  # 古いデータを削除
                        depth_image_queue.put_nowait(colored_depth)
                        logger.debug(f"Depth visualization added to queue, shape: {colored_depth.shape}")
                    except Exception as e:
                        logger.warning(f"Failed to update depth image queue: {e}")
                else:
                    logger.warning("Empty visualization result")
            except Exception as e:
                logger.error(f"Failed to visualize depth map: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
            # 深度マップから絶対深度に変換
            absolute_depth = convert_to_absolute_depth(
                depth_map, 
                scaling_factor=depth_config.get("scaling_factor", 15.0)
            )

            # 10フレームに1回のみ点群処理を実行
            if frame_count % 10 == 0:
                try:
                    # 深度から点群を生成
                    points = depth_to_point_cloud(
                        absolute_depth,
                        fx=depth_config.get("fx", 500),
                        fy=depth_config.get("fy", 500)
                    )
                    
                    # 占有グリッドを生成
                    occupancy_grid = create_top_down_occupancy_grid(points)
                    
                    # トップビューを可視化
                    topview = visualize_occupancy_grid(occupancy_grid)
                    
                    # キューに追加
                    if not topview_image_queue.full():
                        topview_image_queue.put_nowait(topview)
                        logger.debug("Topview visualization added to queue")
                except Exception as e:
                    logger.error(f"Failed to process point cloud: {e}")
            
            # 定期的にログ出力
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} depth frames, inference time: {inference_time:.3f}s")
                
            # 100フレームごとにデバッグ画像を保存
            if frame_count % 100 == 0:
                try:
                    debug_vis = create_depth_visualization(depth_map, frame.shape)
                    cv2.imwrite(f"depth_frame_{frame_count}.jpg", debug_vis)
                except Exception:
                    pass
                
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Error in depth processing thread: {e}")
            import traceback
            logger.error(traceback.format_exc())
            time.sleep(1.0)  # エラー時は少し長めに待機
    
    logger.info(f"Depth processing thread stopped. Processed {frame_count} frames.")

def get_depth_grid_stream():
    """深度グリッド画像ストリームを生成します"""
    try:
        # デフォルトの深度グリッド画像（空のグリッド）を準備
        default_grid_image = np.zeros((480, 640, 3), dtype=np.uint8) + 30
        
        # グリッド線を描画
        for row in range(1, 6):
            y = row * 80
            cv2.line(default_grid_image, (0, y), (640, y), (100, 100, 100), 1)
        
        for col in range(1, 8):
            x = col * 80
            cv2.line(default_grid_image, (x, 0), (x, 480), (100, 100, 100), 1)
        
        # 最後に表示した有効な画像を保持
        last_valid_grid_image = default_grid_image.copy()
        
        while True:
            try:
                # 深度グリッド画像があればそれを使用（キューから取り出さずに参照）
                if not depth_grid_queue.empty():
                    current_grid_image = depth_grid_queue.queue[0]  # キューから取り出さずに参照
                    # 有効な画像なら最後の有効画像として保存
                    if current_grid_image is not None and current_grid_image.shape[0] > 0:
                        last_valid_grid_image = current_grid_image.copy()
                
                # 前回の有効な深度グリッド画像を使用
                grid_image = last_valid_grid_image
                    
                # JPEGエンコード
                ret, buffer = cv2.imencode('.jpg', grid_image)
                if not ret:
                    print("[WARN] JPEG encode failed for grid image.")
                    continue
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.05)  # 更新は低いフレームレートでOK
                
            except Exception as e:
                print(f"[ERROR] Error in depth grid stream: {e}")
                time.sleep(0.1)
    
    except Exception as e:
        print(f"[ERROR] Fatal error in depth grid stream: {e}")

@app.get("/video")
async def video_endpoint():
    """ビデオストリームのエンドポイント"""
    return StreamingResponse(get_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/depth_video")
async def depth_video_endpoint():
    """深度画像ストリームのエンドポイント"""
    return StreamingResponse(get_depth_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/topview")
async def topview_endpoint():
    """トップビュー画像ストリームのエンドポイント"""
    return StreamingResponse(get_topview_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/depth_grid")
async def depth_grid_endpoint():
    """深度グリッド画像ストリームのエンドポイント"""
    return StreamingResponse(get_depth_grid_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/depth_metrics")
async def depth_metrics():
    """深度推定の統計情報を取得"""
    if not depth_data_queue.empty():
        depth_map = depth_data_queue.queue[0]  # キューからポップせずに参照
        
        # 深度マップの統計情報
        metrics = {
            "min_depth": float(depth_map.min()),
            "max_depth": float(depth_map.max()),
            "mean_depth": float(depth_map.mean()),
            "shape": depth_map.shape,
            "last_inference_time": time.time()
        }
        return metrics
    else:
        return {"error": "No depth data available"}

if __name__ == "__main__":
    import platform
    logger.info(f"システム: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"OpenCV: {cv2.__version__}")
    logger.info(f"設定: カメラ={camera_config.get('device_index', 0)}, ポート={server_config.get('port', 8888)}")
    logger.info(f"axengine利用可能: {HAS_AXENGINE}")
    
    # サーバー起動
    import uvicorn
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8888)
    uvicorn.run(app, host=host, port=port)