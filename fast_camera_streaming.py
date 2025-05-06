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
depth_grid_image_queue = queue.Queue(maxsize=2)  # 深度グリッド用のキュー

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
camera_instance = None  # カメラインスタンスをグローバルに保持

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

def initialize_camera(index=0, width=640, height=480, force_new=False):
    """カメラを初期化します"""
    global camera_instance
    
    # 既に初期化済みのカメラがあり、force_newがFalseなら再利用
    if camera_instance is not None and camera_instance.isOpened() and not force_new:
        logger.info("Reusing existing camera instance")
        return camera_instance
    
    # 古いインスタンスがあれば解放
    if camera_instance is not None:
        camera_instance.release()
        camera_instance = None
    
    try:
        # プラットフォームに応じた初期化
        import platform
        if platform.system() == "Linux":
            logger.info(f"Initializing camera with V4L2: index {index}")
            cam = cv2.VideoCapture(index, cv2.CAP_V4L2)
        else:
            logger.info(f"Initializing camera with default API: index {index}")
            cam = cv2.VideoCapture(index)
        
        if not cam.isOpened():
            logger.error(f"Failed to open camera at index {index}")
            return None
        
        # 設定
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # グローバル変数に保存
        camera_instance = cam
        
        return cam
    except Exception as e:
        logger.error(f"Camera initialization error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def get_video_stream():
    """ビデオストリームを生成します"""
    camera = initialize_camera()
    frame_counter = 0

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
                logger.warning("Failed to read frame. Skipping...")
                time.sleep(0.1)
                continue
                

            frame_counter += 1
                

            # フレームをキューに追加（古いフレームは捨てる）
            try:
                if frame_queue.full():
                    # キューがいっぱいなら古いフレームを取り出して捨てる
                    frame_queue.get_nowait()
                frame_queue.put_nowait(frame)
                
                # 定期的にキュー状態を確認
                if frame_counter % 100 == 0:
                    logger.info(f"Frame queue size: {frame_queue.qsize()}/{frame_queue.maxsize}")
            except Exception as e:
                logger.warning(f"Queue operation error: {e}")

            # JPEGエンコード
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logger.warning("JPEG encode failed.")
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # フレームレート制限
            if frame_counter % 2 == 0:  # 2フレームに1回だけ送信
                time.sleep(0.05)  # 約20FPS
            else:
                time.sleep(0.001)  # 高速処理（キューに追加するため）

    finally:
        camera.release()
        logger.info("Camera released")

def get_depth_stream():
    """深度画像ストリーム生成関数"""
    # 前回の有効な深度マップを保持
    last_valid_depth_image = create_default_depth_image()
    
    while True:
        try:
            # キューが空の場合は前回の有効な深度画像を使用
            if depth_image_queue.empty():
                frame = last_valid_depth_image  # デフォルト画像ではなく前回の有効な画像を使用
                logger.debug("Using last valid depth image")
            else:
                # 新しい深度画像を取得
                frame = depth_image_queue.get_nowait()
                
                # 有効な画像の場合のみ保存（サイズチェック）
                if frame is not None and frame.shape[0] > 0:
                    last_valid_depth_image = frame.copy()  # 有効な画像を保存
            
            # JPEG にエンコード
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logger.warning("JPEG encode failed for depth image.")
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
        frame_count = 0
        
        while True:
            try:
                frame_count += 1
                
                # 5フレームごとにキューから取り出す（更新を強制）
                if frame_count % 5 == 0 and not topview_image_queue.empty():
                    current_topview = topview_image_queue.get_nowait()  # キューから取り出す
                    logger.debug("Retrieved new topview from queue")
                # 通常フレームでは参照のみ
                elif not topview_image_queue.empty():
                    current_topview = topview_image_queue.queue[0]  # 参照のみ
                else:
                    current_topview = None
                
                # 有効な画像なら最後の有効画像として保存
                if current_topview is not None and current_topview.shape[0] > 0:
                    last_valid_topview = current_topview.copy()  # コピーして保存
                
                # 前回の有効なトップビュー画像を使用
                topview_image = last_valid_topview
                     
                # JPEGエンコード
                ret, buffer = cv2.imencode('.jpg', topview_image)
                if not ret:
                    logger.warning("JPEG encode failed for topview image.")
                    continue
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)  # 更新間隔を0.05から0.1に延長
                
            except Exception as e:
                logger.error(f"Error in topview stream: {e}")  # printからloggerに変更
                time.sleep(0.1)
    
    except Exception as e:
        logger.error(f"Fatal error in topview stream: {e}")  # printからloggerに変更

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
    process_count = 0
    process_every_n_frames = 1  # 3フレームに1回だけ処理
    last_log_time = time.time()
    
    # 前回の有効な深度マップと可視化結果を保存
    last_valid_depth_map = None
    last_valid_colored_depth = None
    
    while is_running:
        try:
            # キューからフレームを取得
            frame = frame_queue.get(timeout=1.0)
            frame_count += 1
            
            # N フレームに1回だけ処理
            if frame_count % process_every_n_frames != 0:
                continue
                
            process_count += 1
            # 処理開始時間を記録
            start_time = time.time()
            
            # 深度推論実行
            current_depth_map, inference_time = depth_processor.predict(frame)
            
            # 深度マップが有効かチェック
            is_valid_depth = (current_depth_map is not None and current_depth_map.size > 0)
            
            # 有効でない場合、前回の有効なマップを使用
            if not is_valid_depth:
                logger.warning("Invalid depth map received. Using last valid map.")
                if last_valid_depth_map is not None:
                    current_depth_map = last_valid_depth_map
                else:
                    # まだ有効なマップがない場合はスキップ
                    logger.warning("No previous valid depth map available. Skipping...")
                    continue
            else:
                # 有効なマップを保存
                last_valid_depth_map = current_depth_map.copy()
            
            # 深度マップを可視化
            try:
                current_colored_depth = create_depth_visualization(current_depth_map, frame.shape)
                
                # 可視化が有効かチェック
                is_valid_visualization = (current_colored_depth is not None and 
                                         current_colored_depth.shape[0] > 0)
                
                if is_valid_visualization:
                    # 有効な可視化結果を保存
                    last_valid_colored_depth = current_colored_depth.copy()
                    
                    # キューに追加
                    try:
                        if depth_image_queue.full():
                            depth_image_queue.get_nowait()
                        depth_image_queue.put_nowait(current_colored_depth)
                    except Exception as e:
                        logger.warning(f"Failed to update depth image queue: {e}")
                else:
                    # 無効な場合、前回の有効な可視化結果を使用
                    logger.warning("Invalid depth visualization. Using last valid one.")
                    if last_valid_colored_depth is not None:
                        try:
                            if depth_image_queue.full():
                                depth_image_queue.get_nowait()
                            depth_image_queue.put_nowait(last_valid_colored_depth)
                        except Exception as e:
                            logger.warning(f"Failed to update depth image queue with previous result: {e}")
            except Exception as e:
                logger.error(f"Error in depth visualization: {e}")
                # エラー時も前回の有効結果を使用
                if last_valid_colored_depth is not None:
                    try:
                        if depth_image_queue.full():
                            depth_image_queue.get_nowait()
                        depth_image_queue.put_nowait(last_valid_colored_depth)
                    except Exception as ex:
                        logger.warning(f"Queue operation failed: {ex}")
            
            # 深度マップから絶対深度に変換
            absolute_depth = convert_to_absolute_depth(
                current_depth_map, 
                scaling_factor=depth_config.get("scaling_factor", 15.0)
            )

            # 深度グリッドの生成を追加
            try:
                # 深度の範囲を確認
                valid_depths = absolute_depth[absolute_depth > 0.1]
                if valid_depths.size > 0:
                    min_abs_depth = valid_depths.min()
                    max_abs_depth = valid_depths.max()
                    logger.info(f"Absolute depth range: min={min_abs_depth:.2f}m, max={max_abs_depth:.2f}m")
                
                # 深度グリッド生成（絶対深度を渡す）
                depth_grid = create_depth_grid_visualization(
                    current_depth_map,
                    absolute_depth=absolute_depth,  # 絶対深度を渡す
                    grid_size=(12, 16),  # 12x16のグリッド
                    max_distance=max(10.0, max_abs_depth * 1.2),  # 動的に最大距離を設定
                    cell_size=60        # セルサイズ60ピクセル
                )
                
                # 深度グリッドをキューに追加
                try:
                    if depth_grid_image_queue.full():
                        depth_grid_image_queue.get_nowait()  # 古いデータを削除
                    depth_grid_image_queue.put_nowait(depth_grid)
                    logger.debug("Depth grid visualization added to queue")
                except Exception as e:
                    logger.warning(f"Failed to update depth grid queue: {e}")
            except Exception as e:
                logger.error(f"Failed to create depth grid: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # 定期的にログ出力
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} depth frames, inference time: {inference_time:.3f}s")
                # キューサイズの確認を追加
                logger.info(f"Queue sizes: topview={topview_image_queue.qsize()}/{topview_image_queue.maxsize}, depth={depth_image_queue.qsize()}/{depth_image_queue.maxsize}")
                
            # 100フレームごとにデバッグ画像を保存
            if frame_count % 100 == 0:
                try:
                    debug_vis = create_depth_visualization(current_depth_map, frame.shape)
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
        default_grid_image = np.zeros((480, 480, 3), dtype=np.uint8) + 30
        
        # グリッド線を描画
        for i in range(9):  # 8x8グリッドなので9本の線
            pos = i * 60  # 60ピクセルごと
            cv2.line(default_grid_image, (0, pos), (480, pos), (100, 100, 100), 1)
            cv2.line(default_grid_image, (pos, 0), (pos, 480), (100, 100, 100), 1)
        
        # 最後に表示した有効な画像を保持
        last_valid_grid_image = default_grid_image.copy()
        frame_count = 0
        
        while True:
            try:
                frame_count += 1
                
                # 5フレームごとにキューから取り出す（更新を強制）
                if frame_count % 5 == 0 and not depth_grid_image_queue.empty():
                    current_grid_image = depth_grid_image_queue.get_nowait()  # キューから取り出す
                    logger.debug("Retrieved new depth grid from queue")
                # 通常フレームでは参照のみ
                elif not depth_grid_image_queue.empty():
                    current_grid_image = depth_grid_image_queue.queue[0]  # 参照のみ
                else:
                    current_grid_image = None
                
                # 有効な画像なら最後の有効画像として保存
                if current_grid_image is not None and current_grid_image.shape[0] > 0:
                    last_valid_grid_image = current_grid_image.copy()
                
                # 前回の有効な深度グリッド画像を使用
                grid_image = last_valid_grid_image
                    
                # JPEGエンコード
                ret, buffer = cv2.imencode('.jpg', grid_image)
                if not ret:
                    logger.warning("JPEG encode failed for grid image.")
                    continue
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)  # 更新間隔を0.05から0.1に延長
                
            except Exception as e:
                logger.error(f"Error in depth grid stream: {e}")
                time.sleep(0.1)
    
    except Exception as e:
        logger.error(f"Fatal error in depth grid stream: {e}")

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