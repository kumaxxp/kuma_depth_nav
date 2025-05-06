from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import io  # <- これを追加
import cv2
import numpy as np
import time
import threading
import queue
import os
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles

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

# 静的ファイル配信の設定（app定義の直後に追加）
import os
from fastapi.staticfiles import StaticFiles

# staticディレクトリがなければ作成
if not os.path.exists("static"):
    os.makedirs("static")
    logger.info("Created static directory")

# エラー画像の生成
try:
    error_img = np.zeros((256, 384, 3), dtype=np.uint8)
    error_img[:, :] = [0, 0, 150]  # 赤い背景
    cv2.putText(error_img, "ERROR", (120, 128), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.imwrite("static/error.jpg", error_img)
    logger.info("Created error image")
except Exception as e:
    logger.error(f"Failed to create error image: {e}")

# 静的ファイル配信を設定
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("Static files directory mounted")
except Exception as e:
    logger.error(f"Failed to mount static files: {e}")

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
                max-width: 1280px;  /* 全体の幅を制限 */
                margin: 0 auto;
            }
            .video-container {
                display: grid;
                grid-template-columns: repeat(2, 1fr);  /* 2列のグリッド */
                gap: 15px;
                background: white;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .video-item {
                text-align: center;
            }
            h2 {
                color: #333;
                margin-top: 0;
            }
            h3 {
                margin-top: 0;
                margin-bottom: 10px;
            }
            img {
                max-width: 100%;  /* 最大幅を100%に */
                height: auto;     /* 高さを自動調整 */
                border-radius: 5px;
                display: inline-block;
            }
            .status {
                margin-top: 10px;
                padding: 10px;
                background-color: #e6f7ff;
                border-left: 4px solid #1890ff;
                margin-bottom: 15px;
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
        <div class="container">
            <h1>Kuma Depth Navigation System</h1>
            <div class="status" id="statusBar">
                System Status: Running
            </div>
            
            <div class="video-container">
                <!-- カメラビュー -->
                <div class="video-item">
                    <h3>カメラ</h3>
                    <img src="/video" width="384" height="256" alt="Camera Feed">
                </div>
                
                <!-- 深度マップビュー -->
                <div class="video-item">
                    <h3>深度マップ</h3>
                    <img src="/depth_video" width="384" height="256" alt="Depth Map">
                </div>
                
                <!-- トップビュー -->
                <div class="video-item">
                    <h3>トップビュー</h3>
                    <img src="/topview" width="384" height="384" alt="Top View">
                </div>
                
                <!-- 深度グリッド -->
                <div class="video-item">
                    <h3>深度グリッド</h3>
                    <img src="/depth_grid" width="384" height="288" alt="Depth Grid">
                </div>
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
    # テスト用のデフォルト画像（明確に認識できるパターン）
    default_image = np.zeros((200, 200, 3), dtype=np.uint8)
    # グレーの背景
    default_image[:, :] = [50, 50, 50]
    # 格子パターン
    for i in range(0, 200, 20):
        cv2.line(default_image, (0, i), (200, i), [100, 100, 100], 1)
        cv2.line(default_image, (i, 0), (i, 200), [100, 100, 100], 1)
    # メッセージを追加
    cv2.putText(default_image, "Waiting for data...", (30, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, [200, 200, 200], 1)
    
    last_valid_topview = default_image.copy()  # 最初のデフォルト画像を保存
    
    while True:
        try:
            if not topview_image_queue.empty():
                try:
                    # キューから画像を取得
                    topview = topview_image_queue.get(timeout=0.1)
                    last_valid_topview = topview.copy()
                    logger.debug(f"Got topview from queue: shape={topview.shape}")
                except Exception as e:
                    logger.warning(f"Failed to get topview from queue: {e}")
                    # エラー発生時は前回の有効画像を使用
                    topview = last_valid_topview
            else:
                # キューが空の場合は前回の有効な画像を使用
                topview = last_valid_topview
                
            # JEPGエンコード
            ret, buffer = cv2.imencode('.jpg', topview)
            if not ret:
                logger.warning("JPEG encode failed for topview.")
                continue
            
            # フレームを返す
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # 更新間隔を調整（0.1秒 = 10FPS）
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error in topview stream: {e}")
            time.sleep(0.1)
    
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
                # 深度マップを可視化
                current_colored_depth = create_depth_visualization(current_depth_map, frame.shape)
                
                # 可視化が有効かチェック
                is_valid_visualization = (current_colored_depth is not None and 
                                         current_colored_depth.shape[0] > 0)
                
                if is_valid_visualization:
                    # トップビュー変換対象領域を表示（中央の有効領域）
                    h, w = current_colored_depth.shape[:2]
                    
                    # 有効なトップビュー変換領域（中央80%程度を使用）
                    margin_percent = 0.1  # 画像の端から10%を除外
                    x1 = int(w * margin_percent)
                    y1 = int(h * margin_percent)
                    x2 = int(w * (1.0 - margin_percent))
                    y2 = int(h * (1.0 - margin_percent))
                    
                    # 関心領域を描画
                    cv2.rectangle(current_colored_depth, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 前方方向を示す矢印を描画
                    arrow_start = (w // 2, h - 20)
                    arrow_end = (w // 2, h - 60)
                    cv2.arrowedLine(current_colored_depth, arrow_start, arrow_end, 
                                   (0, 255, 0), 2, tipLength=0.3)
                    
                    # 「トップビュー領域」というテキストを追加
                    cv2.putText(current_colored_depth, "Top View Area", (x1 + 10, y1 + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    
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
            
            # 深度マップから絶対深度に変換した後に追加
            # トップビューの生成（10フレームに1回程度）
            if is_valid_depth and frame_count % 10 == 0:
                logger.info("Starting topview generation")
                try:
                    # 深度マップのサイズを取得
                    h, w = absolute_depth.shape[:2]
                    
                    # 変換に使用する中央部分の範囲を計算
                    margin_percent = 0.1  # 10%マージンを使用（深度表示と同じ）
                    x1 = int(w * margin_percent)
                    y1 = int(h * margin_percent)
                    x2 = int(w * (1.0 - margin_percent))
                    y2 = int(h * (1.0 - margin_percent))
                    
                    # 深度マップの中央部分を抽出
                    center_depth = absolute_depth[y1:y2, x1:x2].copy()
                    
                    # 点群に変換（中央部分のみ）
                    logger.debug(f"Converting center depth to point cloud, depth shape: {center_depth.shape}")
                    point_cloud = depth_to_point_cloud(
                        center_depth,  # 中央部分のみ使用
                        fx=depth_config.get("fx", 500),
                        fy=depth_config.get("fy", 500)
                    )
                    logger.debug(f"Point cloud shape: {point_cloud.shape}")
                    
                    # 占有グリッドに変換
                    logger.debug("Creating occupancy grid")
                    occupancy_grid = create_top_down_occupancy_grid(
                        point_cloud,
                        grid_resolution=0.05,
                        grid_width=200,
                        grid_height=200,
                        height_threshold=0.5
                    )
                    
                    # デバッグ: グリッドの値の分布をログ出力
                    if occupancy_grid is not None:
                        logger.debug(f"Occupancy grid shape: {occupancy_grid.shape}")
                        values, counts = np.unique(occupancy_grid, return_counts=True)
                        logger.debug(f"Occupancy grid values: {values}, counts: {counts}")
                    else:
                        logger.warning("Occupancy grid is None")
                    
                    # 占有グリッドを可視化
                    logger.debug("Visualizing occupancy grid")
                    topview_image = visualize_occupancy_grid(occupancy_grid)
                    
                    # トップビューに追加情報を付与
                    h_top, w_top = topview_image.shape[:2]
                    # カメラ位置（下中央）
                    cam_pos = (w_top // 2, h_top - 20)
                    # カメラから前方への矢印を描画
                    cv2.arrowedLine(topview_image, cam_pos, (w_top // 2, 20), 
                                   (0, 255, 0), 2, tipLength=0.1)

                    # 説明テキスト
                    cv2.putText(topview_image, "Camera", (cam_pos[0] - 30, cam_pos[1] + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(topview_image, "Forward", (w_top // 2 - 30, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    logger.debug(f"Topview image shape: {topview_image.shape}")
                    
                    # トップビューをキューに追加
                    try:
                        if topview_image_queue.full():
                            topview_image_queue.get_nowait()  # 古いデータを削除
                        topview_image_queue.put_nowait(topview_image)
                        logger.debug("Topview visualization added to queue")
                    except Exception as e:
                        logger.warning(f"Failed to update topview queue: {e}")
                except Exception as e:
                    logger.error(f"Failed to create topview: {e}")
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
                    # cv2.imwrite(f"depth_frame_{frame_count}.jpg", debug_vis)
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

@app.get("/test_topview")
async def test_topview_endpoint():
    """トップビュー生成のテスト用エンドポイント"""
    try:
        # テスト用のトップビュー画像を作成
        test_grid = np.zeros((200, 200), dtype=np.uint8)
        
        # 中央に障害物を配置
        for i in range(200):
            for j in range(200):
                dist = np.sqrt((i-100)**2 + (j-100)**2)
                if dist < 50:  # 中央に円形の障害物
                    test_grid[i, j] = 255
                elif (i > 90 and i < 110) or (j > 90 and j < 110):  # 十字線
                    test_grid[i, j] = 128
        
        # トップビューを可視化
        test_image = visualize_occupancy_grid(test_grid)
        
        # テキスト追加
        cv2.putText(test_image, "Test Pattern", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # キューに追加
        if topview_image_queue.full():
            topview_image_queue.get_nowait()
        topview_image_queue.put_nowait(test_image)
        
        # JPEGにエンコード
        ret, buffer = cv2.imencode('.jpg', test_image)
        
        # StreamingResponseで直接返す
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Test topview error: {e}")
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.get("/test_grid")
async def test_grid_endpoint():
    """テスト用のグリッドパターンを生成するエンドポイント"""
    try:
        # テスト用のグリッドを作成（異なるパターン）
        test_grid = np.zeros((200, 200), dtype=np.uint8)
        
        # 1と2の市松模様パターン
        for i in range(20):
            for j in range(20):
                if (i + j) % 2 == 0:
                    test_grid[i*10:(i+1)*10, j*10:(j+1)*10] = 1  # 障害物
                else:
                    test_grid[i*10:(i+1)*10, j*10:(j+1)*10] = 2  # 通行可能
        
        # 中央に円形の通行可能領域
        center_x, center_y = 100, 100
        for i in range(200):
            for j in range(200):
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                if dist < 50:
                    test_grid[i, j] = 2  # 通行可能な円
        
        # 可視化
        test_image = visualize_occupancy_grid(test_grid)
        
        # キューに画像を追加
        if topview_image_queue.full():
            topview_image_queue.get_nowait()
        topview_image_queue.put_nowait(test_image)
        
        # 画像をエンコード
        ret, buffer = cv2.imencode('.jpg', test_image)
        
        # 画像を直接返す
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"Error in test grid: {e}")
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

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