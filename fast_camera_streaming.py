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
    create_depth_grid_visualization,
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
    # 共通のHTMLテンプレート
    html_content = """
    <html>
        <head>
            <title>Depth Camera Stream</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { display: flex; flex-wrap: wrap; justify-content: center; }
                .video-container { margin: 10px; text-align: center; }
                h1 { text-align: center; color: #333; }
                h3 { color: #555; }
                .stats { margin-top: 20px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }
                .refresh-btn { padding: 5px 10px; margin-top: 10px; cursor: pointer; }
                .warning { color: red; text-align: center; padding: 10px; }
                .row { display: flex; justify-content: center; flex-wrap: wrap; }
            </style>
        </head>
        <body>
            <h1>Depth Anything カメラストリーム</h1>
    """
    
    # axengineがあるかどうかで表示内容を変える
    if HAS_AXENGINE:
        html_content += """
            <div class="row">
                <div class="video-container">
                    <h3>RGB画像</h3>
                    <img src="/video" width="640" height="480" />
                </div>
                <div class="video-container">
                    <h3>深度推定</h3>
                    <img src="/depth_video" width="640" height="480" />
                </div>
            </div>
            <div class="row">
                <div class="video-container">
                    <h3>トップビュー</h3>
                    <img src="/topview" width="640" height="480" />
                </div>
                <div class="video-container">
                    <h3>深度グリッド</h3>
                    <img src="/depth_grid" width="640" height="480" />
                </div>
            </div>
        """
    else:
        html_content += """
            <p class="warning">axengine未インストールのため、深度推定は無効です。</p>
            <div class="container">
                <div class="video-container">
                    <h3>RGB画像</h3>
                    <img src="/video" width="640" height="480" />
                </div>
            </div>
        """
    
    # 共通のフッター
    html_content += """
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
                        last_valid_topview = current_topview.copy()
                
                # 前回の有効なトップビュー画像を使用
                topview_image = last_valid_topview
                    
                # JPEGエンコード
                ret, buffer = cv2.imencode('.jpg', topview_image)
                if not ret:
                    print("[WARN] JPEG encode failed for topview image.")
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
        logger.error("Failed to initialize depth model. Thread stopping.")
        return
    
    frame_count = 0
    skipped_frames = 0
    
    # 設定から絶対深度変換のスケーリング係数を取得
    scaling_factor = config["depth"].get("scaling_factor", 15.0)
    
    # 深度処理スレッド内で定期的に画像を保存
    debug_save_interval = 100  # 100フレームごとに保存
    
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
            else:
                # 深度マップの統計情報をログに出力
                if frame_count % 10 == 0:  # 10フレームごとに出力
                    valid_depths = depth_map[depth_map > 0.01]
                    if len(valid_depths) > 0:
                        min_val = valid_depths.min()
                        max_val = valid_depths.max()
                        mean_val = valid_depths.mean()
                        logger.info(f"Depth stats - Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}")
                    else:
                        logger.warning("No valid depth values found!")

            # 深度マップを可視化する前
            logger.info(f"Visualizing depth map with shape: {depth_map.shape}")

            # テストモードフラグ
            test_mode = False
            if test_mode:
                # テスト用のダミー深度マップを生成
                h, w = frame.shape[:2]
                dummy_depth = np.zeros((1, 256, 384, 1), dtype=np.float32)
                for y in range(256):
                    # 上から下に向かって0.1→0.9のグラデーション
                    value = 0.1 + 0.8 * (y / 255)
                    dummy_depth[0, y, :, 0] = value
                logger.info("Generated dummy depth map for testing")
                depth_map = dummy_depth
            
            # 深度マップを可視化
            logger.info(f"Depth map before visualization - shape: {depth_map.shape}, type: {type(depth_map)}")
            logger.info(f"Depth map range: min={np.min(depth_map)}, max={np.max(depth_map)}")

            # オプション: 相対深度から絶対深度への変換
            # absolute_depth = convert_to_absolute_depth(depth_map, scaling_factor)
                
            # 深度データをキューに追加（最適化版）
            try:
                with depth_data_queue.mutex:  # キューのロックを取得
                    if depth_data_queue.full():
                        depth_data_queue.queue.clear()  # キューを空にする（コピーなし）
                    depth_data_queue.queue.append(depth_map)  # 新しいデータを追加
                    depth_data_queue.not_empty.notify()  # 待機中のスレッドに通知
            except Exception as e:
                logger.warning(f"Failed to update depth data queue: {e}")
            
            # 深度マップを可視化
            try:
                # 元の単純な方法で深度マップを可視化
                colored_depth = create_depth_visualization(depth_map, frame.shape)
                
                # 可視化に成功したら深度画像をキューに追加
                if colored_depth is not None:
                    try:
                        if depth_image_queue.full():
                            depth_image_queue.get_nowait()  # 古い深度データを削除
                        depth_image_queue.put_nowait(colored_depth)
                    except Exception as e:
                        logger.warning(f"Failed to update depth image queue: {e}")
            except Exception as e:
                logger.warning(f"Failed to visualize depth map: {e}")

            # 点群生成とトップビュー生成（5フレームごとに実行して負荷軽減）
            frame_count += 1
            if frame_count % 5 == 0:
                try:
                    # 深度マップから点群生成
                    fx = config["depth"].get("fx", 500)  
                    fy = config["depth"].get("fy", 500)
                    cx = config["depth"].get("cx", depth_map.shape[2] / 2)  
                    cy = config["depth"].get("cy", depth_map.shape[1] / 2)
                    points = depth_to_point_cloud(depth_map, fx, fy, cx, cy)
                    
                    # 点群が生成できなかった場合はスキップ
                    if points is None or points.shape[0] == 0:
                        logger.warning("Empty point cloud. Skipping topview generation...")
                        continue
                    
                    # 占有グリッド生成
                    occupancy_grid = create_top_down_occupancy_grid(points)
                    
                    # トップビュー画像生成
                    topview_image = visualize_occupancy_grid(occupancy_grid)
                    
                    # 画像が正常に生成されたらキューに追加（最適化版）
                    if topview_image is not None and topview_image.shape[0] > 0:
                        with topview_image_queue.mutex:
                            if topview_image_queue.full():
                                topview_image_queue.queue.clear()
                            topview_image_queue.queue.append(topview_image)
                            topview_image_queue.not_empty.notify()
                except Exception as e:
                    logger.error(f"Error generating topview: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ログ出力（20フレームごと）
            if frame_count % 20 == 0:
                logger.info(f"Depth inference completed in {inference_time:.3f}s, shape: {depth_map.shape}")
            
            # 深度処理スレッド内で定期的に画像を保存
            if frame_count % debug_save_interval == 0:
                try:
                    # 元のフレームを保存
                    cv2.imwrite(f"debug_frame_{frame_count}.jpg", frame)
                    
                    # 深度マップを可視化して保存
                    colored_depth = create_depth_visualization(depth_map, frame.shape)
                    if colored_depth is not None:
                        cv2.imwrite(f"debug_depth_{frame_count}.jpg", colored_depth)
                        logger.info(f"Debug images saved for frame {frame_count}")
                except Exception as e:
                    logger.error(f"Failed to save debug images: {e}")
            
        except queue.Empty:
            # タイムアウト - 何もしない
            pass
        except Exception as e:
            logger.error(f"Error in depth processing thread: {e}")
            import traceback
            traceback.print_exc()  # スタックトレースを出力
            time.sleep(1.0)  # エラー発生時は少し長めに待機
    
    logger.info(f"Depth processing thread stopped. Processed {frame_count} frames, skipped {skipped_frames} frames.")

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
    import uvicorn
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8888)
    uvicorn.run(app, host=host, port=port)