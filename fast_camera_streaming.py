from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import time
import threading
import queue
import os
from contextlib import asynccontextmanager

# axengine をインポート
try:
    import axengine as axe
    HAS_AXENGINE = True
    print("[INFO] axengine successfully imported")
except ImportError:
    HAS_AXENGINE = False
    print("[WARN] axengine is not installed. Running in basic mode without depth estimation.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフスパン管理"""
    global process_thread, is_running
    
    # 起動時処理
    process_thread = threading.Thread(target=depth_processing_thread, daemon=True)
    process_thread.start()
    print("[INFO] Started depth processing thread")
    
    yield  # アプリケーション実行中
    
    # 終了時処理
    is_running = False
    if process_thread:
        process_thread.join(timeout=2.0)
    print("[INFO] Stopped depth processing thread")

app = FastAPI(lifespan=lifespan)

# モデルパス
MODEL_PATH = '/opt/m5stack/data/depth_anything/compiled.axmodel'

# グローバル変数
frame_queue = queue.Queue(maxsize=2)  # 最新のフレームだけを保持するキュー
depth_image_queue = queue.Queue(maxsize=1)  # 深度画像キュー
depth_data_queue = queue.Queue(maxsize=1)  # 深度データキュー
process_thread = None
is_running = True
depth_model = None  # Depth Anythingモデル

@app.get("/", response_class=HTMLResponse)
async def root():
    # axengine がなければシンプルな表示に
    if not HAS_AXENGINE:
        return """
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
                </style>
            </head>
            <body>
                <h1>Depth Anything カメラストリーム</h1>
                <div class="container">
                    <div class="video-container">
                        <h3>RGB画像</h3>
                        <img src="/video" width="640" height="480" />
                    </div>
                    <div class="video-container">
                        <h3>深度推定</h3>
                        <img src="/depth_video" width="640" height="480" />
                    </div>
                </div>
            </body>
        </html>
        """

def initialize_depth_model():
    """Depth Anythingモデルを初期化"""
    if not HAS_AXENGINE:
        print("[WARN] axengine not installed. Cannot initialize depth model.")
        return None
        
    try:
        print(f"[INFO] Loading model from {MODEL_PATH}")
        # axengineを使用してモデルをロード
        session = axe.InferenceSession(MODEL_PATH)
        print("[INFO] Model loaded successfully")
        return session
    except Exception as e:
        print(f"[ERROR] Failed to initialize depth model: {e}")
        return None

def process_frame(frame: np.ndarray, target_size=(384, 256)) -> np.ndarray:
    """入力フレームの前処理を行う"""
    if frame is None:
        raise ValueError("フレームの読み込みに失敗しました")
    
    resized_frame = cv2.resize(frame, target_size)
    # RGB -> BGR の変換とバッチ次元の追加
    return np.expand_dims(resized_frame[..., ::-1], axis=0)

def create_depth_visualization(depth_map: np.ndarray, original_shape) -> np.ndarray:
    """深度マップの可視化を行う"""
    depth_feature = depth_map.reshape(depth_map.shape[-2:])
    
    # 正規化と色付け
    normalized = (depth_feature - depth_feature.min()) / (depth_feature.max() - depth_feature.min())
    depth_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    
    # 元の画像サイズにリサイズ
    depth_resized = cv2.resize(depth_colored, (original_shape[1], original_shape[0]))
    
    return depth_resized

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
                print("[WARN] Camera not open. Retrying...")
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
            time.sleep(0.001)

    finally:
        camera.release()

def get_depth_stream():
    """深度画像ストリームを生成します"""
    try:
        # デフォルトの画像を用意 (青いグラデーション)
        default_depth_image = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(480):
            default_depth_image[i, :] = [0, 0, int(255 * i / 480)]
        
        while True:
            try:
                # 深度画像があればそれを使用、なければデフォルト画像
                if not depth_image_queue.empty():
                    depth_image = depth_image_queue.get_nowait()
                else:
                    depth_image = default_depth_image.copy()
                    
                # JPEGエンコード
                ret, buffer = cv2.imencode('.jpg', depth_image)
                if not ret:
                    print("[WARN] JPEG encode failed for depth image.")
                    continue
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.05)  # 深度推論は低いフレームレートでOK
                
            except Exception as e:
                print(f"[ERROR] Error in depth stream: {e}")
                time.sleep(0.1)
    
    except Exception as e:
        print(f"[ERROR] Fatal error in depth stream: {e}")

def depth_processing_thread():
    """深度推論を行うスレッド"""
    global is_running, depth_model
    print("[INFO] Depth processing thread started")
    
    # モデル初期化
    depth_model = initialize_depth_model()
    
    if depth_model is None:
        print("[ERROR] Failed to initialize depth model. Thread stopping.")
        return
    
    # 入力名の取得
    input_name = depth_model.get_inputs()[0].name
    print(f"[INFO] Model input name: {input_name}")
    
    frame_count = 0
    skipped_frames = 0
    
    while is_running:
        try:
            # キューからフレームを取得
            frame = frame_queue.get(timeout=1.0)
            
        #    # 2フレームごとに処理（処理負荷軽減のため）
        #    frame_count += 1
        #    if frame_count % 2 != 0:
        #        skipped_frames += 1
        #        continue
                
            # 画像の前処理
            start_time = time.time()
            input_tensor = process_frame(frame)
            
            # 深度推論
            output = depth_model.run(None, {input_name: input_tensor})
            inference_time = time.time() - start_time
            
            # 深度マップを取得
            depth_map = output[0]
                
            # 深度データをキューに追加
            try:
                if depth_data_queue.full():
                    depth_data_queue.get_nowait()
                depth_data_queue.put_nowait(depth_map)
            except:
                pass
            
            # 深度マップを可視化
            colored_depth = create_depth_visualization(depth_map, frame.shape)
            
            # 深度画像をキューに追加
            try:
                if depth_image_queue.full():
                    depth_image_queue.get_nowait()
                depth_image_queue.put_nowait(colored_depth)
            except:
                pass
                
            #print(f"[INFO] Depth inference completed in {inference_time:.3f}s, "
            #      f"shape: {depth_map.shape}")
            
        except queue.Empty:
            # タイムアウト - 何もしない
            pass
        except Exception as e:
            print(f"[ERROR] Error in depth processing thread: {e}")
            import traceback
            traceback.print_exc()  # スタックトレースを出力
            time.sleep(1.0)  # エラー発生時は少し長めに待機
    
    print(f"[INFO] Depth processing thread stopped. Processed {frame_count} frames, skipped {skipped_frames} frames.")

@app.get("/video")
async def video_endpoint():
    """ビデオストリームのエンドポイント"""
    return StreamingResponse(get_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/depth_video")
async def depth_video_endpoint():
    """深度画像ストリームのエンドポイント"""
    return StreamingResponse(get_depth_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)