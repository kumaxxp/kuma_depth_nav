from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import time
import threading
import queue
import logging
from contextlib import asynccontextmanager

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# グローバル変数
frame_queue = queue.Queue(maxsize=10)  # フレームキュー
process_thread = None  # 画像処理スレッド
is_processing = False  # 処理スレッドの状態
lock = threading.Lock()  # スレッドロック
latest_processed_frame = None  # 処理済みフレーム

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPIアプリケーションのライフサイクルを管理します。
    """
    global is_processing, process_thread
    
    # アプリケーション起動時の処理
    logger.info("Starting up...")
    is_processing = True
    process_thread = threading.Thread(target=processing_thread, daemon=True)
    process_thread.start()
    logger.info("Started processing thread.")
    
    yield
    
    # アプリケーション終了時の処理
    logger.info("Shutting down...")
    is_processing = False
    if process_thread:
        process_thread.join()
    logger.info("Stopped processing thread.")

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>USBカメラ ストリーム</title>
        </head>
        <body>
            <h1>USBカメラの映像を表示中</h1>
            <img src="/video" width="640" height="480" />
        </body>
    </html>
    """

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
                logger.warning("Failed to read frame. Skipping...")
                continue

            # フレームをキューに追加
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                logger.warning("Frame queue is full. Dropping frame.")

            # JPEGエンコード
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logger.warning("JPEG encode failed.")
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.005)

    finally:
        camera.release()

def process_frame(frame):
    """フレームを処理します（depth anythingなど）"""
    # ここに画像処理のコードを記述
    # 例：グレースケール変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 例：エッジ検出
    edges = cv2.Canny(gray, 100, 200)
    
    # カラー画像に変換
    processed_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return processed_frame

def processing_thread():
    """画像処理スレッド"""
    global is_processing, latest_processed_frame
    
    while is_processing:
        try:
            frame = frame_queue.get(timeout=1)  # キューからフレームを取得
            
            processed_frame = process_frame(frame)  # フレームを処理
            
            with lock:
                latest_processed_frame = processed_frame  # 処理済みフレームを更新
                
        except queue.Empty:
            logger.debug("Frame queue is empty.")
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            time.sleep(0.1)

@app.get("/video")
async def video_endpoint():
    """ビデオストリームのエンドポイント"""
    return StreamingResponse(get_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)