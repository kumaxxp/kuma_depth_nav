from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import time
import threading
import queue

app = FastAPI()

# グローバル変数
frame_queue = queue.Queue(maxsize=1)  # 最新のフレームだけを保持するキュー
process_thread = None
is_running = True

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
    cam = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cam

def get_video_stream():
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
            time.sleep(0.005)

    finally:
        camera.release()

# 何もしないスレッド関数
def dummy_process_thread():
    global is_running
    print("[INFO] Dummy process thread started")
    
    while is_running:
        try:
            # キューからフレームを取得
            frame = frame_queue.get(timeout=1.0)
            
            # 何もしない（フレームを受け取るだけ）
            # print("[DEBUG] Got frame", frame.shape) # デバッグ用
            
        except queue.Empty:
            # タイムアウトしても何もしない
            pass
        except Exception as e:
            print(f"[ERROR] Error in dummy process thread: {e}")
            time.sleep(0.1)
    
    print("[INFO] Dummy process thread stopped")

@app.get("/video")
async def video_endpoint():
    return StreamingResponse(get_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.on_event("startup")
async def startup_event():
    # アプリケーション起動時に処理スレッドを開始
    global process_thread
    process_thread = threading.Thread(target=dummy_process_thread, daemon=True)
    process_thread.start()
    print("[INFO] Started dummy process thread")

@app.on_event("shutdown")
async def shutdown_event():
    # アプリケーション終了時に処理スレッドを停止
    global is_running
    is_running = False
    if process_thread:
        process_thread.join(timeout=2.0)  # 最大2秒待機
    print("[INFO] Stopped dummy process thread")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)