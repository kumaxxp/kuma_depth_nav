"""
Depth Navigation Camera Stream Module

このモジュールはUSBカメラからの映像をキャプチャし、Webブラウザ上でストリーミング表示するための
FastAPIベースのWebサービスを提供します。カメラ映像はMJPEGフォーマットでストリーミングされます。

使用方法:
- 直接実行すると、0.0.0.0:8888でサービスが起動します
- ブラウザで http://localhost:8888 にアクセスするとストリームが表示されます
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import time

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    ルートエンドポイント
    
    カメラストリームを表示するためのシンプルなHTMLページを返します
    
    Returns:
        HTMLResponse: カメラ映像を表示するHTMLページ
    """
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
    """
    USBカメラを初期化します
    
    Args:
        index (int): カメラデバイスのインデックス（デフォルト: 0）
        width (int): キャプチャ幅（デフォルト: 640px）
        height (int): キャプチャ高さ（デフォルト: 480px）
        
    Returns:
        cv2.VideoCapture: 初期化されたカメラオブジェクト
    """
    cam = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cam

def get_video_stream():
    """
    ビデオストリームジェネレーター
    
    カメラからフレームを連続的に読み取り、MJPEGストリーム形式に変換して
    yield します。カメラ接続問題を自動的に処理します。
    
    Yields:
        bytes: MJPEGストリーム形式のフレームデータ
    """
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

            # JPEGエンコード
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("[WARN] JPEG encode failed.")
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.005)  # フレームレート制御（約200fps制限）

    finally:
        camera.release()

@app.get("/video")
async def video_endpoint():
    """
    ビデオストリーミングエンドポイント
    
    MJPEGフォーマットでカメラストリームを提供します
    
    Returns:
        StreamingResponse: MJPEGストリームのレスポンス
    """
    return StreamingResponse(get_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
