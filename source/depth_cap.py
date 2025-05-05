"""
Depth Navigation Camera Stream Module

このモジュールはUSBカメラからの映像をキャプチャし、Webブラウザ上でストリーミング表示するための
FastAPIベースのWebサービスを提供します。カメラ映像はMJPEGフォーマットでストリーミングされます。

使用方法:
- 直接実行すると、0.0.0.0:8888でサービスが起動します
- ブラウザで http://localhost:8888 にアクセスするとストリームが表示されます
- 解像度変更: http://localhost:8888/?width=1280&height=720
"""
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import time
import logging
import sys
import os
import signal
import psutil
import uvicorn
from typing import Optional

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# グローバル変数
camera = None
app = FastAPI(title="Depth Camera Stream")

# ファビコン対応
@app.get("/favicon.ico")
async def favicon():
    """ファビコンを提供します"""
    # プロジェクトのルートディレクトリにfavicon.icoを作成するか、
    # デフォルトのダミーアイコンを返す
    favicon_path = os.path.join(os.path.dirname(__file__), "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    # ファビコンがない場合は空のレスポンス
    return Response(content=b"", media_type="image/x-icon")

@app.get("/", response_class=HTMLResponse)
async def root(width: Optional[int] = None, height: Optional[int] = None):
    """
    ルートエンドポイント
    
    カメラストリームを表示するためのシンプルなHTMLページを返します
    
    Args:
        width: 表示幅（オプション）
        height: 表示高さ（オプション）
    
    Returns:
        HTMLResponse: カメラ映像を表示するHTMLページ
    """
    # デフォルト値
    disp_width = width or 640
    disp_height = height or 480
    
    return f"""
    <html>
        <head>
            <title>USBカメラ ストリーム</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    text-align: center;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #333;
                    margin-bottom: 30px;
                }}
                .stream-container {{
                    margin: 20px auto;
                    box-shadow: 0 0 10px rgba(0,0,0,0.2);
                    display: inline-block;
                    background-color: #000;
                    padding: 5px;
                    border-radius: 5px;
                }}
                .controls {{
                    margin: 20px 0;
                }}
                .status {{
                    margin-top: 20px;
                    font-size: 0.9em;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <h1>デプスカメラストリーム</h1>
            <div class="stream-container">
                <img src="/video?width={disp_width}&height={disp_height}" width="{disp_width}" height="{disp_height}" />
            </div>
            <div class="controls">
                <a href="/?width=320&height=240">低解像度</a> |
                <a href="/?width=640&height=480">標準解像度</a> |
                <a href="/?width=1280&height=720">高解像度</a>
            </div>
            <div class="status">
                OpenCV: {cv2.__version__} | <a href="/status" target="_blank">システム状態</a>
            </div>
        </body>
    </html>
    """

def initialize_camera(index=0, width=640, height=480, max_attempts=3):
    """
    USBカメラを初期化します
    
    Args:
        index (int): カメラデバイスのインデックス（デフォルト: 0）
        width (int): キャプチャ幅（デフォルト: 640px）
        height (int): キャプチャ高さ（デフォルト: 480px）
        max_attempts (int): 初期化試行回数（デフォルト: 3）
        
    Returns:
        cv2.VideoCapture: 初期化されたカメラオブジェクト、失敗時はNone
    """
    for attempt in range(max_attempts):
        try:
            # Linux環境でV4L2を使用
            cam = cv2.VideoCapture(index, cv2.CAP_V4L2)
            
            if not cam.isOpened():
                logger.warning(f"カメラ初期化試行 {attempt+1}/{max_attempts} 失敗")
                if attempt < max_attempts - 1:
                    time.sleep(1.0)
                    continue
                else:
                    logger.error(f"カメラindex={index}を開けませんでした")
                    return None
            
            # カメラプロパティ設定
            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # 実際に設定された値を確認
            actual_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            # テストフレームを読み込んで動作確認
            success, _ = cam.read()
            if not success:
                logger.warning("カメラはオープンできましたが、フレーム読込に失敗")
                cam.release()
                if attempt < max_attempts - 1:
                    time.sleep(1.0)
                    continue
                else:
                    return None
                    
            logger.info(f"カメラ初期化成功: {actual_width}x{actual_height}")
            return cam
            
        except Exception as e:
            logger.error(f"カメラ初期化エラー: {e}")
            if attempt < max_attempts - 1:
                time.sleep(1.0)
            else:
                return None
    
    return None

def get_video_stream(width=640, height=480):
    """
    ビデオストリームジェネレーター
    
    カメラからフレームを連続的に読み取り、MJPEGストリーム形式に変換して
    yield します。カメラ接続問題を自動的に処理します。
    
    Args:
        width (int): 要求された画像幅
        height (int): 要求された画像高さ
    
    Yields:
        bytes: MJPEGストリーム形式のフレームデータ
    """
    global camera
    retry_count = 0
    max_retries = 5
    consecutive_failures = 0
    max_consecutive_failures = 10
    
    try:
        if camera is None or not camera.isOpened():
            camera = initialize_camera(width=width, height=height)
        
        if camera is None:
            logger.error("カメラの初期化に失敗しました")
            return

        while True:
            # カメラ接続チェック
            if not camera.isOpened():
                if retry_count >= max_retries:
                    logger.error(f"{max_retries}回の再試行後もカメラに接続できません")
                    break
                    
                logger.warning(f"カメラ未接続。再試行中... ({retry_count + 1}/{max_retries})")
                camera.release()
                time.sleep(1.0)
                camera = initialize_camera(width=width, height=height)
                retry_count += 1
                continue
            
            # フレーム取得
            success, frame = camera.read()
            if not success or frame is None:
                consecutive_failures += 1
                logger.warning(f"フレーム読み取り失敗 ({consecutive_failures}/{max_consecutive_failures})")
                
                # 連続失敗が一定回数を超えたらカメラ再初期化
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning("連続フレーム取得失敗。カメラを再初期化します")
                    camera.release()
                    time.sleep(1.0)
                    camera = initialize_camera(width=width, height=height)
                    consecutive_failures = 0
                
                time.sleep(0.1)
                continue
            
            # フレーム取得成功時はカウンタリセット
            retry_count = 0
            consecutive_failures = 0

            # JPEGエンコード
            try:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    logger.warning("JPEGエンコード失敗")
                    continue
                    
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.005)  # フレームレート制御（約200fps制限）
                
            except Exception as e:
                logger.error(f"フレーム処理エラー: {e}")
                time.sleep(0.1)

    except Exception as e:
        logger.error(f"ストリーム処理中の予期せぬエラー: {e}")
    
    # finally句はここでは使わない - カメラはグローバル変数なので、
    # ストリームが終了してもカメラを保持しておく

@app.get("/video")
async def video_endpoint(width: Optional[int] = Query(640), height: Optional[int] = Query(480)):
    """
    ビデオストリーミングエンドポイント
    
    MJPEGフォーマットでカメラストリームを提供します
    
    Args:
        width: 要求解像度の幅
        height: 要求解像度の高さ
    
    Returns:
        StreamingResponse: MJPEGストリームのレスポンス
    """
    return StreamingResponse(
        get_video_stream(width=width, height=height), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/status")
async def status_endpoint():
    """
    システムステータスエンドポイント
    
    Returns:
        dict: システムとカメラの情報を含む辞書
    """
    # システムリソース情報を取得
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    # カメラ情報
    camera_status = "connected" if camera and camera.isOpened() else "disconnected"
    camera_info = {}
    
    if camera and camera.isOpened():
        camera_info = {
            "width": camera.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": camera.get(cv2.CAP_PROP_FPS),
            "format": "MJPG",
            "device_index": 0  # 使用しているデバイスインデックス
        }
    
    return {
        "status": "running",
        "opencv_version": cv2.__version__,
        "timestamp": time.time(),
        "system": {
            "cpu_percent": process.cpu_percent(),
            "memory_usage_mb": mem_info.rss / (1024 * 1024),
            "uptime_seconds": time.time() - process.create_time()
        },
        "camera": {
            "status": camera_status,
            **camera_info
        }
    }

def cleanup():
    """アプリケーション終了時の後処理"""
    global camera
    if camera is not None:
        logger.info("カメラリソースを解放します")
        camera.release()

def signal_handler(sig, frame):
    """シグナルハンドラ - 正常終了処理"""
    logger.info(f"シグナル {sig} を受信。シャットダウンします。")
    cleanup()
    sys.exit(0)

if __name__ == "__main__":
    # シグナルハンドラ登録
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill
    
    try:
        logger.info(f"カメラストリーミングサーバー起動中... OpenCV {cv2.__version__}")
        uvicorn.run(app, host="0.0.0.0", port=8888)
    finally:
        cleanup()