"""
Depth Navigation Camera Stream Module

このモジュールはUSBカメラからの映像をキャプチャし、Webブラウザ上でストリーミング表示するための
FastAPIベースのWebサービスを提供します。カメラ映像はMJPEGフォーマットでストリーミングされます。

使用方法:
- 直接実行すると、0.0.0.0:8888でサービスが起動します
- ブラウザで http://localhost:8888 にアクセスするとストリームが表示されます
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import time
import logging
import sys

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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
            <meta charset="utf-8">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    text-align: center;
                }
                h1 {
                    color: #333;
                }
                .stream-container {
                    margin: 20px auto;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    display: inline-block;
                }
            </style>
        </head>
        <body>
            <h1>USBカメラの映像を表示中</h1>
            <div class="stream-container">
                <img src="/video" width="640" height="480" />
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

def get_video_stream():
    """
    ビデオストリームジェネレーター
    
    カメラからフレームを連続的に読み取り、MJPEGストリーム形式に変換して
    yield します。カメラ接続問題を自動的に処理します。
    
    Yields:
        bytes: MJPEGストリーム形式のフレームデータ
    """
    camera = None
    retry_count = 0
    max_retries = 5
    consecutive_failures = 0
    max_consecutive_failures = 10
    
    try:
        camera = initialize_camera()
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
                camera = initialize_camera()
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
                    camera = initialize_camera()
                    consecutive_failures = 0
                
                time.sleep(0.1)
                continue
            
            # フレーム取得成功時はカウンタリセット
            retry_count = 0
            consecutive_failures = 0

            # JPEGエンコード
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
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
    
    finally:
        if camera is not None:
            camera.release()
            logger.info("カメラリソースを解放しました")

@app.get("/video")
async def video_endpoint():
    """
    ビデオストリーミングエンドポイント
    
    MJPEGフォーマットでカメラストリームを提供します
    
    Returns:
        StreamingResponse: MJPEGストリームのレスポンス
    """
    return StreamingResponse(get_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
async def status_endpoint():
    """
    システムステータスエンドポイント
    
    Returns:
        dict: システム情報を含む辞書
    """
    return {
        "status": "running",
        "opencv_version": cv2.__version__,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"カメラストリーミングサーバー起動中... OpenCV {cv2.__version__}")
    uvicorn.run(app, host="0.0.0.0", port=8888)
