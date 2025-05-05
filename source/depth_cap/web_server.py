"""
Webサーバーモジュール

FastAPIベースのWebサーバーを提供し、カメラ映像のストリーミングと
カメラ状態のモニタリングを行います。
"""
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, Response
import cv2
import numpy as np
import time
import logging
import os
import json
from typing import Optional, Dict, Any

# ロガー設定
logger = logging.getLogger(__name__)

def create_app(camera_manager, image_processor=None) -> FastAPI:
    """
    FastAPIアプリケーションを作成します
    
    Args:
        camera_manager: カメラマネージャーインスタンス
        image_processor: 画像処理クラスのインスタンス(オプション)
        
    Returns:
        FastAPI: 設定済みのFastAPIアプリケーション
    """
    app = FastAPI(title="Depth Camera Stream")
    
    @app.get("/favicon.ico")
    async def favicon():
        """ファビコンを提供します"""
        favicon_path = os.path.join(os.path.dirname(__file__), "favicon.ico")
        if os.path.exists(favicon_path):
            return FileResponse(favicon_path)
        return Response(content=b"", media_type="image/x-icon")
    
    @app.get("/", response_class=HTMLResponse)
    async def root(width: Optional[int] = None, height: Optional[int] = None):
        """
        ルートエンドポイント
        
        カメラストリームを表示するためのHTMLページを返します
        
        Args:
            width: 表示幅（オプション）
            height: 表示高さ（オプション）
        
        Returns:
            HTMLResponse: カメラ映像を表示するHTMLページ
        """
        # デフォルト値
        disp_width = width or 640
        disp_height = height or 480
        
        has_processor = "true" if image_processor else "false"
        
        return f"""
        <html>
            <head>
                <title>デプスカメラストリーム</title>
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
                    .stream-tabs {{
                        margin: 20px 0;
                    }}
                    .stream-tabs button {{
                        padding: 8px 16px;
                        margin: 0 5px;
                        background-color: #ddd;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                    }}
                    .stream-tabs button.active {{
                        background-color: #007bff;
                        color: white;
                    }}
                </style>
            </head>
            <body>
                <h1>デプスカメラストリーム</h1>
                
                <div class="stream-tabs">
                    <button id="raw-btn" class="active" onclick="switchStream('raw')">元映像</button>
                    <button id="processed-btn" onclick="switchStream('processed')" 
                            style="display: {'' if image_processor else 'none'}">処理映像</button>
                </div>
                
                <div class="stream-container">
                    <img id="stream-img" src="/video?width={disp_width}&height={disp_height}" 
                         width="{disp_width}" height="{disp_height}" />
                </div>
                
                <div class="controls">
                    <a href="/?width=320&height=240">低解像度</a> |
                    <a href="/?width=640&height=480">標準解像度</a> |
                    <a href="/?width=1280&height=720">高解像度</a>
                </div>
                
                <div class="status">
                    OpenCV: {cv2.__version__} | 
                    <a href="/status" target="_blank">システム状態</a> |
                    <a href="/api/camera-info" target="_blank">カメラ情報</a>
                </div>
                
                <script>
                    const hasProcessor = {has_processor};
                    
                    function switchStream(type) {{
                        const img = document.getElementById('stream-img');
                        const rawBtn = document.getElementById('raw-btn');
                        const processedBtn = document.getElementById('processed-btn');
                        
                        if (type === 'raw') {{
                            img.src = `/video?width={disp_width}&height={disp_height}`;
                            rawBtn.classList.add('active');
                            processedBtn.classList.remove('active');
                        }} else {{
                            img.src = `/video/processed?width={disp_width}&height={disp_height}`;
                            rawBtn.classList.remove('active');
                            processedBtn.classList.add('active');
                        }}
                    }}
                </script>
            </body>
        </html>
        """
    
    def generate_mjpeg_frame(frame: np.ndarray) -> bytes:
        """MJPEGストリーム用のフレームデータを生成"""
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            raise RuntimeError("JPEGエンコード失敗")
        return (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    def get_raw_video_stream(width=640, height=480):
        """
        生のビデオストリームジェネレーター
        
        Args:
            width: 要求解像度幅
            height: 要求解像度高さ
            
        Yields:
            bytes: MJPEGフレームデータ
        """
        if not camera_manager.is_running:
            camera_manager.start_capture()
        
        no_frame_count = 0
        max_no_frame = 10  # フレーム取得失敗の最大許容回数
        
        try:
            while True:
                frame = camera_manager.get_latest_frame()
                
                if frame is None:
                    no_frame_count += 1
                    if no_frame_count > max_no_frame:
                        logger.warning(f"{max_no_frame}回連続でフレーム取得失敗")
                        # ダミーフレームを生成
                        dummy = np.zeros((height, width, 3), dtype=np.uint8)
                        cv2.putText(dummy, "No Signal", (width//4, height//2), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        yield generate_mjpeg_frame(dummy)
                    time.sleep(0.1)
                    continue
                
                # フレーム取得成功
                no_frame_count = 0
                yield generate_mjpeg_frame(frame)
                time.sleep(0.005)  # フレームレート制限
                
        except Exception as e:
            logger.error(f"ストリーム生成中エラー: {e}")
    
    def get_processed_video_stream(width=640, height=480):
        """
        処理済みビデオストリームジェネレーター
        
        Args:
            width: 要求解像度幅
            height: 要求解像度高さ
            
        Yields:
            bytes: MJPEGフレームデータ
        """
        if not image_processor or not camera_manager.is_running:
            if not camera_manager.is_running:
                camera_manager.start_capture()
            if image_processor and not image_processor.is_running:
                image_processor.start_processing()
        
        no_frame_count = 0
        max_no_frame = 10  # フレーム取得失敗の最大許容回数
        
        try:
            while True:
                # 処理済みフレームの取得
                frame = None
                if image_processor:
                    frame = image_processor.get_latest_processed_frame()
                
                # 処理済みフレームがなければ生フレームを使用
                if frame is None:
                    frame = camera_manager.get_latest_frame()
                
                if frame is None:
                    no_frame_count += 1
                    if no_frame_count > max_no_frame:
                        # ダミーフレームを生成
                        dummy = np.zeros((height, width, 3), dtype=np.uint8)
                        cv2.putText(dummy, "Processing...", (width//4, height//2), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        yield generate_mjpeg_frame(dummy)
                    time.sleep(0.1)
                    continue
                
                # フレーム取得成功
                no_frame_count = 0
                yield generate_mjpeg_frame(frame)
                time.sleep(0.005)  # フレームレート制限
                
        except Exception as e:
            logger.error(f"処理済みストリーム生成中エラー: {e}")
    
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
        # 必要に応じてカメラプロパティを更新
        if (camera_manager.width != width or camera_manager.height != height) and camera_manager.camera:
            camera_manager.width = width
            camera_manager.height = height
            # 理想的にはここでカメラのリセットが必要だが、実装は省略
            
        return StreamingResponse(
            get_raw_video_stream(width, height), 
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    @app.get("/video/processed")
    async def processed_video_endpoint(width: Optional[int] = Query(640), height: Optional[int] = Query(480)):
        """
        処理済みビデオストリーミングエンドポイント
        
        画像処理後のMJPEGフォーマットでストリームを提供します
        
        Args:
            width: 要求解像度の幅
            height: 要求解像度の高さ
        
        Returns:
            StreamingResponse: MJPEGストリームのレスポンス
        """
        if not image_processor:
            return HTTPException(status_code=404, detail="画像処理モジュールがありません")
            
        return StreamingResponse(
            get_processed_video_stream(width, height), 
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    @app.get("/status")
    async def status_page():
        """システムステータスページ"""
        return HTMLResponse("""
        <html>
            <head>
                <title>システムステータス</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body { font-family: Arial, sans-serif; padding: 20px; }
                    h1 { color: #333; }
                    .status-container { 
                        background: #f9f9f9;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        padding: 15px;
                        margin: 20px 0;
                        white-space: pre-wrap;
                    }
                    .refresh-btn {
                        background: #007bff;
                        color: white;
                        border: none;
                        padding: 10px 20px;
                        border-radius: 4px;
                        cursor: pointer;
                    }
                </style>
            </head>
            <body>
                <h1>システムステータス</h1>
                <button class="refresh-btn" onclick="refreshStatus()">更新</button>
                <div class="status-container">
                    <pre id="status-data">読み込み中...</pre>
                </div>
                
                <script>
                    async function refreshStatus() {
                        try {
                            const response = await fetch('/api/status');
                            if (response.ok) {
                                const data = await response.json();
                                document.getElementById('status-data').textContent = 
                                    JSON.stringify(data, null, 2);
                            } else {
                                document.getElementById('status-data').textContent = 
                                    'エラー: ' + response.status;
                            }
                        } catch (err) {
                            document.getElementById('status-data').textContent = 
                                'エラー: ' + err.message;
                        }
                    }
                    
                    // 初回読み込み
                    refreshStatus();
                    
                    // 定期更新
                    setInterval(refreshStatus, 3000);
                </script>
            </body>
        </html>
        """)
    
    @app.get("/api/status")
    async def status_endpoint():
        """
        システムステータスAPI
        
        Returns:
            dict: システム情報を含む辞書
        """
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        status_data = {
            "status": "running",
            "timestamp": time.time(),
            "opencv_version": cv2.__version__,
            "system": {
                "cpu_percent": process.cpu_percent(),
                "memory_usage_mb": mem_info.rss / (1024 * 1024),
                "uptime_seconds": time.time() - process.create_time()
            },
            "camera": camera_manager.get_camera_info()
        }
        
        # 画像処理情報の追加
        if image_processor:
            status_data["processor"] = image_processor.get_processor_info()
            status_data["analysis"] = image_processor.get_analysis_data()
            
        return status_data
    
    @app.get("/api/camera-info")
    async def camera_info_endpoint():
        """
        カメラ情報API
        
        Returns:
            dict: カメラ情報を含む辞書
        """
        return camera_manager.get_camera_info()
        
    return app