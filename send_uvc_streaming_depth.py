from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import time
import axengine as axe
import subprocess
from concurrent.futures import ThreadPoolExecutor
import queue
import os
import threading

app = FastAPI()
MODEL_PATH = '/opt/m5stack/data/depth_anything/compiled.axmodel'

# グローバル変数として最新のカメラフレームと深度マップを保持
latest_frame = None
latest_depth = None
frame_lock = threading.Lock()
depth_lock = threading.Lock()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>USBカメラ ストリーム</title>
        </head>
        <body>
            <h1>USBカメラの映像を表示中</h1>
            <img src="/video" width="1280" height="480" />
        </body>
    </html>
    """

def initialize_camera(index=0, width=320, height=240):  # 解像度を下げる
    cam = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cam

def initialize_model(model_path: str):
    # AXENGINEの最適化設定を追加
    options = {}
    options["axe.input_layout"] = "NHWC"  # 入力レイアウトを明示
    options["axe.output_layout"] = "NHWC" # 出力レイアウトを明示
    options["axe.use_dsp"] = "true"       # DSP使用を有効化
    
    session = axe.InferenceSession(model_path, options)
    print("[INFO] Depth model loaded with optimizations")
    return session

def process_for_depth(frame: np.ndarray, target_size=(256, 192)) -> np.ndarray:  # サイズ縮小
    resized = cv2.resize(frame, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)  # NHWC uint8
    return tensor

def create_depth_visualization(depth_map: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
    depth_feature = depth_map.squeeze()
    # 固定値で正規化して計算コスト削減
    normalized = np.clip((depth_feature - 0.1) / 0.8, 0, 1)
    
    # 効率的な色変換
    depth_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    # 元画像を縮小して合わせる（高速化）
    resized_original = cv2.resize(original_frame, (depth_colored.shape[1], depth_colored.shape[0]))
    return depth_colored  # 元のサイズに戻さない（表示時にリサイズ）

# カメラ画像を連続的に取得するスレッド関数
def camera_thread():
    global latest_frame
    camera = initialize_camera()
    
    try:
        while True:
            # バッファから古いフレームを捨てる
            for _ in range(3):
                camera.grab()
            success, frame = camera.retrieve()
            if not success or frame is None:
                time.sleep(0.01)
                continue
                
            # 最新フレームを更新（スレッドセーフに）
            with frame_lock:
                latest_frame = frame.copy()
            
            # カメラのフレームレート制御のための短い待機
            time.sleep(0.01)
    finally:
        camera.release()

# 深度推論を行うスレッド関数
def depth_processing_thread():
    global latest_depth, latest_frame
    model = initialize_model(MODEL_PATH)
    input_name = model.get_inputs()[0].name
    
    # 前回処理したフレームを記録
    last_processed_frame = None
    
    try:
        while True:
            current_frame = None
            
            # 最新フレームを取得（スレッドセーフに）
            with frame_lock:
                if latest_frame is not None:
                    current_frame = latest_frame.copy()
            
            # 新しいフレームがあり、前回処理したものと異なる場合のみ処理
            if current_frame is not None:
                try:
                    # フレームハッシュを比較する代わりに、一定間隔でのみ処理
                    input_tensor = process_for_depth(current_frame)
                    output = model.run(None, {input_name: input_tensor})[0]
                    depth_vis = create_depth_visualization(output, current_frame)
                    
                    # 深度マップを更新（スレッドセーフに）
                    with depth_lock:
                        latest_depth = depth_vis
                        
                    last_processed_frame = current_frame
                except Exception as e:
                    print(f"[ERROR] Depth processing failed: {e}")
                    
            # 処理間隔を調整（必要に応じて）
            time.sleep(0.1)  # 深度処理は10FPS程度で十分
    except Exception as e:
        print(f"[ERROR] Depth thread error: {e}")

# ストリーミング用のジェネレーター関数
def get_video_stream():
    global latest_frame, latest_depth
    empty_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    
    try:
        while True:
            # 最新のカメラフレームと深度マップを取得（スレッドセーフに）
            current_frame = None
            current_depth = None
            
            with frame_lock:
                if latest_frame is not None:
                    current_frame = latest_frame.copy()
            
            with depth_lock:
                if latest_depth is not None:
                    current_depth = latest_depth.copy()
            
            # カメラフレームがない場合は空のフレーム
            if current_frame is None:
                current_frame = empty_frame
                
            # 深度マップがない場合はカメラフレームを複製
            if current_depth is None:
                current_depth = current_frame
                
            # フレームを並べて表示
            # 小さいサイズを使用してパフォーマンス向上
            combined = np.hstack([
                cv2.resize(current_frame, (320, 240)), 
                cv2.resize(current_depth, (320, 240))
            ])
            
            ret, buffer = cv2.imencode('.jpg', combined)
            if ret:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                      
            # ストリーミングのフレームレート調整
            time.sleep(0.033)  # 約30FPSでストリーミング
    except Exception as e:
        print(f"[ERROR] Streaming error: {e}")

@app.get("/video")
async def video_endpoint():
    return StreamingResponse(get_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    # M5Stackのシステムリソース最適化
    try:
        # CPU優先度を最大に設定
        os.system("sudo renice -n -20 -p $(pgrep -f 'python.*send_uvc_streaming_depth.py')")
        os.system("sudo ionice -c 1 -n 0 -p $(pgrep -f 'python.*send_uvc_streaming_depth.py')")
    except:
        pass
        
    try:
        # M5StackのGPU/DSPメモリを最大化
        subprocess.call("sudo sh -c 'echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'", shell=True)
    except:
        pass
    
    # カメラスレッドと深度処理スレッドを開始
    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=depth_processing_thread, daemon=True).start()
    print("[INFO] Camera and depth processing threads started")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
