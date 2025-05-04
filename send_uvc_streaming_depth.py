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

app = FastAPI()
MODEL_PATH = '/opt/m5stack/data/depth_anything/compiled.axmodel'

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

def get_video_stream():
    camera = initialize_camera()
    model = initialize_model(MODEL_PATH)
    input_name = model.get_inputs()[0].name
    
    # 並列処理用のキューとエグゼキューター
    frame_queue = queue.Queue(maxsize=2)
    result_queue = queue.Queue(maxsize=2)
    executor = ThreadPoolExecutor(max_workers=1)
    
    def process_depth_async(input_frame):
        try:
            input_tensor = process_for_depth(input_frame)
            output = model.run(None, {input_name: input_tensor})[0]
            depth_vis = create_depth_visualization(output, input_frame)
            return depth_vis
        except Exception as e:
            print(f"[ERROR] Depth processing failed: {e}")
            return input_frame
    
    # フレームスキップを有効化して処理負荷を軽減
    frame_skip = 3  # より多くスキップ
    frame_count = 0
    
    # 初期フレーム取得
    try:
        while True:
            # バッファから古いフレームを捨てる
            for _ in range(3):
                camera.grab()
            success, frame = camera.retrieve()
            if not success or frame is None:
                time.sleep(0.01)
                continue
            
            # フレームスキップ処理
            frame_count += 1
            if frame_count % frame_skip != 0:
                # 軽量な処理だけ行ってJPEGを返す
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue
                
            # 非同期で深度処理を実行
            if frame_queue.qsize() < 2:
                frame_queue.put(frame)
                future = executor.submit(process_depth_async, frame.copy())
                
                # 前の結果が利用可能なら表示
                try:
                    if not result_queue.empty():
                        depth_vis = result_queue.get(block=False)
                        # 小さいサイズを使用してパフォーマンス向上
                        combined = np.hstack([cv2.resize(frame, (320, 240)), 
                                             cv2.resize(depth_vis, (320, 240))])
                        ret, buffer = cv2.imencode('.jpg', combined)
                        if ret:
                            yield (b'--frame\r\n'
                                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    else:
                        # 初回は元のフレームだけ表示
                        combined = np.hstack([frame, frame])
                        ret, buffer = cv2.imencode('.jpg', combined)
                        if ret:
                            yield (b'--frame\r\n'
                                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
                    # 完了した深度処理結果を取得
                    if future.done():
                        result_queue.put(future.result())
                except Exception as e:
                    print(f"[ERROR] Frame processing error: {e}")
                    
            time.sleep(0.01)  # スリープ時間を延長して負荷軽減
    finally:
        executor.shutdown()
        camera.release()

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
        
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
