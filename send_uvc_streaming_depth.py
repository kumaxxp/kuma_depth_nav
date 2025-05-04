from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import time
import axengine as axe
import subprocess
import os
import threading
import queue

app = FastAPI()
MODEL_PATH = '/opt/m5stack/data/depth_anything/compiled.axmodel'

# グローバル変数として最新のカメラフレームと深度マップを保持
depth_queue = queue.Queue(maxsize=1)  # 常に最新の深度マップのみを保持
frame_lock = threading.Lock()
processing_count = 0

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

def initialize_camera(index=0, width=320, height=240):
    # v4l2-ctl を使ってサポートされる基本的な設定のみを適用
    try:
        subprocess.run(["v4l2-ctl", f"--set-fmt-video=width={width},height={height},pixelformat=MJPG"], check=False)
        subprocess.run(["v4l2-ctl", "--set-parm=30"], check=False)
    except Exception as e:
        print(f"[WARN] Failed to apply v4l2-ctl settings: {e}")

    cam = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cam

def initialize_model(model_path: str):
    try:
        print(f"[INFO] Loading model from {model_path}")
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # axengineのバージョンに応じて適切な初期化方法を試す
        try:
            # オプションを使用した初期化（新しいバージョン向け）
            options = {}
            options["axe.input_layout"] = "NHWC"  # 入力レイアウトを明示
            options["axe.output_layout"] = "NHWC" # 出力レイアウトを明示
            options["axe.use_dsp"] = "true"       # DSP使用を有効化
            
            session = axe.InferenceSession(model_path, options)
        except TypeError:
            # オプションなしで初期化（古いバージョン向け）
            print("[INFO] Falling back to basic model initialization")
            session = axe.InferenceSession(model_path)
            
        # モデル情報を表示
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        print(f"[INFO] Model inputs: {[x.name for x in inputs]}")
        print(f"[INFO] Model input shapes: {[x.shape for x in inputs]}")
        print(f"[INFO] Model outputs: {[x.name for x in outputs]}")
        print(f"[INFO] Model output shapes: {[x.shape for x in outputs]}")
            
        print("[INFO] Depth model loaded successfully")
        return session
    except Exception as e:
        print(f"[ERROR] Model initialization failed: {e}")
        raise

def process_for_depth(frame: np.ndarray, target_size=(256, 192)) -> np.ndarray:
    # カメラフレームを深度モデル用に前処理
    resized = cv2.resize(frame, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)  # NHWC uint8
    return tensor

def create_depth_visualization(depth_map: np.ndarray) -> np.ndarray:
    try:
        if depth_map is None:
            return None
            
        depth_feature = depth_map.squeeze()
        
        # 動的な正規化 - 現在のフレームの最小・最大値に基づいて正規化
        depth_min = np.min(depth_feature)
        depth_max = np.max(depth_feature)
        
        # 正規化範囲を調整して、より鮮明な視覚化を行う
        normalized = (depth_feature - depth_min) / (depth_max - depth_min + 1e-6)
        
        # 色マップを適用
        depth_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # 深度値を示すテキストを重ねる
        cv2.putText(
            depth_colored,
            f"Min: {depth_min:.2f}, Max: {depth_max:.2f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        return depth_colored
    except Exception as e:
        print(f"[ERROR] Failed to create depth visualization: {e}")
        return None

# 深度推論を行うバックグラウンドスレッド
def depth_processing_thread():
    global processing_count
    
    print("[INFO] Starting depth processing thread")
    model = initialize_model(MODEL_PATH)
    
    input_name = model.get_inputs()[0].name
    print(f"[INFO] Using model input name: {input_name}")
    
    print("[INFO] Depth processing thread ready")
    
    while True:
        try:
            # 深度キューからフレームを取得
            frame = None
            
            with frame_lock:
                if not frame_queue.empty():
                    frame = frame_queue.get_nowait()
            
            if frame is not None:
                # 深度処理を実行
                input_tensor = process_for_depth(frame)
                outputs = model.run(None, {input_name: input_tensor})
                
                if outputs and len(outputs) > 0:
                    # 深度マップを視覚化
                    depth_vis = create_depth_visualization(outputs[0])
                    
                    # すでにキューにあるデータを古いとして破棄し、新しい結果だけを保持
                    while not depth_queue.empty():
                        depth_queue.get_nowait()
                    
                    depth_queue.put(depth_vis)
                    processing_count += 1
            
            # 処理間隔を調整
            time.sleep(0.05)  # 20FPS程度で十分
            
        except Exception as e:
            print(f"[ERROR] Depth processing error: {e}")
            time.sleep(1.0)  # エラー時は少し長めに待機

# キューにフレームを追加するためのグローバル変数
frame_queue = queue.Queue(maxsize=2)

def get_video_stream():
    camera = initialize_camera()
    times = []  # パフォーマンス測定用
    last_report = time.time()
    frame_count = 0
    
    # 空のデプスマップ用のプレースホルダー
    empty_depth = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(empty_depth, "DEPTH PROCESSING...", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    try:
        while True:
            start_time = time.perf_counter()

            # カメラフレームを取得
            if not camera.isOpened():
                print("[WARN] Camera not open. Retrying...")
                camera.release()
                time.sleep(0.5)
                camera = initialize_camera()
                continue

            success, frame = camera.read()
            if not success or frame is None:
                print("[WARN] Failed to read frame. Skipping...")
                time.sleep(0.01)
                continue

            # 深度処理スレッドに最新フレームを渡す
            with frame_lock:
                # キューがいっぱいなら古いフレームを破棄
                while not frame_queue.empty() and frame_queue.qsize() >= 2:
                    frame_queue.get_nowait()
                frame_queue.put(frame.copy())
            
            # 現在の深度マップを取得（利用可能であれば）
            current_depth = None
            if not depth_queue.empty():
                current_depth = depth_queue.get()
            else:
                current_depth = empty_depth
            
            # フレームを結合
            if current_depth is not None and current_depth.shape[0] > 0:
                # リサイズして横に並べる
                combined = np.hstack([
                    cv2.resize(frame, (320, 240)),
                    cv2.resize(current_depth, (320, 240))
                ])
            else:
                # 深度マップがない場合はカメラ映像を複製
                combined = np.hstack([
                    cv2.resize(frame, (320, 240)), 
                    empty_depth
                ])
            
            # フレーム番号とカウンターを表示
            frame_count += 1
            cv2.putText(
                combined,
                f"Frame: {frame_count} Depth: {processing_count}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # JPEG変換して送信
            ret, buffer = cv2.imencode('.jpg', combined)
            if ret:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # パフォーマンス測定
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
            
            # パフォーマンスレポート
            if time.time() - last_report >= 5.0:
                if times:
                    avg_time = sum(times) / len(times)
                    max_time = max(times)
                    min_time = min(times)
                    fps = len(times) / 5.0
                    print(f"[PERF] Avg: {avg_time:.4f}s, Max: {max_time:.4f}s, Min: {min_time:.4f}s, FPS: {fps:.1f}")
                    times.clear()
                last_report = time.time()
            
            # フレームレートを調整
            time.sleep(0.005)  # 軽いスリープで負荷軽減

    finally:
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
    
    # 深度処理スレッドを開始
    threading.Thread(target=depth_processing_thread, daemon=True).start()
    print("[INFO] Depth processing thread started")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
