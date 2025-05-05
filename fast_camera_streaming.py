from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import time
import threading
import queue
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

app = FastAPI()

# グローバル変数
frame_queue = queue.Queue(maxsize=1)  # 最新のフレームだけを保持するキュー
depth_image_queue = queue.Queue(maxsize=1)  # 深度画像キュー
depth_data_queue = queue.Queue(maxsize=1)  # 深度データキュー
process_thread = None
is_running = True
depth_model = None  # Depth Anythingモデル

# モデル設定
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE}")

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Depth Camera Stream</title>
        </head>
        <body>
            <h1>カメラ映像と深度推定</h1>
            <div style="display: flex; justify-content: center;">
                <div style="margin-right: 20px;">
                    <h3>RGB画像</h3>
                    <img src="/video" width="640" height="480" />
                </div>
                <div>
                    <h3>深度推定</h3>
                    <img src="/depth_video" width="640" height="480" />
                </div>
            </div>
        </body>
    </html>
    """

def initialize_depth_model():
    """Depth Anythingモデルを初期化"""
    try:
        print("[INFO] Loading Depth Anything model...")
        
        # Depth Anything モデルを読み込む
        # ここでは簡易的なモデル初期化の例を示します
        # 実際には正しいパスとモデル設定が必要です
        try:
            # transformers libraryから読み込む場合
            from transformers import AutoModelForDepthEstimation
            model_id = "LiheYoung/depth-anything-small-hf"
            model = AutoModelForDepthEstimation.from_pretrained(model_id)
            model.to(DEVICE)
            model.eval()
            print("[INFO] Depth Anything model loaded successfully")
            return model
        except ImportError:
            print("[WARN] Failed to import from transformers, trying custom implementation...")
            
            # 自前で実装する場合のスタブコード
            # 実際のモデルに置き換えてください
            class DummyDepthModel:
                def __init__(self):
                    pass
                
                def to(self, device):
                    return self
                
                def eval(self):
                    return self
                
                def __call__(self, x):
                    # ダミー深度マップを返す
                    h, w = x.shape[-2:]
                    fake_depth = torch.zeros((1, 1, h, w), device=x.device)
                    # 深度のグラデーションを生成
                    for i in range(h):
                        fake_depth[0, 0, i, :] = i / h
                    return {"predicted_depth": fake_depth}
            
            model = DummyDepthModel()
            print("[INFO] Dummy depth model initialized")
            return model
            
    except Exception as e:
        print(f"[ERROR] Failed to initialize depth model: {e}")
        return None

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
            time.sleep(0.005)

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

def preprocess_image(frame):
    """画像を前処理してモデル入力用に変換"""
    # PIL画像に変換
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # 前処理
    preprocess = transforms.Compose([
        transforms.Resize((384, 384)),  # モデルの入力サイズに合わせる
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(pil_image).unsqueeze(0)  # バッチ次元を追加
    return input_tensor

def colorize_depth(depth_map):
    """深度マップをカラー画像に変換"""
    # 正規化
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    normalized_depth = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    
    # カラーマップの適用
    colored = plt.cm.viridis(normalized_depth)  # viridisカラーマップを使用
    colored = (colored[:, :, :3] * 255).astype(np.uint8)  # アルファチャンネルを除去し、uint8に変換
    
    return colored

def depth_processing_thread():
    """深度推論を行うスレッド"""
    global is_running, depth_model
    print("[INFO] Depth processing thread started")
    
    # モデル初期化
    depth_model = initialize_depth_model()
    
    if depth_model is None:
        print("[ERROR] Failed to initialize depth model. Thread stopping.")
        return
    
    frame_count = 0
    skipped_frames = 0
    
    while is_running:
        try:
            # キューからフレームを取得
            frame = frame_queue.get(timeout=1.0)
            
            # 10フレームごとに処理（処理負荷軽減のため）
            frame_count += 1
            if frame_count % 10 != 0:
                skipped_frames += 1
                continue
                
            # 画像の前処理
            input_tensor = preprocess_image(frame)
            input_tensor = input_tensor.to(DEVICE)
            
            # 深度推論
            with torch.no_grad():  # 勾配計算不要
                start_time = time.time()
                output = depth_model(input_tensor)
                inference_time = time.time() - start_time
                
            # 深度マップを取得
            if isinstance(output, dict) and "predicted_depth" in output:
                depth_map = output["predicted_depth"]
            else:
                # モデル出力形式に合わせて調整
                depth_map = output
                
            # GPU上のテンソルをCPUに移し、NumPy配列に変換
            depth_map = depth_map.squeeze().cpu().numpy()
            
            # 深度データをキューに追加
            try:
                if depth_data_queue.full():
                    depth_data_queue.get_nowait()
                depth_data_queue.put_nowait(depth_map)
            except:
                pass
            
            # 深度マップをカラライズして可視化
            colored_depth = colorize_depth(depth_map)
            
            # 元画像のサイズにリサイズ
            h, w = frame.shape[:2]
            colored_depth_resized = cv2.resize(colored_depth, (w, h))
            
            # 深度画像をキューに追加
            try:
                if depth_image_queue.full():
                    depth_image_queue.get_nowait()
                depth_image_queue.put_nowait(colored_depth_resized)
            except:
                pass
                
            print(f"[INFO] Depth inference completed in {inference_time:.3f}s, " 
                  f"shape: {depth_map.shape}, range: {depth_map.min():.3f}-{depth_map.max():.3f}")
            
        except queue.Empty:
            # タイムアウト - 何もしない
            pass
        except Exception as e:
            print(f"[ERROR] Error in depth processing thread: {e}")
            time.sleep(0.5)
    
    print(f"[INFO] Depth processing thread stopped. Processed {frame_count} frames, skipped {skipped_frames} frames.")

@app.get("/video")
async def video_endpoint():
    """ビデオストリームのエンドポイント"""
    return StreamingResponse(get_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/depth_video")
async def depth_video_endpoint():
    """深度画像ストリームのエンドポイント"""
    return StreamingResponse(get_depth_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時に処理スレッドを開始"""
    global process_thread
    process_thread = threading.Thread(target=depth_processing_thread, daemon=True)
    process_thread.start()
    print("[INFO] Started depth processing thread")

@app.on_event("shutdown")
async def shutdown_event():
    """アプリケーション終了時に処理スレッドを停止"""
    global is_running
    is_running = False
    if process_thread:
        process_thread.join(timeout=2.0)  # 最大2秒待機
    print("[INFO] Stopped depth processing thread")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)