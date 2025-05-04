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
import collections  # パフォーマンス測定用のdequeを使用するため

app = FastAPI()
MODEL_PATH = '/opt/m5stack/data/depth_anything/compiled.axmodel'

# グローバル変数として最新のカメラフレームと深度マップを保持
latest_frame = None
latest_depth = None
frame_lock = threading.Lock()
depth_lock = threading.Lock()
# デバッグ情報用
depth_processing_count = 0
depth_error_count = 0

# パフォーマンス測定用の変数
perf_timings = {
    'camera_capture': collections.deque(maxlen=50),  # 5秒間のデータ（10FPS想定）
    'depth_inference': collections.deque(maxlen=50), 
    'visualization': collections.deque(maxlen=50),
    'stream_processing': collections.deque(maxlen=150)  # ストリーミングは30FPS想定
}
perf_lock = threading.Lock()
last_perf_report_time = time.time()

# パフォーマンス記録用デコレータ
def time_it(timing_key):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = (time.time() - start_time) * 1000  # ミリ秒に変換
            with perf_lock:
                perf_timings[timing_key].append(elapsed)
            return result
        return wrapper
    return decorator

# パフォーマンスレポートを出力する関数
def report_performance():
    global last_perf_report_time
    current_time = time.time()
    
    # 5秒ごとにレポートを出力
    if current_time - last_perf_report_time >= 5.0:
        with perf_lock:
            report = []
            report.append("\n===== パフォーマンス レポート (ms) =====")
            
            for key, timings in perf_timings.items():
                if timings:
                    avg = sum(timings) / len(timings)
                    min_val = min(timings)
                    max_val = max(timings)
                    count = len(timings)
                    fps = 1000 / avg if avg > 0 else 0
                    report.append(f"{key}: 平均={avg:.2f}, 最小={min_val:.2f}, 最大={max_val:.2f}, " +
                                 f"サンプル={count}, FPS={fps:.1f}")
                else:
                    report.append(f"{key}: データなし")
            
            print("\n".join(report))
            last_perf_report_time = current_time

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

def process_for_depth(frame: np.ndarray, target_size=(384, 256)) -> np.ndarray:  # サイズ縮小
    try:
        # 入力検証と詳細ログ
        if frame is None:
            print("[ERROR] Input frame is None")
            return None
            
        #print(f"[DEBUG] Processing frame: shape={frame.shape}, type={frame.dtype}")
        
        resized = cv2.resize(frame, target_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)  # NHWC uint8
        
        #print(f"[DEBUG] Input tensor: shape={tensor.shape}, type={tensor.dtype}, " 
        #      f"min={tensor.min()}, max={tensor.max()}")
        
        return tensor
    except Exception as e:
        print(f"[ERROR] Failed to process frame for depth: {e}")
        return None

def create_depth_visualization_ori(depth_map: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
    """深度マップの可視化を行う"""
    depth_feature = depth_map.reshape(depth_map.shape[-2:])
    
    # 正規化と色付け
    normalized = (depth_feature - depth_feature.min()) / (depth_feature.max() - depth_feature.min())
    depth_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    
    # 元の画像サイズにリサイズ
    depth_resized = cv2.resize(depth_colored, (original_frame.shape[1], original_frame.shape[0]))

    return depth_resized
    
#    # 元画像と深度マップを横に並べる
#    return np.concatenate([original_frame, depth_resized], axis=1)


def create_depth_visualization(depth_map: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
    try:
        if depth_map is None:
            print("[ERROR] Depth map is None")
            return original_frame
            
        #print(f"[DEBUG] Depth map: shape={depth_map.shape}, type={depth_map.dtype}, "
        #      f"min={depth_map.min()}, max={depth_map.max()}")
            
        depth_feature = depth_map.squeeze()
        
        # 値がNaNや無限大でないかチェック
        if np.isnan(depth_feature).any() or np.isinf(depth_feature).any():
            print("[WARNING] Depth map contains NaN or Inf values, replacing with zeros")
            depth_feature = np.nan_to_num(depth_feature, nan=0.0, posinf=1.0, neginf=0.0)
        
        # データの範囲をチェックして異常値を検知
        depth_min = np.min(depth_feature)
        depth_max = np.max(depth_feature)
        
        # 値の範囲が異常に小さい場合は警告
        if abs(depth_max - depth_min) < 1e-6:
            print(f"[WARNING] Depth range too small: min={depth_min}, max={depth_max}")
            # 最小値と最大値を設定して強制的に範囲を作る
            depth_min = 0
            depth_max = 1
            normalized = np.zeros_like(depth_feature)
        else:
            # 正規化 - ロバストにするためにクリッピングも適用
            normalized = (depth_feature - depth_min) / (depth_max - depth_min + 1e-6)
            # 念のため0-1の範囲に収める
            normalized = np.clip(normalized, 0, 1)
        
        # 正規化データのヒストグラムを文字ベースで出力（簡易的な分布確認）
        if depth_processing_count % 30 == 0:  # 30フレームごとに表示
            hist, _ = np.histogram(normalized, bins=10, range=(0, 1))
            total = hist.sum()
            if total > 0:
                hist_norm = hist / total
                print("[HISTOGRAM]", "".join(["█" * int(h * 50) for h in hist_norm]))
        
        # uint8に変換して色マップを適用
        depth_uint8 = (normalized * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        
        # 深度情報をオーバーレイ表示
        cv2.putText(
            depth_colored,
            f"Min: {depth_min:.4f}, Max: {depth_max:.4f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # 境界線用のグラデーションバーを追加
        h, w = depth_colored.shape[:2]
        gradient_bar = np.zeros((20, w, 3), dtype=np.uint8)
        for x in range(w):
            color_value = int(255 * x / w)
            gradient_bar[:, x] = cv2.applyColorMap(np.array([[color_value]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        
        # テキストラベル追加
        cv2.putText(gradient_bar, "近い", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(gradient_bar, "遠い", (w-40, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # グラデーションバーを結合
        depth_with_scale = np.vstack([depth_colored, gradient_bar])
        
        #print(f"[DEBUG] Depth visualization created: shape={depth_with_scale.shape}")
        
        return depth_with_scale
        
    except Exception as e:
        print(f"[ERROR] Failed to create depth visualization: {e}")
        import traceback
        traceback.print_exc()
        return original_frame

@time_it('camera_capture')
def camera_capture_frame(camera):
    """カメラフレームをキャプチャする関数"""
    # バッファから古いフレームを捨てる
    for _ in range(3):
        camera.grab()
    success, frame = camera.retrieve()
    return success, frame

# カメラ画像を連続的に取得するスレッド関数
def camera_thread():
    global latest_frame
    camera = initialize_camera()
    
    try:
        while True:
            success, frame = camera_capture_frame(camera)
            if not success or frame is None:
                time.sleep(0.01)
                continue
                
            # 最新フレームを更新（スレッドセーフに）
            with frame_lock:
                latest_frame = frame.copy()
            
            # パフォーマンスレポート
            report_performance()
            
            # カメラのフレームレート制御のための短い待機
            time.sleep(0.01)
    finally:
        camera.release()

@time_it('depth_inference')
def run_depth_inference(model, input_name, input_tensor):
    """深度推論を実行する関数"""
    return model.run(None, {input_name: input_tensor})

@time_it('visualization')
def visualize_depth(output, current_frame):
    """深度マップを可視化する関数"""
    return create_depth_visualization_ori(output, current_frame)

def depth_to_point_cloud(depth_map, fx=500, fy=500, cx=192, cy=128):
    """深度マップから点群を生成する
    
    Args:
        depth_map: 深度マップ (H, W)
        fx, fy: 焦点距離
        cx, cy: 画像中心
    
    Returns:
        points: 3D点群 (N, 3)
    """
    try:
        height, width = 384, 256
       
        # 深度マップの値をチェック
        if np.isnan(depth_map).any() or np.isinf(depth_map).any():
            print("[警告] 深度マップにNaNまたは無限大の値があります。修正します。")
            depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=10.0, neginf=0.0)
            
        # データ範囲をチェック
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        if depth_min < 0 or depth_max > 100:  # 合理的な範囲をチェック
            print(f"[警告] 深度値が異常です: 最小={depth_min}, 最大={depth_max}")
            depth_map = np.clip(depth_map, 0.0, 10.0)  # 安全な範囲にクリップ
        
        # 画像座標のグリッドを作成
        x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
        
        # カメラ座標系に変換
        X = (x_grid - cx) * depth_map / fx
        Y = (y_grid - cy) * depth_map / fy
        Z = depth_map
        
        # 形状を変形して点群に
#        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
#        
#        # 無効な点（深度値が0に近い）を除外
#        valid_points = points[points[:, 2] > 0.1]
#        
#        return valid_points
        return np.zeros((0, 3))  # 空の点群を返す
    
    except Exception as e:
        print(f"[エラー] 点群変換中にエラーが発生しました: {e}")
        return np.zeros((0, 3))  # 空の点群を返す


# 深度推論を行うスレッド関数
def depth_processing_thread():
    global latest_depth, latest_frame, depth_processing_count, depth_error_count
    
    print("[INFO] Starting depth processing thread")
    try:
        model = initialize_model(MODEL_PATH)
        
        # モデルの入力名を確認
        inputs = model.get_inputs()
        if not inputs:
            print("[ERROR] Model has no inputs")
            return
            
        input_name = inputs[0].name
        print(f"[INFO] Using model input name: {input_name}")
        
        # 前回処理したフレームを記録
        last_processed_frame = None
        
        print("[INFO] Depth processing thread ready")
        
        while True:
            try:
                current_frame = None
                
                # 最新フレームを取得（スレッドセーフに）
                with frame_lock:
                    if latest_frame is not None:
                        current_frame = latest_frame.copy()
                
                # 新しいフレームがある場合のみ処理
                if current_frame is not None:
                    # フレーム処理とモデル推論
                    input_tensor = process_for_depth(current_frame)
                    
                    if input_tensor is not None:
                        # 深度推論実行
                        outputs = run_depth_inference(model, input_name, input_tensor)
                        
                        if outputs is None or len(outputs) == 0:
                            print("[ERROR] Model returned no outputs")
                        else:
                            output = outputs[0]
                            
                            # 深度の可視化
                            depth_vis = visualize_depth(output, current_frame)
                            print(f"[INFO] Depth visualization created: {depth_vis.shape}")

                            # 深度マップを点群に変換
                            points = depth_to_point_cloud(output)                           
                            print(f"[INFO] Point cloud generated: {points.shape[0]} points") 
                            
                            # 深度マップを更新（スレッドセーフに）
                            with depth_lock:
                                latest_depth = depth_vis
                                
                            depth_processing_count += 1
                            if depth_processing_count % 10 == 0:
                                print(f"[INFO] Processed {depth_processing_count} depth frames successfully")
                    else:
                        print("[ERROR] Failed to prepare input tensor")
                        depth_error_count += 1
                        
                # パフォーマンスレポート
                report_performance()
                        
                # 処理間隔を調整（必要に応じて）
                time.sleep(0.1)  # 深度処理は10FPS程度で十分
                
            except Exception as e:
                depth_error_count += 1
                print(f"[ERROR] Depth processing iteration failed: {e}")
                if depth_error_count > 10:
                    print("[CRITICAL] Too many errors in depth processing, retrying model initialization")
                    try:
                        model = initialize_model(MODEL_PATH)
                        input_name = model.get_inputs()[0].name
                        depth_error_count = 0
                    except Exception as reinit_error:
                        print(f"[CRITICAL] Failed to reinitialize model: {reinit_error}")
                
                time.sleep(1.0)  # エラー発生時は少し長めに待機
    except Exception as e:
        print(f"[CRITICAL] Depth thread fatal error: {e}")
        import traceback
        traceback.print_exc()

@time_it('stream_processing')
def process_stream_frame():
    """ストリーミング用のフレームを生成する関数"""
    global latest_frame, latest_depth
    empty_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    
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
        
    # 深度マップがない場合はグレースケールのプレースホルダー
    if current_depth is None:
        current_depth = np.zeros_like(current_frame)
        # 「DEPTH PROCESSING...」テキストを表示
        cv2.putText(
            current_depth, 
            "DEPTH PROCESSING...", 
            (10, 120), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
    # フレームを並べて表示
    # 小さいサイズを使用してパフォーマンス向上
    combined = np.hstack([
        cv2.resize(current_frame, (320, 240)), 
        cv2.resize(current_depth, (320, 240))
    ])
    
    return combined

# ストリーミング用のジェネレーター関数
def get_video_stream():
    frame_count = 0
    
    try:
        while True:
            start_time = time.time()
            
            # ストリームフレームを生成
            combined = process_stream_frame()
            
            # フレーム番号を追加
            frame_count += 1
            cv2.putText(
                combined, 
                f"Frame: {frame_count} Depth: {depth_processing_count}", 
                (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
            
            # パフォーマンス情報を表示
            with perf_lock:
                if 'depth_inference' in perf_timings and perf_timings['depth_inference']:
                    avg_inf = sum(perf_timings['depth_inference']) / len(perf_timings['depth_inference'])
                    cv2.putText(
                        combined, 
                        f"Inf: {avg_inf:.1f}ms", 
                        (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        1
                    )
            
            ret, buffer = cv2.imencode('.jpg', combined)
            if ret:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
            # パフォーマンスレポート
            report_performance()
                      
            # 処理時間計測とフレームレート調整
            process_time = time.time() - start_time
            if process_time < 0.033:  # 目標は約30FPS
                time.sleep(0.033 - process_time)
    except Exception as e:
        print(f"[ERROR] Streaming error: {e}")
        import traceback
        traceback.print_exc()

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
