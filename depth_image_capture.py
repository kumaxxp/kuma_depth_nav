#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度画像キャプチャツール

depth_occupancy_mapping.pyで使用するための深度画像をキャプチャするためのツール。
カメラからの深度データをキャプチャして保存します。
Linux環境に対応。WebインターフェースでSSH環境でも使用可能。
"""

import numpy as np
import cv2
import argparse
import os
import time
import sys
import gc  # ガベージコレクションのための追加
import traceback
import threading
import queue
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn

# FastAPIアプリケーションの初期化
app = FastAPI()

# グローバル変数
latest_frame = None
latest_depth = None
frame_lock = threading.Lock()
depth_lock = threading.Lock()
capture_event = threading.Event()
stop_event = threading.Event()
frame_count = 0
output_dir = "depth_captures"
depth_pattern = "objects"

def initialize_camera(index=0, width=320, height=240):
    """カメラを初期化する"""
    try:
        # Linux環境ではCAP_V4L2を使用
        cam = cv2.VideoCapture(index, cv2.CAP_V4L2)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not cam.isOpened():
            print("[エラー] カメラを開けませんでした")
            return None
        print("[INFO] カメラが正常に初期化されました")
        return cam
    except Exception as e:
        print(f"[エラー] カメラ初期化中にエラーが発生しました: {e}")
        traceback.print_exc()
        return None

def camera_capture_frame(camera):
    """バッファをクリアしてカメラフレームを取得する"""
    if camera is None:
        return False, None
    
    # バッファから古いフレームを捨てる
    for _ in range(3):
        camera.grab()
        
    success, frame = camera.retrieve()
    return success, frame

def process_frame(frame: np.ndarray, target_size=(384, 256)) -> np.ndarray:
    """フレームを処理してモデル入力用に準備する"""
    if frame is None or frame.size == 0:
        raise ValueError("空のフレームが入力されました。")
    
    resized = cv2.resize(frame, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # 常にNHWCフォーマットのuint8テンソルを返す
    tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)  # NHWC uint8
    if tensor.nbytes % np.dtype(np.uint8).itemsize != 0:
        raise ValueError("[エラー] テンソルバッファサイズがデータ型と一致しません")
    return tensor

def create_synthetic_depth(frame, pattern="random"):
    """
    テスト用の合成深度マップを生成する
    
    Args:
        frame: カメラフレーム
        pattern: 生成パターン ("random", "gradient", "objects")
    
    Returns:
        depth_map: 生成された深度マップ
    """
    h, w = frame.shape[:2]
    
    if pattern == "random":
        # ランダムな深度マップ
        depth_map = np.random.uniform(0.5, 5.0, (h, w)).astype(np.float32)
    
    elif pattern == "gradient":
        # グラデーション深度マップ
        x = np.linspace(0, 1, w)
        depth_map = np.tile(x, (h, 1))
        depth_map = depth_map * 5.0  # 0-5mの範囲にスケール
        
    elif pattern == "objects":
        # 物体を含む深度マップ
        depth_map = np.ones((h, w), dtype=np.float32) * 5.0  # 背景は5m
        
        # 中央に円形の物体
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 4
        
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = dist_from_center <= radius
        depth_map[mask] = 2.0  # 中央の物体は2m
        
        # 左側に四角形の物体
        x_start = w // 4 - radius // 2
        x_end = w // 4 + radius // 2
        y_start = h // 2 - radius // 2
        y_end = h // 2 + radius // 2
        depth_map[y_start:y_end, x_start:x_end] = 1.0  # 左の物体は1m
        
        # 右側に四角形の物体
        x_start = 3 * w // 4 - radius // 2
        x_end = 3 * w // 4 + radius // 2
        y_start = h // 2 - radius // 2
        y_end = h // 2 + radius // 2
        depth_map[y_start:y_end, x_start:x_end] = 1.5  # 右の物体は1.5m
    
    else:
        # デフォルト - グレースケールベースの深度マップ
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 明るい部分を近く、暗い部分を遠くに
        depth_map = 5.0 - (gray / 255.0 * 4.5)
    
    # ノイズを追加
    noise = np.random.normal(0, 0.05, depth_map.shape)
    depth_map += noise
    depth_map = np.clip(depth_map, 0.1, 10.0)  # 値の範囲をクリップ
    
    return depth_map

def create_depth_visualization(depth_map: np.ndarray, original_frame: np.ndarray, with_gradient_bar=False) -> np.ndarray:
    """深度マップのカラー表示を作成する"""
    try:
        depth_feature = depth_map.reshape(depth_map.shape[-2:])
        
        # NaNや無限大の値をチェック
        if np.isnan(depth_feature).any() or np.isinf(depth_feature).any():
            print("[警告] 深度マップにNaNまたは無限大の値があります。修正します。")
            depth_feature = np.nan_to_num(depth_feature, nan=0.0, posinf=10.0, neginf=0.0)
            
        # 最小値と最大値をチェック
        depth_min = np.min(depth_feature)
        depth_max = np.max(depth_feature)
        
        # 値の範囲が異常に小さい場合
        if abs(depth_max - depth_min) < 1e-6:
            print(f"[警告] 深度の範囲が小さすぎます: min={depth_min}, max={depth_max}")
            normalized = np.zeros_like(depth_feature)
        else:
            normalized = (depth_feature - depth_min) / (depth_max - depth_min + 1e-6)
            normalized = np.clip(normalized, 0, 1)  # 0-1の範囲に収める
        
        # JET カラーマップを使用 (send_uvc_streaming_depthと同様)
        depth_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        depth_resized = cv2.resize(depth_colored, (original_frame.shape[1], original_frame.shape[0]))
        
        # 深度情報を表示
        cv2.putText(
            depth_resized,
            f"Depth: Min={depth_min:.2f}, Max={depth_max:.2f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        if not with_gradient_bar:
            return depth_resized
        
        # 境界線用のグラデーションバーを追加（必要な場合のみ）
        h, w = depth_resized.shape[:2]
        gradient_bar = np.zeros((20, w, 3), dtype=np.uint8)
        for x in range(w):
            color_value = int(255 * x / w)
            gradient_bar[:, x] = cv2.applyColorMap(np.array([[color_value]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
        
        # テキストラベル追加
        cv2.putText(gradient_bar, "近い", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(gradient_bar, "遠い", (w-40, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # グラデーションバーを結合
        depth_with_scale = np.vstack([depth_resized, gradient_bar])
        
        return depth_with_scale
    except Exception as e:
        print(f"[エラー] 深度マップの可視化に失敗しました: {e}")
        traceback.print_exc()
        # エラーが発生した場合、元のフレームを返す
        return original_frame.copy()

def save_depth_data(frame, depth_map, timestamp, output_dir):
    """深度データとカメラフレームを保存する"""
    # 出力ディレクトリがなければ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # ファイル名を生成
    rgb_filename = os.path.join(output_dir, f"rgb_{timestamp}.jpg")
    depth_vis_filename = os.path.join(output_dir, f"depth_vis_{timestamp}.jpg")
    depth_raw_filename = os.path.join(output_dir, f"depth_raw_{timestamp}.npy")
    
    # RGBフレームを保存
    cv2.imwrite(rgb_filename, frame)
    
    # 深度マップの可視化（保存用にはグラデーションバー付き）を保存
    depth_vis = create_depth_visualization(depth_map, frame, with_gradient_bar=True)
    cv2.imwrite(depth_vis_filename, depth_vis)
    
    # 生の深度データをnumpy形式で保存（後処理用）
    np.save(depth_raw_filename, depth_map)
    
    print(f"[情報] データを保存しました: {rgb_filename}, {depth_vis_filename}, {depth_raw_filename}")
    return rgb_filename, depth_vis_filename, depth_raw_filename

# カメラスレッド関数
def camera_thread_function(camera_index, width, height, pattern):
    global latest_frame, latest_depth, frame_count, depth_pattern
    
    depth_pattern = pattern
    camera = initialize_camera(camera_index, width, height)
    if camera is None:
        print("[エラー] カメラを初期化できません")
        return
    
    try:
        print("[情報] カメラスレッド開始")
        while not stop_event.is_set():
            start_time = time.time()
            
            success, frame = camera_capture_frame(camera)
            if not success or frame is None:
                print("[警告] フレームの取得に失敗しました")
                time.sleep(0.1)
                continue
            
            # フレームからの合成深度マップを作成
            depth_map = create_synthetic_depth(frame, depth_pattern)
            
            # 最新フレームと深度を更新（スレッドセーフに）
            with frame_lock:
                latest_frame = frame.copy()
            
            with depth_lock:
                latest_depth = depth_map.copy()
                
            # カウンター更新
            frame_count += 1
            
            # キャプチャイベントが設定されていたら画像を保存
            if capture_event.is_set():
                timestamp = int(time.time() * 1000)
                save_depth_data(frame, depth_map, timestamp, output_dir)
                capture_event.clear()
                
            # フレームレート制御
            process_time = time.time() - start_time
            if process_time < 0.033:  # 目標30FPS
                time.sleep(0.033 - process_time)
                
    except Exception as e:
        print(f"[エラー] カメラスレッドでエラーが発生しました: {e}")
        traceback.print_exc()
    finally:
        if camera is not None:
            camera.release()
        print("[情報] カメラスレッド終了")

# ストリーミング用のジェネレーター関数
def generate_frames():
    global latest_frame, latest_depth, frame_count
    
    empty_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    
    try:
        while not stop_event.is_set():
            start_time = time.time()
            
            # 最新フレームと深度マップを取得（スレッドセーフに）
            current_frame = None
            current_depth_map = None
            
            with frame_lock:
                if latest_frame is not None:
                    current_frame = latest_frame.copy()
            
            with depth_lock:
                if latest_depth is not None:
                    current_depth_map = latest_depth.copy()
            
            # フレームとデプスマップが取得できない場合は空のフレームを使用
            if current_frame is None:
                current_frame = empty_frame
                
            if current_depth_map is None:
                # 空のフレームの場合は灰色の深度マップを作成
                current_depth_map = np.ones_like(current_frame[:,:,0]).astype(np.float32) * 2.5
            
            # 深度マップの可視化（表示用）
            depth_vis = create_depth_visualization(current_depth_map, current_frame, with_gradient_bar=False)
            
            # 表示用の画像を作成
            display_img = np.hstack([current_frame, depth_vis])
            
            # フレーム番号を追加
            cv2.putText(
                display_img, 
                f"Frame: {frame_count}", 
                (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
            
            # JPEGエンコードしてストリーミング
            ret, buffer = cv2.imencode('.jpg', display_img)
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # フレームレート制御（ブラウザの負荷を考慮して15FPS程度に）
            process_time = time.time() - start_time
            if process_time < 0.066:  # 約15FPS
                time.sleep(0.066 - process_time)
    except Exception as e:
        print(f"[エラー] フレーム生成中にエラーが発生しました: {e}")
        traceback.print_exc()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>深度画像キャプチャツール</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    text-align: center;
                    background-color: #f0f0f0;
                }
                h1 {
                    color: #333;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .video-container {
                    margin: 20px 0;
                }
                .controls {
                    margin: 20px 0;
                    padding: 10px;
                    background-color: #fff;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                button {
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 10px 20px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border-radius: 4px;
                }
                button:hover {
                    background-color: #45a049;
                }
                select {
                    padding: 8px 12px;
                    margin: 8px 0;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    box-sizing: border-box;
                }
                .status {
                    font-weight: bold;
                    margin: 10px 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>深度画像キャプチャツール</h1>
                <div class="video-container">
                    <img src="/video_feed" width="100%" />
                </div>
                <div class="controls">
                    <button onclick="captureImage()">画像キャプチャ</button>
                    <div>
                        <label for="pattern">深度パターン:</label>
                        <select id="pattern" onchange="changePattern()">
                            <option value="objects">物体</option>
                            <option value="gradient">グラデーション</option>
                            <option value="random">ランダム</option>
                            <option value="grayscale">グレースケール</option>
                        </select>
                    </div>
                    <div id="status" class="status"></div>
                </div>
            </div>
            
            <script>
                // 画像キャプチャ関数
                function captureImage() {
                    document.getElementById('status').textContent = '画像キャプチャ中...';
                    fetch('/capture')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('status').textContent = data.message;
                            setTimeout(() => {
                                document.getElementById('status').textContent = '';
                            }, 3000);
                        })
                        .catch(error => {
                            document.getElementById('status').textContent = 'エラー: ' + error;
                        });
                }
                
                // パターン変更関数
                function changePattern() {
                    const pattern = document.getElementById('pattern').value;
                    document.getElementById('status').textContent = 'パターン変更中...';
                    fetch('/change_pattern?pattern=' + pattern)
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('status').textContent = data.message;
                            setTimeout(() => {
                                document.getElementById('status').textContent = '';
                            }, 3000);
                        })
                        .catch(error => {
                            document.getElementById('status').textContent = 'エラー: ' + error;
                        });
                }
            </script>
        </body>
    </html>
    """

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/capture")
async def capture():
    capture_event.set()
    return {"message": f"画像をキャプチャしました。保存先: {output_dir}"}

@app.get("/change_pattern")
async def change_pattern(pattern: str):
    global depth_pattern
    
    if pattern in ["objects", "gradient", "random", "grayscale"]:
        depth_pattern = pattern
        return {"message": f"深度パターンを {pattern} に変更しました"}
    else:
        return {"message": "無効なパターンです"}

def depth_capture_main():
    """メインの深度キャプチャ関数"""
    global output_dir, depth_pattern
    
    parser = argparse.ArgumentParser(description='深度画像キャプチャツール')
    parser.add_argument('--camera', type=int, default=0, help='使用するカメラのインデックス')
    parser.add_argument('--width', type=int, default=640, help='キャプチャ幅')
    parser.add_argument('--height', type=int, default=480, help='キャプチャ高さ')
    parser.add_argument('--output', type=str, default='depth_captures', help='出力ディレクトリ')
    parser.add_argument('--pattern', type=str, default='objects', choices=['random', 'gradient', 'objects', 'grayscale'],
                        help='合成深度パターン (random, gradient, objects, grayscale)')
    parser.add_argument('--port', type=int, default=8080, help='Webサーバーポート')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Webサーバーホスト')
    
    args = parser.parse_args()
    
    # グローバル変数の設定
    output_dir = args.output
    depth_pattern = args.pattern
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # Linuxシステムリソースの最適化（send_uvc_streaming_depthから参考）
    try:
        # CPU優先度を最大に設定
        os.system(f"sudo renice -n -20 -p {os.getpid()}")
        os.system(f"sudo ionice -c 1 -n 0 -p {os.getpid()}")
        print("[情報] システムリソースを最適化しました")
    except Exception as e:
        print(f"[警告] システムリソース最適化に失敗しました: {e}")
    
    print("\n===== 深度画像キャプチャツール =====")
    print(f"カメラインデックス: {args.camera}")
    print(f"解像度: {args.width}x{args.height}")
    print(f"出力ディレクトリ: {args.output}")
    print(f"合成深度パターン: {args.pattern}")
    print(f"Webサーバー: http://{args.host}:{args.port}")
    print("==============================")
    print("[情報] Webブラウザでアクセスして使用してください")
    print("==============================\n")
    
    # カメラスレッドの開始
    camera_thread = threading.Thread(
        target=camera_thread_function,
        args=(args.camera, args.width, args.height, args.pattern),
        daemon=True
    )
    camera_thread.start()
    
    # FastAPIサーバーの起動
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\n[情報] ユーザーによる中断")
    finally:
        # 終了処理
        stop_event.set()
        camera_thread.join(timeout=2.0)
        gc.collect()
        print("[情報] 終了しました")

if __name__ == "__main__":
    depth_capture_main()