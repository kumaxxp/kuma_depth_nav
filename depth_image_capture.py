#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度画像キャプチャツール

depth_occupancy_mapping.pyで使用するための深度画像をキャプチャするためのツール。
カメラからの深度データをキャプチャして保存します。
Linux環境に対応。
"""

import numpy as np
import cv2
import argparse
import os
import time
import sys
import gc  # ガベージコレクションのための追加
import traceback

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

def create_depth_visualization(depth_map: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
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
        
        # 境界線用のグラデーションバーを追加
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
    
    # 深度マップの可視化を保存
    depth_vis = create_depth_visualization(depth_map, frame)
    cv2.imwrite(depth_vis_filename, depth_vis)
    
    # 生の深度データをnumpy形式で保存（後処理用）
    np.save(depth_raw_filename, depth_map)
    
    print(f"[情報] データを保存しました: {rgb_filename}, {depth_vis_filename}, {depth_raw_filename}")
    return rgb_filename, depth_vis_filename, depth_raw_filename

def depth_capture_main():
    """メインの深度キャプチャ関数"""
    parser = argparse.ArgumentParser(description='深度画像キャプチャツール')
    parser.add_argument('--camera', type=int, default=0, help='使用するカメラのインデックス')
    parser.add_argument('--width', type=int, default=640, help='キャプチャ幅')
    parser.add_argument('--height', type=int, default=480, help='キャプチャ高さ')
    parser.add_argument('--output', type=str, default='depth_captures', help='出力ディレクトリ')
    parser.add_argument('--pattern', type=str, default='objects', choices=['random', 'gradient', 'objects', 'grayscale'],
                        help='合成深度パターン (random, gradient, objects, grayscale)')
    parser.add_argument('--continuous', action='store_true', help='連続キャプチャモード')
    parser.add_argument('--interval', type=float, default=1.0, help='連続キャプチャの間隔（秒）')
    parser.add_argument('--display-width', type=int, default=1280, help='表示画面の幅')
    
    args = parser.parse_args()
    
    # カメラを初期化
    camera = initialize_camera(args.camera, args.width, args.height)
    if camera is None:
        print("[エラー] カメラを初期化できません")
        return
    
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
    print("==============================")
    print("[情報] 's'キーで深度画像をキャプチャ、'q'キーで終了")
    if args.continuous:
        print(f"[情報] 連続キャプチャモード有効 (間隔: {args.interval}秒)")
    print("==============================\n")
    
    last_capture_time = 0
    frame_count = 0
    
    try:
        while True:
            start_time = time.time()
            
            success, frame = camera_capture_frame(camera)
            if not success:
                print("[警告] フレームの取得に失敗しました")
                time.sleep(0.1)
                continue
            
            # フレームからの合成深度マップを作成
            depth_map = create_synthetic_depth(frame, args.pattern)
            
            # 深度マップの可視化
            depth_vis = create_depth_visualization(depth_map, frame)
            
            # 表示用の画像を作成
            display_img = np.hstack([frame, depth_vis])
            
            # 表示サイズが大きすぎる場合は縮小
            if display_img.shape[1] > args.display_width:
                scale = args.display_width / display_img.shape[1]
                display_img = cv2.resize(display_img, None, fx=scale, fy=scale)
            
            # フレーム番号を追加
            frame_count += 1
            cv2.putText(
                display_img, 
                f"Frame: {frame_count}", 
                (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
            
            # 表示
            cv2.imshow("カメラ & 深度マップ", display_img)
            
            # キー入力をチェック
            key = cv2.waitKey(1) & 0xFF
            
            # 連続キャプチャモード
            current_time = time.time()
            if args.continuous and (current_time - last_capture_time) >= args.interval:
                timestamp = int(current_time * 1000)  # ミリ秒単位のタイムスタンプ
                save_depth_data(frame, depth_map, timestamp, args.output)
                last_capture_time = current_time
            
            # キー操作
            if key == ord('s'):
                # 手動キャプチャ
                timestamp = int(time.time() * 1000)
                save_depth_data(frame, depth_map, timestamp, args.output)
            elif key == ord('q'):
                break
            
            # フレームレート制御
            process_time = time.time() - start_time
            if process_time < 0.033:  # 目標30FPS
                time.sleep(0.033 - process_time)
                
    except KeyboardInterrupt:
        print("\n[情報] ユーザーによる中断")
    except Exception as e:
        print(f"[エラー] 予期しないエラーが発生しました: {e}")
        traceback.print_exc()
    finally:
        # リソースを解放
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        gc.collect()
        print("[情報] 終了しました")

if __name__ == "__main__":
    depth_capture_main()