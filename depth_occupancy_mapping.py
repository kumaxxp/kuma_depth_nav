import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import time
import threading
import queue
import sys

# 設定パラメータ
GRID_RESOLUTION = 0.05  # メートル単位でのグリッドサイズ
GRID_WIDTH = 100  # グリッドの幅（セル数）
GRID_HEIGHT = 100  # グリッドの高さ（セル数）
HEIGHT_THRESHOLD = 0.3  # 高さの閾値（メートル）- この値より低いものは通行可能とする
MODEL_PATH = '/opt/m5stack/data/depth_anything/compiled.axmodel'  # モデルパス

# グローバル変数として最新のカメラフレームを保持
latest_frame = None
frame_lock = threading.Lock()

def initialize_model(model_path: str):
    """Depth Anythingモデルを初期化する"""
    try:
        print(f"[INFO] モデルを読み込み中: {model_path}")
        if not os.path.exists(model_path):
            print(f"[エラー] モデルファイルが見つかりません: {model_path}")
            return None, None

        # axengineのインポートをここで行う（エラー処理のため）
        try:
            import axengine as axe
            print("[INFO] axengineをインポートしました")
        except ImportError as e:
            print(f"[エラー] axengineのインポートに失敗しました: {e}")
            return None, None
            
        # axengineのバージョンに応じて適切な初期化方法を試す
        session = None
        try:
            # 最もシンプルな初期化方法を試す
            print("[INFO] 基本的なモデル初期化を試みます")
            session = axe.InferenceSession(model_path)
        except Exception as e:
            print(f"[警告] シンプルな初期化に失敗: {e}")
            try:
                # オプションを使用した初期化（新しいバージョン向け）
                print("[INFO] オプション付き初期化を試みます")
                options = {
                    "axe.input_layout": "NHWC",
                    "axe.output_layout": "NHWC"
                }
                session = axe.InferenceSession(model_path, options)
            except Exception as e:
                print(f"[エラー] モデル初期化に失敗しました: {e}")
                return None, None
            
        # モデル情報を表示
        try:
            inputs = session.get_inputs()
            if not inputs or len(inputs) == 0:
                print("[エラー] モデルには入力がありません")
                return None, None
                
            input_info = inputs[0]
            input_name = input_info.name
            print(f"[INFO] モデル入力名: {input_name}")
            print(f"[INFO] モデル入力形状: {input_info.shape}")
            print(f"[INFO] モデル入力データ型: {input_info.dtype}")
            
            outputs = session.get_outputs()
            if outputs and len(outputs) > 0:
                print(f"[INFO] モデル出力名: {outputs[0].name}")
                print(f"[INFO] モデル出力形状: {outputs[0].shape}")
            
            return session, input_name
        except Exception as e:
            print(f"[エラー] モデル情報の取得に失敗しました: {e}")
            return None, None
    except Exception as e:
        print(f"[エラー] モデル初期化プロセスに失敗しました: {e}")
        return None, None

def initialize_camera(index=0, width=320, height=240):
    """カメラを初期化する - より安定した設定を使用"""
    try:
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

def camera_thread_function(camera, stop_event):
    """カメラからフレームを連続的に取得するスレッド関数"""
    global latest_frame
    
    print("[INFO] カメラスレッドを開始しました")
    
    try:
        while not stop_event.is_set():
            success, frame = camera_capture_frame(camera)
            if success and frame is not None:
                # 最新フレームを更新（スレッドセーフに）
                with frame_lock:
                    latest_frame = frame.copy()
            else:
                time.sleep(0.01)
                continue
                
            # カメラのフレームレート制御のための短い待機
            time.sleep(0.05)  # 20FPS程度
    except Exception as e:
        print(f"[エラー] カメラスレッド内でエラーが発生しました: {e}")
    finally:
        print("[INFO] カメラスレッドを終了します")
        if camera is not None:
            camera.release()

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
        height, width = depth_map.shape
        
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
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # 無効な点（深度値が0に近い）を除外
        valid_points = points[points[:, 2] > 0.1]
        
        return valid_points
    except Exception as e:
        print(f"[エラー] 点群変換中にエラーが発生しました: {e}")
        return np.zeros((0, 3))  # 空の点群を返す

def create_top_down_occupancy_grid(points, grid_resolution=GRID_RESOLUTION, 
                                  grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT, 
                                  height_threshold=HEIGHT_THRESHOLD):
    """3D点群から天頂からの占有グリッドを生成する
    
    Args:
        points: 3D点群 (N, 3)
        grid_resolution: グリッドの解像度（メートル）
        grid_width, grid_height: グリッドサイズ
        height_threshold: 通行可能と判断する高さの閾値
    
    Returns:
        occupancy_grid: 占有グリッド（0: 不明、1: 占有、2: 通行可能）
    """
    # グリッドを初期化 (0: 不明、1: 占有、2: 通行可能)
    occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
    
    # カメラ位置をグリッドの中心下部に設定
    camera_x = grid_width // 2
    camera_y = grid_height - 10  # 下部に少し余裕を持たせる
    
    # カメラ位置（原点）からの相対位置を計算
    # X: 左右方向、Y: 高さ方向、Z: 奥行き方向
    # 2Dグリッド上での位置に変換：
    # グリッドX = カメラX - 点のX / 解像度
    # グリッドY = カメラY - 点のZ / 解像度
    
    if len(points) == 0:
        return occupancy_grid
    
    for point in points:
        x, y, z = point
        
        # XZ平面上のグリッド座標に変換
        grid_x = int(camera_x - x / grid_resolution)
        grid_y = int(camera_y - z / grid_resolution)
        
        # グリッド範囲内かチェック
        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            # 高さが閾値以下なら通行可能
            if abs(y) < height_threshold:
                occupancy_grid[grid_y, grid_x] = 2  # 通行可能
            else:
                occupancy_grid[grid_y, grid_x] = 1  # 占有
    
    return occupancy_grid

def visualize_occupancy_grid(occupancy_grid, frame=None, depth_vis=None):
    """占有グリッドを可視化する"""
    try:
        # 表示用の画像を作成
        grid_vis = np.zeros((occupancy_grid.shape[0], occupancy_grid.shape[1], 3), dtype=np.uint8)
        grid_vis[occupancy_grid == 0] = [211, 211, 211]  # 不明: ライトグレー
        grid_vis[occupancy_grid == 1] = [0, 0, 139]      # 占有: ダークレッド
        grid_vis[occupancy_grid == 2] = [0, 128, 0]      # 通行可能: 緑
        
        # カメラ位置をマーク
        camera_x = occupancy_grid.shape[1] // 2
        camera_y = occupancy_grid.shape[0] - 10
        cv2.circle(grid_vis, (camera_x, camera_y), 5, (255, 0, 0), -1)
        
        # フルビジュアライゼーションが要求された場合のみmatplotlibを使用
        if frame is not None and depth_vis is not None:
            try:
                plt.figure(figsize=(12, 10))
                
                plt.subplot(2, 2, 1)
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.title('カメラ画像')
                plt.axis('off')
                
                plt.subplot(2, 2, 2)
                plt.imshow(depth_vis)
                plt.title('深度マップ')
                plt.axis('off')
                
                plt.subplot(2, 2, 3)
                
                # 占有グリッドの色を設定
                cmap = plt.cm.colors.ListedColormap(['lightgray', 'darkred', 'green'])
                bounds = [0, 1, 2, 3]
                norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
                
                plt.imshow(occupancy_grid, cmap=cmap, norm=norm)
                plt.colorbar(ticks=[0.5, 1.5, 2.5], 
                             label='セルの状態', 
                             boundaries=bounds,
                             values=[0, 1, 2])
                plt.clim(0, 3)
                
                # カメラ位置をマーク
                plt.scatter(camera_x, camera_y, c='blue', marker='^', s=100, label='カメラ位置')
                
                plt.title('天頂からの占有グリッド')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig('occupancy_grid.png')
                plt.close()
            except Exception as e:
                print(f"[警告] matplotlibによる可視化に失敗しました: {e}")
                # エラーが発生しても継続
        
        return grid_vis
        
    except Exception as e:
        print(f"[エラー] グリッドの可視化に失敗しました: {e}")
        # エラーが発生した場合、空のグリッドを返す
        return np.zeros((100, 100, 3), dtype=np.uint8)

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
        
        depth_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
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
        
        return depth_resized
    except Exception as e:
        print(f"[エラー] 深度マップの可視化に失敗しました: {e}")
        # エラーが発生した場合、元のフレームを返す
        return original_frame.copy()

# テスト用の合成画像と深度マップを生成する関数
def create_synthetic_data(width=384, height=256):
    """テスト用の合成RGB画像と深度マップを生成する
    
    Returns:
        frame: RGB画像
        depth_map: 合成深度マップ
    """
    print("[情報] テスト用の合成画像と深度マップを生成します")
    
    # RGB画像を生成（単純な背景に四角形と円を描画）
    frame = np.ones((height, width, 3), dtype=np.uint8) * 200  # グレーの背景
    
    # いくつかのオブジェクト（障害物）を描画
    cv2.rectangle(frame, (50, 50), (120, 120), (0, 0, 255), -1)  # 赤い四角形
    cv2.rectangle(frame, (200, 70), (280, 140), (255, 0, 0), -1)  # 青い四角形
    cv2.circle(frame, (300, 180), 40, (0, 255, 0), -1)  # 緑の円
    
    # 通路を描画（床面と見なす領域）
    cv2.rectangle(frame, (140, 0), (220, height-1), (200, 200, 150), -1)  # 通路
    
    # 深度マップを生成（単純な値を設定）
    depth_map = np.ones((height, width), dtype=np.float32) * 5.0  # 背景は遠い（5m）
    
    # オブジェクトの深度を設定
    depth_mask = np.zeros((height, width), dtype=np.float32)
    cv2.rectangle(depth_mask, (50, 50), (120, 120), 1.0, -1)  # 1mの距離
    cv2.rectangle(depth_mask, (200, 70), (280, 140), 1.5, -1)  # 1.5mの距離
    cv2.circle(depth_mask, (300, 180), 40, 2.0, -1)  # 2mの距離
    
    # 通路の深度を設定
    cv2.rectangle(depth_mask, (140, 0), (220, height-1), 3.0, -1)  # 3mの距離
    
    # 深度マップを更新（障害物がある場所は近いので値が小さく、通路は中間の値）
    depth_map = np.where(depth_mask > 0, depth_mask, depth_map)
    
    # ノイズを追加してリアルさを向上
    noise = np.random.normal(0, 0.05, depth_map.shape)
    depth_map += noise
    depth_map = np.clip(depth_map, 0.1, 10.0)  # 値の範囲をクリップ
    
    return frame, depth_map

def run_dummy_model(input_tensor):
    """ダミーの深度推論を実行（axengineなしで動作確認できるようにする）"""
    print("[情報] ダミーの深度推論を実行します")
    # 入力テンソルの形状を取得（NHWC形式を想定）
    n, h, w, c = input_tensor.shape
    
    # 深度マップのサイズを計算
    depth_h, depth_w = h, w
    
    # 単純なダミー深度マップを生成
    depth_map = np.ones((depth_h, depth_w), dtype=np.float32) * 5.0  # 背景は5m
    
    # 中央に円形の物体を配置
    center_x, center_y = depth_w // 2, depth_h // 2
    radius = min(depth_w, depth_h) // 4
    
    y, x = np.ogrid[:depth_h, :depth_w]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = dist_from_center <= radius
    depth_map[mask] = 2.0  # 中央の物体は2m
    
    # 左側に四角形の物体
    x_start = depth_w // 4 - radius // 2
    x_end = depth_w // 4 + radius // 2
    y_start = depth_h // 2 - radius // 2
    y_end = depth_h // 2 + radius // 2
    depth_map[y_start:y_end, x_start:x_end] = 1.0  # 左の物体は1m
    
    # 右側に四角形の物体
    x_start = 3 * depth_w // 4 - radius // 2
    x_end = 3 * depth_w // 4 + radius // 2
    y_start = depth_h // 2 - radius // 2
    y_end = depth_h // 2 + radius // 2
    depth_map[y_start:y_end, x_start:x_end] = 1.5  # 右の物体は1.5m
    
    # ノイズを追加
    noise = np.random.normal(0, 0.05, depth_map.shape)
    depth_map += noise
    depth_mapは np.clip(depth_map, 0.1, 10.0)  # 値の範囲をクリップ
    
    # モデル出力の形式に合わせる
    # 不明な場合は単一の深度マップを返す
    return [depth_map]

def process_depth_image_with_camera():
    """カメラ入力を使って深度処理とOccupancy Grid Mappingを行う（改良版）"""
    global latest_frame
    
    # axengineが利用可能かチェック
    model_available = True
    try:
        import axengine
        session, input_name = initialize_model(MODEL_PATH)
        if session is None:
            print("[警告] モデル初期化に失敗しました。ダミーモデルを使用します。")
            model_available = False
    except ImportError:
        print("[警告] axengineがインポートできません。ダミーモデルを使用します。")
        session, input_name = None, None
        model_available = False
    
    # カメラを初期化
    camera = initialize_camera()
    if camera is None:
        print("[エラー] カメラを開けません。--synthetic オプションを使用してテスト用の合成データを試してください。")
        return
    
    # カメラスレッドを開始
    stop_event = threading.Event()
    camera_thread = threading.Thread(target=camera_thread_function, args=(camera, stop_event))
    camera_thread.daemon = True
    camera_thread.start()
    
    print("[INFO] カメラスレッドを起動しました。'q'キーで終了します。")
    
    try:
        wait_frames = 0
        while True:
            # フレームを待機
            frame = None
            with frame_lock:
                if latest_frame is not None:
                    frame = latest_frame.copy()
            
            if frame is None:
                wait_frames += 1
                if wait_frames % 10 == 0:
                    print(f"[警告] カメラフレーム待機中... ({wait_frames})")
                time.sleep(0.1)
                continue
                
            wait_frames = 0
            
            try:
                # フレームを処理
                input_tensor = process_frame(frame)
                
                # 深度推定を実行
                if model_available and session is not None:
                    try:
                        depth_output = session.run(None, {input_name: input_tensor})[0]
                    except Exception as e:
                        print(f"[エラー] モデル推論中にエラーが発生しました: {e}")
                        depth_output = run_dummy_model(input_tensor)[0]
                else:
                    depth_output = run_dummy_model(input_tensor)[0]
                    
                depth_map = depth_output.squeeze()
                
                # 深度データの可視化
                depth_vis = create_depth_visualization(depth_map, frame)
                
                # 深度を点群に変換
                points = depth_to_point_cloud(depth_map)
                
                # 占有グリッドを作成
                occupancy_grid = create_top_down_occupancy_grid(points)
                
                # グリッドを可視化
                grid_vis = visualize_occupancy_grid(occupancy_grid)
                
                # 結果を表示
                cv2.imshow("カメラ", frame)
                cv2.imshow("深度マップ", depth_vis)
                cv2.imshow("占有グリッド", grid_vis)
                
            except Exception as e:
                print(f"[エラー] フレーム処理中にエラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
            
            # キー入力をチェック
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 現在のフレームを保存
                timestamp = int(time.time())
                cv2.imwrite(f"camera_frame_{timestamp}.jpg", frame)
                cv2.imwrite(f"depth_map_{timestamp}.jpg", depth_vis)
                cv2.imwrite(f"occupancy_grid_{timestamp}.jpg", grid_vis)
                print(f"[情報] フレームを保存しました: camera_frame_{timestamp}.jpg")
                
    finally:
        # リソースを解放
        stop_event.set()
        camera_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        print("[INFO] 処理を終了しました")

def process_depth_image(img_path=None, use_camera=False, use_synthetic=False):
    """深度画像を処理してOccupancy Gridを作成する"""
    
    if use_synthetic:
        # 合成データを生成
        frame, depth_map = create_synthetic_data()
        
        # 深度データの可視化
        depth_vis = create_depth_visualization(depth_map, frame)
        
        # 深度を点群に変換
        points = depth_to_point_cloud(depth_map)
        
        # 占有グリッドを作成
        occupancy_grid = create_top_down_occupancy_grid(points)
        
        # グリッドを可視化して保存
        grid_vis = visualize_occupancy_grid(occupancy_grid, frame, depth_vis)
        
        print(f"[情報] 結果を'occupancy_grid.png'に保存しました")
        
        # 結果を表示
        cv2.imshow("合成入力画像", frame)
        cv2.imshow("深度マップ", depth_vis)
        cv2.imshow("占有グリッド", grid_vis)
        print("[情報] 任意のキーを押すと終了します。's'キーを押すと現在のフレームを保存します。")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s'):
                # 現在の画像を保存
                timestamp = int(time.time())
                cv2.imwrite(f"synthetic_frame_{timestamp}.jpg", frame)
                cv2.imwrite(f"synthetic_depth_{timestamp}.jpg", depth_vis)
                cv2.imwrite(f"synthetic_grid_{timestamp}.jpg", grid_vis)
                print(f"[情報] 画像を保存しました: synthetic_frame_{timestamp}.jpg")
            else:
                break
                
        cv2.destroyAllWindows()
        return
    
    if use_camera:
        # 改良版のカメラ処理関数を呼び出す
        process_depth_image_with_camera()
        return
    
    elif img_path:
        if not os.path.exists(img_path):
            print(f"[エラー] 画像ファイルが見つかりません: {img_path}")
            return
        
        try:
            # モデルを初期化
            session, input_name = initialize_model(MODEL_PATH)
            
            # 画像ファイルを読み込み
            frame = cv2.imread(img_path)
            
            # フレームを処理
            input_tensor = process_frame(frame)
            
            # 深度推定を実行
            depth_output = session.run(None, {input_name: input_tensor})[0]
            depth_map = depth_output.squeeze()
            
            # 深度データの可視化
            depth_vis = create_depth_visualization(depth_map, frame)
            
            # 深度を点群に変換
            points = depth_to_point_cloud(depth_map)
            
            # 占有グリッドを作成
            occupancy_grid = create_top_down_occupancy_grid(points)
            
            # グリッドを可視化して保存
            visualize_occupancy_grid(occupancy_grid, frame, depth_vis)
            
            print(f"[情報] 結果を'occupancy_grid.png'に保存しました")
            
            # 結果を表示
            cv2.imshow("カメラ", frame)
            cv2.imshow("深度マップ", depth_vis)
            grid_vis = visualize_occupancy_grid(occupancy_grid)
            cv2.imshow("占有グリッド", grid_vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"[エラー] 画像処理中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("[エラー] 画像パスを指定するか、カメラを使用するか、--synthetic オプションを使用してください")

def check_dependencies():
    """必要な依存関係をチェックする"""
    print("\n===== 依存関係チェック =====")
    
    # OpenCVをチェック
    try:
        cv_version = cv2.__version__
        print(f"[✓] OpenCV: {cv_version}")
    except:
        print("[✗] OpenCV: インストールされていないか、インポートできません")
        
    # NumPyをチェック
    try:
        np_version = np.__version__
        print(f"[✓] NumPy: {np_version}")
    except:
        print("[✗] NumPy: インストールされていないか、インポートできません")
        
    # Matplotlibをチェック
    try:
        import matplotlib
        mpl_version = matplotlib.__version__
        print(f"[✓] Matplotlib: {mpl_version}")
    except ImportError:
        print("[✗] Matplotlib: インストールされていないか、インポートできません")
        
    # axengineをチェック
    try:
        import axengine
        print(f"[✓] axengine: インポートに成功しました")
        try:
            # バージョン情報の取得を試みる
            version = getattr(axengine, "__version__", "不明")
            print(f"[✓] axengine バージョン: {version}")
        except:
            print(f"[✓] axengine: バージョン情報は取得できませんでした")
    except ImportError:
        print("[✗] axengine: インストールされていないか、インポートできません")
        print("    ダミーの深度モデルを使用して実行できます")
    
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='深度データからOccupancy Gridを生成します')
    parser.add_argument('--image', type=str, help='入力画像のパス')
    parser.add_argument('--camera', action='store_true', help='カメラを使用する')
    parser.add_argument('--synthetic', action='store_true', help='テスト用の合成データを使用する')
    parser.add_argument('--check', action='store_true', help='依存関係をチェックする')
    
    args = parser.parse_args()
    
    if args.check:
        check_dependencies()
        sys.exit(0)
        
    if args.synthetic:
        process_depth_image(use_synthetic=True)
    elif args.camera:
        process_depth_image(use_camera=True)
    elif args.image:
        process_depth_image(img_path=args.image)
    else:
        print("[エラー] --image、--camera、--synthetic、または --check オプションを指定してください")
        parser.print_help()