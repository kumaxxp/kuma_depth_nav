import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import axengine as axe
import time
import threading
import queue

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
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
            
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
            print("[INFO] 基本的なモデル初期化に切り替えます")
            session = axe.InferenceSession(model_path)
            
        # モデル情報を表示
        input_info = session.get_inputs()[0]
        print("[INFO] モデル入力形状:", input_info.shape)
        print("[INFO] モデル入力データ型:", input_info.dtype)
        return session, input_info.name
    except Exception as e:
        print(f"[エラー] モデル初期化に失敗しました: {e}")
        raise

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
    
    if MODEL_PATH.endswith(".axmodel"):
        tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)  # NHWC uint8 for axmodel
        if tensor.nbytes % np.dtype(np.uint8).itemsize != 0:
            raise ValueError("[エラー] テンソルバッファサイズがデータ型と一致しません")
        return tensor
    
    rgb = rgb.astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0)

def depth_to_point_cloud(depth_map, fx=500, fy=500, cx=192, cy=128):
    """深度マップから点群を生成する
    
    Args:
        depth_map: 深度マップ (H, W)
        fx, fy: 焦点距離
        cx, cy: 画像中心
    
    Returns:
        points: 3D点群 (N, 3)
    """
    height, width = depth_map.shape
    
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
    plt.figure(figsize=(12, 10))
    
    # 複数のプロットを配置
    if frame is not None and depth_vis is not None:
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
    camera_x = occupancy_grid.shape[1] // 2
    camera_y = occupancy_grid.shape[0] - 10
    plt.scatter(camera_x, camera_y, c='blue', marker='^', s=100, label='カメラ位置')
    
    plt.title('天頂からの占有グリッド')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('occupancy_grid.png')
    plt.close()
    
    # 表示用の画像を作成
    grid_vis = np.zeros((occupancy_grid.shape[0], occupancy_grid.shape[1], 3), dtype=np.uint8)
    grid_vis[occupancy_grid == 0] = [211, 211, 211]  # 不明: ライトグレー
    grid_vis[occupancy_grid == 1] = [0, 0, 139]      # 占有: ダークレッド
    grid_vis[occupancy_grid == 2] = [0, 128, 0]      # 通行可能: 緑
    
    # カメラ位置をマーク
    cv2.circle(grid_vis, (camera_x, camera_y), 5, (255, 0, 0), -1)
    
    return grid_vis

def create_depth_visualization(depth_map: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
    """深度マップのカラー表示を作成する"""
    depth_feature = depth_map.reshape(depth_map.shape[-2:])
    normalized = (depth_feature - depth_feature.min()) / (depth_feature.max() - depth_feature.min() + 1e-6)
    depth_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    depth_resized = cv2.resize(depth_colored, (original_frame.shape[1], original_frame.shape[0]))
    return depth_resized

# テスト用の合成画像と深度マップを生成する関数を追加
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

def process_depth_image_with_camera():
    """カメラ入力を使って深度処理とOccupancy Grid Mappingを行う（改良版）"""
    global latest_frame
    
    # 深度推論モデルを初期化
    try:
        session, input_name = initialize_model(MODEL_PATH)
    except Exception as e:
        print(f"[エラー] モデル初期化に失敗しました: {e}")
        return
    
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
                depth_output = session.run(None, {input_name: input_tensor})[0]
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
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
        visualize_occupancy_grid(occupancy_grid, frame, depth_vis)
        
        print(f"[情報] 結果を'occupancy_grid.png'に保存しました")
        
        # 結果を表示
        cv2.imshow("合成入力画像", frame)
        cv2.imshow("深度マップ", depth_vis)
        grid_vis = visualize_occupancy_grid(occupancy_grid)
        cv2.imshow("占有グリッド", grid_vis)
        print("[情報] 任意のキーを押すと終了します")
        cv2.waitKey(0)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='深度データからOccupancy Gridを生成します')
    parser.add_argument('--image', type=str, help='入力画像のパス')
    parser.add_argument('--camera', action='store_true', help='カメラを使用する')
    parser.add_argument('--synthetic', action='store_true', help='テスト用の合成データを使用する')
    
    args = parser.parse_args()
    
    if args.synthetic:
        process_depth_image(use_synthetic=True)
    elif args.camera:
        process_depth_image(use_camera=True)
    elif args.image:
        process_depth_image(img_path=args.image)
    else:
        print("[エラー] --image、--camera、または --synthetic オプションを指定してください")
        parser.print_help()