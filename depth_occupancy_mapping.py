import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import axengine as axe
import time

# 設定パラメータ
GRID_RESOLUTION = 0.05  # メートル単位でのグリッドサイズ
GRID_WIDTH = 100  # グリッドの幅（セル数）
GRID_HEIGHT = 100  # グリッドの高さ（セル数）
HEIGHT_THRESHOLD = 0.3  # 高さの閾値（メートル）- この値より低いものは通行可能とする
MODEL_PATH = '/opt/m5stack/data/depth_anything/compiled.axmodel'  # モデルパス

def initialize_model(model_path: str):
    """Depth Anythingモデルを初期化する"""
    session = axe.InferenceSession(model_path)
    input_info = session.get_inputs()[0]
    print("[INFO] モデル入力形状:", input_info.shape)
    print("[INFO] モデル入力データ型:", input_info.dtype)
    return session, input_info.name

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

def process_depth_image(img_path=None, use_camera=False):
    """深度画像を処理してOccupancy Gridを作成する"""
    
    session, input_name = initialize_model(MODEL_PATH)
    
    if use_camera:
        # カメラからの入力を処理
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[エラー] カメラが開けません")
            return
        
        print("[情報] 'q'キーで終了します")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[警告] フレームを取得できませんでした")
                    continue
                
                # フレームを処理
                input_tensor = process_frame(frame)
                
                # 深度推定を実行
                try:
                    depth_output = session.run(None, {input_name: input_tensor})[0]
                except Exception as e:
                    print(f"[エラー] モデル実行中にエラーが発生しました: {e}")
                    continue
                
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
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    elif img_path:
        if not os.path.exists(img_path):
            print(f"[エラー] 画像ファイルが見つかりません: {img_path}")
            return
        
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
    
    else:
        print("[エラー] 画像パスを指定するか、カメラを使用してください")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='深度データからOccupancy Gridを生成します')
    parser.add_argument('--image', type=str, help='入力画像のパス')
    parser.add_argument('--camera', action='store_true', help='カメラを使用する')
    
    args = parser.parse_args()
    
    if args.camera:
        process_depth_image(use_camera=True)
    elif args.image:
        process_depth_image(img_path=args.image)
    else:
        print("[エラー] --imageまたは--cameraオプションを指定してください")
        parser.print_help()