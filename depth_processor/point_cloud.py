"""
深度マップから点群を生成し、トップビュー（天頂視点）表示を行う機能を提供します。
"""

import numpy as np
import cv2

# デフォルトパラメータ
GRID_RESOLUTION = 0.06  # メートル/セル
GRID_WIDTH = 100        # 横方向のセル数
GRID_HEIGHT = 100       # 縦方向のセル数
HEIGHT_THRESHOLD = 0.3  # 通行可能と判定する高さの閾値（メートル）
MAX_DEPTH = 6.0         # 最大深度（メートル）

def depth_to_point_cloud(depth_map, fx=500, fy=500, cx=None, cy=None):
    """
    深度マップから3D点群を生成します。
    
    Args:
        depth_map (numpy.ndarray): 深度マップ
        fx, fy (float): カメラの焦点距離
        cx, cy (float): 画像の中心座標。Noneの場合は画像の中心が使用されます。
    
    Returns:
        numpy.ndarray: 形状 (N, 3) の点群データ
    """
    # 深度マップの形状を取得
    if len(depth_map.shape) > 2:
        depth_map = depth_map.squeeze()  # バッチ次元や不要な次元を除去
    
    height, width = depth_map.shape
    
    # 画像中心を設定
    if cx is None:
        cx = width / 2
    if cy is None:
        cy = height / 2
    
    # 各ピクセルに対して座標を計算
    x_indices, y_indices = np.meshgrid(
        np.arange(width), np.arange(height)
    )
    
    # ピクセル座標をカメラ座標系に変換
    normalized_x = (x_indices - cx) / fx
    normalized_y = (y_indices - cy) / fy
    
    # 3D点座標を計算 (x, y, z)
    # カメラ座標系: Z が前方、X が右、Y が下方向
    x = normalized_x * depth_map
    y = normalized_y * depth_map
    z = depth_map
    
    # 無効な深度値（ゼロや非常に大きい値）をマスク
    valid_mask = (z > 0.1) & (z < MAX_DEPTH)
    
    # 点群データを形成
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    z_valid = z[valid_mask]
    
    # (N, 3) 形式の点群データを返す
    points = np.vstack((x_valid, y_valid, z_valid)).T
    
    return points

def create_top_down_occupancy_grid(points, grid_resolution=GRID_RESOLUTION, 
                                  grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT, 
                                  height_threshold=HEIGHT_THRESHOLD):
    """
    3D点群から天頂視点の占有グリッドを生成します。
    
    Args:
        points (numpy.ndarray): 形状 (N, 3) の3D点群データ
        grid_resolution (float): グリッドの解像度（メートル/セル）
        grid_width (int): グリッドの幅（セル数）
        grid_height (int): グリッドの高さ（セル数）
        height_threshold (float): 通行可能と判定する高さの閾値（メートル）
    
    Returns:
        numpy.ndarray: 形状 (grid_height, grid_width) の占有グリッド
            0: 不明（データなし）
            1: 占有（障害物）
            2: 通行可能
    """
    # 初期化: すべてのセルを「不明」に設定
    grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
    
    # グリッドの中心
    grid_center_x = grid_width // 2
    grid_center_y = grid_height - 10  # カメラの少し前を中心にする
    
    if points.shape[0] == 0:
        return grid  # 点がない場合は空のグリッドを返す
    
    # 点群データをグリッド座標に変換
    # X軸（左右）をグリッドの横方向にマッピング
    grid_x = np.round(points[:, 0] / grid_resolution + grid_center_x).astype(int)
    # Z軸（前後）をグリッドの縦方向にマッピング
    grid_y = grid_center_y - np.round(points[:, 2] / grid_resolution).astype(int)
    # Y軸（上下）は高さとして使用
    height = points[:, 1]
    
    # グリッド内の点のみを処理
    valid_idx = (grid_x >= 0) & (grid_x < grid_width) & (grid_y >= 0) & (grid_y < grid_height)
    grid_x = grid_x[valid_idx]
    grid_y = grid_y[valid_idx]
    height = height[valid_idx]
    
    # グリッドセルごとに高さ情報を集計
    for i, (x, y) in enumerate(zip(grid_x, grid_y)):
        if height[i] < -height_threshold:
            # 床または通行可能な領域
            grid[y, x] = 2
        else:
            # 障害物
            grid[y, x] = 1
    
    return grid

def visualize_occupancy_grid(occupancy_grid):
    """
    占有グリッドを視覚化する関数
    Args:
        occupancy_grid: 占有グリッド（0=不明、1=障害物、2=通行可能）
    Returns:
        可視化された画像
    """
    # グリッドのサイズ
    grid_h, grid_w = occupancy_grid.shape
    
    # 表示用のキャンバスを作成（RGB）
    visualization = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # 障害物（値が1のセル）は赤色で表示
    visualization[occupancy_grid == 1] = [0, 0, 255]  # 赤色
    
    # 通行可能（値が2のセル）は緑色で表示
    visualization[occupancy_grid == 2] = [0, 100, 0]  # 緑色
    
    # 不明領域（値が0のセル）はグレーで表示
    visualization[occupancy_grid == 0] = [80, 80, 80]  # グレー
    
    # 中央に車両位置を示す点を描画
    center_x, center_y = grid_w // 2, grid_h - 20
    cv2.circle(visualization, (center_x, center_y), 5, [255, 255, 255], -1)
    
    # グリッド線を描画（10セルごと）
    for i in range(0, grid_h, 10):
        cv2.line(visualization, (0, i), (grid_w, i), [50, 50, 50], 1)
    for j in range(0, grid_w, 10):
        cv2.line(visualization, (j, 0), (j, grid_h), [50, 50, 50], 1)
    
    # 前方向を示す矢印を描画
    cv2.arrowedLine(visualization, 
                   (center_x, center_y),  # 始点
                   (center_x, center_y - 50),  # 終点
                   [255, 255, 255],  # 色
                   2,  # 線の太さ
                   tipLength=0.2)  # 矢印の先端の長さ
    
    return visualization

# 末尾のテストコードを修正
# 問題のコード: h, w = absolute_depth.shape[:2] が存在

# 以下のように修正：
if __name__ == "__main__":
    # このブロックはモジュールが直接実行されたときのみ実行される
    import numpy as np
    
    # テスト用のダミー深度マップ
    test_depth = np.zeros((240, 320), dtype=np.float32)
    
    # 中央に円形の障害物を配置
    for i in range(240):
        for j in range(320):
            dist = np.sqrt((i-120)**2 + (j-160)**2)
            if dist < 50:
                test_depth[i, j] = 0.5  # 近い障害物
            else:
                test_depth[i, j] = 1.0  # 遠い背景
    
    # 点群に変換
    test_points = depth_to_point_cloud(test_depth, 500, 500)
    
    # 占有グリッドに変換
    test_grid = create_top_down_occupancy_grid(test_points, 0.05, 200, 200, 0.5)
    
    # 可視化
    test_vis = visualize_occupancy_grid(test_grid)
    
    # 画像を保存（必要に応じて）
    # import cv2
    # cv2.imwrite("test_topview.jpg", test_vis)
    
    print("Test completed successfully")