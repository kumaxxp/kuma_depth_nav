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
    占有グリッドをカラー画像として可視化します。
    
    Args:
        occupancy_grid (numpy.ndarray): 形状 (grid_height, grid_width) の占有グリッド
            0: 不明（データなし）
            1: 占有（障害物）
            2: 通行可能
    
    Returns:
        numpy.ndarray: BGR形式のカラー画像
    """
    # グリッドの形状
    grid_height, grid_width = occupancy_grid.shape
    
    # カラーマッピング
    colors = {
        0: [211, 211, 211],  # 不明: ライトグレー [BGR]
        1: [0, 0, 255],      # 占有（障害物）: 赤
        2: [0, 255, 0]       # 通行可能: 緑
    }
    
    # カラー画像の初期化
    visualization = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # 各セルを色でマッピング
    for value, color in colors.items():
        mask = (occupancy_grid == value)
        visualization[mask] = color
    
    # カメラ位置（グリッドの中央下部）を青い円で表示
    camera_x = grid_width // 2
    camera_y = grid_height - 10
    cv2.circle(visualization, (camera_x, camera_y), 3, [255, 0, 0], -1)
    
    # 画像を拡大（640x480など）にリサイズ
    visualization = cv2.resize(visualization, (640, 480), interpolation=cv2.INTER_NEAREST)
    
    # グリッドの罫線を描画（オプション）
    grid_line_interval = int(480 / grid_height * 10)  # 10セルごとに罫線
    for i in range(0, 480, grid_line_interval):
        cv2.line(visualization, (0, i), (640, i), [80, 80, 80], 1)
    for i in range(0, 640, grid_line_interval):
        cv2.line(visualization, (i, 0), (i, 480), [80, 80, 80], 1)
    
    # 現在位置と進行方向を示す矢印
    cv2.arrowedLine(visualization, 
                   (320, 400),  # 開始点
                   (320, 300),  # 終了点
                   [255, 0, 0],  # 色（青）
                   2, # 太さ
                   tipLength=0.2) # 矢印の先端の長さ
    
    return visualization