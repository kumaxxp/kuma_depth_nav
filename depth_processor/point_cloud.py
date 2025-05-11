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

def depth_to_point_cloud(depth_data, fx, fy, cx, cy,
                         original_height=None, original_width=None,
                         is_grid_data=False, grid_rows=None, grid_cols=None):
    """
    深度データから3D点群を生成します。
    高解像度深度マップと圧縮グリッドデータの両方に対応します。

    Args:
        depth_data (numpy.ndarray): 深度データ。フル解像度マップまたは圧縮グリッド。
        fx (float): カメラの水平方向の焦点距離。
        fy (float): カメラの垂直方向の焦点距離。
        cx (float): カメラの水平方向の光学中心。
        cy (float): カメラの垂直方向の光学中心。
        original_height (int, optional): グリッドデータの場合の元の深度マップの高さ。
        original_width (int, optional): グリッドデータの場合の元の深度マップの幅。
        is_grid_data (bool): Trueの場合、depth_dataを圧縮グリッドとして処理。
        grid_rows (int, optional): グリッドデータの場合の行数。
        grid_cols (int, optional): グリッドデータの場合の列数。

    Returns:
        numpy.ndarray: (N, 3)形状の点群データ。各点は[x, y, z]。
                       無効な点や点がない場合は空の配列。
    """
    points = []
    if is_grid_data:
        if not (original_height and original_width and grid_rows and grid_cols and
                grid_rows > 0 and grid_cols > 0): # grid_rows/cols > 0 を追加
            # logger.error("For grid data, valid original_height, original_width, grid_rows, and grid_cols must be provided.")
            # print("Error: For grid data, valid original_height, original_width, grid_rows, and grid_cols must be provided.")
            raise ValueError("For grid data, valid original_height, original_width, grid_rows, and grid_cols must be provided.")
        
        # depth_data はこの場合 compressed_grid (grid_rows, grid_cols)
        for r_idx in range(grid_rows):
            for c_idx in range(grid_cols):
                depth_value = depth_data[r_idx, c_idx]
                
                # 無効な深度値はスキップ (0または非常に小さい値)
                if depth_value <= 1e-5: 
                    continue

                # グリッドセルに対応する元の画像上の中心ピクセル座標を計算
                # (u, v)座標系で、uが列方向(width)、vが行方向(height)
                u_center = (c_idx + 0.5) * (original_width / grid_cols)
                v_center = (r_idx + 0.5) * (original_height / grid_rows)
                
                # 3D座標を計算 (カメラ座標系: x右, y下, z前)
                x = (u_center - cx) * depth_value / fx
                y = (v_center - cy) * depth_value / fy
                z = depth_value
                points.append([x, y, z])
        
        if not points:
            return np.empty((0, 3), dtype=np.float32)
        return np.array(points, dtype=np.float32)
    
    else:
        # フル解像度深度マップの処理 (ベクトル化)
        if depth_data is None or depth_data.ndim != 2:
            # print("Error: Full resolution depth_data must be a 2D numpy array.")
            raise ValueError("Full resolution depth_data must be a 2D numpy array.")

        h, w = depth_data.shape
        if h == 0 or w == 0:
            return np.empty((0,3), dtype=np.float32)

        v_coords, u_coords = np.indices((h, w)) # vが行インデックス, uが列インデックス

        valid_mask = depth_data > 1e-5  # 有効な深度点のみを対象

        z_values = depth_data[valid_mask]
        if z_values.size == 0:
            return np.empty((0,3), dtype=np.float32)

        u_values = u_coords[valid_mask]
        v_values = v_coords[valid_mask]

        x_cam = (u_values - cx) * z_values / fx
        y_cam = (v_values - cy) * z_values / fy
        
        # (N, 3) の形状でスタック
        return np.stack((x_cam, y_cam, z_values), axis=-1)

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
        # 深度値が大きいほど近いという新しい実装に合わせて条件を修正
        # 負の値が大きいほど床に近い（低い位置）
        if height[i] < -height_threshold:  # Y座標が負で、しきい値より下（床レベル）
            # 床または通行可能な領域
            grid[y, x] = 2
        else:
            # 障害物（床より上にある物体）
            grid[y, x] = 1
    
    return grid

def visualize_occupancy_grid(occupancy_grid, scale_factor=5):
    """
    占有グリッドを視覚化する関数
    Args:
        occupancy_grid: 占有グリッド（0=不明、1=障害物、2=通行可能）
        scale_factor: 表示を拡大する係数
    Returns:
        可視化された画像
    """
    # グリッドのサイズ
    grid_h, grid_w = occupancy_grid.shape
    
    scaled_h = grid_h * scale_factor
    scaled_w = grid_w * scale_factor
    
    # 表示用のキャンバスを作成（RGB）
    visualization = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)
    
    # 色の定義 (BGR)
    colors = {
        0: [0, 255, 255],  # 不明領域: 黄色
        1: [0, 0, 255],    # 障害物: 赤色
        2: [0, 100, 0]     # 通行可能: 緑色
    }
    
    for r in range(grid_h):
        for c in range(grid_w):
            color = colors.get(occupancy_grid[r, c], [80, 80, 80]) # デフォルトはグレー
            cv2.rectangle(visualization,
                          (c * scale_factor, r * scale_factor),
                          ((c + 1) * scale_factor -1 , (r + 1) * scale_factor -1),
                          color,
                          -1)

    # 中央に車両位置を示す点を描画
    # 元のグリッドでの中心位置
    orig_center_x, orig_center_y = grid_w // 2, grid_h - 10 # y座標を少し調整して中央寄りに
    
    # スケール後の中心位置 (セルの中心になるように調整)
    center_x = orig_center_x * scale_factor + scale_factor // 2
    center_y = orig_center_y * scale_factor + scale_factor // 2
    
    marker_radius = 3 * scale_factor # マーカーの半径もスケール
    cv2.circle(visualization, (center_x, center_y), marker_radius, [255, 255, 255], -1)
    
    # グリッド線を描画（元の10セルごと、スケール後）
    line_color = [50, 50, 50]
    line_thickness = 1
    for i in range(0, grid_h + 1, 10): # +1 して最後の線も描画
        y = i * scale_factor
        cv2.line(visualization, (0, y), (scaled_w, y), line_color, line_thickness)
    for j in range(0, grid_w + 1, 10): # +1 して最後の線も描画
        x = j * scale_factor
        cv2.line(visualization, (x, 0), (x, scaled_h), line_color, line_thickness)
        
    # 前方向を示す矢印を描画
    arrow_length = 10 * scale_factor # 元のグリッドで10セル分の長さ
    arrow_thickness = max(1, scale_factor // 2) # 太さもスケール
    
    cv2.arrowedLine(visualization,
                   (center_x, center_y),
                   (center_x, center_y - arrow_length),
                   [255, 255, 255],
                   arrow_thickness,
                   tipLength=0.3)
    
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