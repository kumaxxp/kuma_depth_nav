"""
深度マップの可視化関連機能
"""

import cv2
import numpy as np
import logging

# ロガーの取得
logger = logging.getLogger("kuma_depth_nav.visualization")

def create_depth_visualization(depth_map, original_shape, add_colorbar=True):
    """深度マップの可視化を行う"""
    try:
        if depth_map is None or depth_map.size == 0:
            logger.warning("Empty depth map received for visualization")
            return create_default_depth_image(
                640 if original_shape is None else original_shape[1],
                480 if original_shape is None else original_shape[0]
            )
            
        # 深度マップの形状をログ出力
        logger.debug(f"Visualizing depth map with shape: {depth_map.shape}")
        
        # 深度マップを2次元に変換
        if len(depth_map.shape) == 4:  # (1, H, W, 1) 形式
            depth_feature = depth_map.reshape(depth_map.shape[1:3])
        elif len(depth_map.shape) == 3:  # (H, W, 1) または (1, H, W) 形式
            if depth_map.shape[2] == 1:
                depth_feature = depth_map.reshape(depth_map.shape[:2])
            else:
                depth_feature = depth_map.reshape(depth_map.shape[1:])
        else:
            depth_feature = depth_map  # すでに2D
            
        # NaNやInfをチェックして置換
        depth_feature = np.nan_to_num(depth_feature, nan=0.5, posinf=1.0, neginf=0.1)
        
        # 値の範囲を確認
        logger.debug(f"Depth feature range: {np.min(depth_feature):.4f} to {np.max(depth_feature):.4f}")
        
        # 深度の正規化（無効値を除外）
        valid_depth = depth_feature[depth_feature > 0.01]
        if len(valid_depth) > 0:
            min_depth = np.percentile(valid_depth, 5)  # 外れ値を除外
            max_depth = np.percentile(valid_depth, 95) # 外れ値を除外
        else:
            logger.warning("No valid depth values found")
            min_depth = 0.1
            max_depth = 0.9
            
        logger.debug(f"Using depth range for normalization: {min_depth:.4f} to {max_depth:.4f}")
        
        # 正規化して0-1範囲にする
        normalized = np.zeros_like(depth_feature, dtype=np.float32)
        valid_mask = depth_feature > 0.01
        if np.any(valid_mask) and (max_depth > min_depth):
            normalized[valid_mask] = np.clip(
                (depth_feature[valid_mask] - min_depth) / (max_depth - min_depth + 1e-6), 
                0, 1
            )
            
        # colormap適用
        depth_colored = cv2.applyColorMap(
            (normalized * 255).astype(np.uint8), 
            cv2.COLORMAP_MAGMA
        )
        
        # 元の画像サイズにリサイズ
        if original_shape is not None and len(original_shape) >= 2:
            return cv2.resize(depth_colored, (original_shape[1], original_shape[0]))
            
        return depth_colored
        
    except Exception as e:
        logger.error(f"Error in create_depth_visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # エラー時はデフォルト画像を返す
        return create_default_depth_image(
            640 if original_shape is None else original_shape[1], 
            480 if original_shape is None else original_shape[0]
        )

def create_default_depth_image(width=640, height=480):
    """
    デフォルトの深度イメージを生成（モデルがない場合のプレースホルダ）
    
    Args:
        width: 画像幅
        height: 画像高さ
        
    Returns:
        デフォルトの深度イメージ
    """
    default_depth_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        default_depth_image[i, :] = [0, 0, int(255 * i / height)]
    return default_depth_image

def depth_to_color(depth_normalized):
    """
    深度値を色に変換（青から赤のグラデーション）
    
    Args:
        depth_normalized: 正規化された深度値（0.0〜1.0）
        
    Returns:
        色（BGR形式）
    """
    # HSV色空間での青から赤へのグラデーション
    hue = int((1.0 - depth_normalized) * 120)  # 0〜120の範囲
    saturation = 255
    value = 255
    
    # HSVからBGRへの変換
    return cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0]

def create_depth_grid_visualization(depth_map, absolute_depth=None, grid_size=(8, 8), max_distance=10.0, cell_size=60):
    """
    深度マップをグリッド形式で可視化（改善版）
    
    Args:
        depth_map: 深度マップ（相対深度）
        absolute_depth: 絶対深度マップ（メートル単位）
        grid_size: グリッドのサイズ（行数, 列数）
        max_distance: 最大深度（メートル）
        cell_size: セルの大きさ（ピクセル）
        
    Returns:
        numpy.ndarray: グリッド可視化画像
    """
    rows, cols = grid_size
    h, w = depth_map.shape[:2] if len(depth_map.shape) > 2 else depth_map.shape
    
    # 絶対深度がない場合は作成
    if absolute_depth is None:
        try:
            # スケーリング係数でメートル単位に変換
            absolute_depth = convert_to_absolute_depth(depth_map, scaling_factor=15.0)
            logger.debug("Created absolute depth from relative depth")
        except Exception as e:
            logger.error(f"Error converting to absolute depth: {e}")
            return np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8) + 30
            
    # グリッド分割のためのインデックス計算
    row_indices = np.linspace(0, h-1, rows+1).astype(int)
    col_indices = np.linspace(0, w-1, cols+1).astype(int)
    
    # デバッグ: 深度値の範囲を確認
    try:
        valid_depths = absolute_depth[absolute_depth > 0.01]
        if valid_depths.size > 0:
            min_depth = valid_depths.min()
            max_depth = valid_depths.max()
            mean_depth = valid_depths.mean()
            logger.debug(f"Depth range: min={min_depth:.2f}m, max={max_depth:.2f}m, mean={mean_depth:.2f}m")
        else:
            logger.warning("No valid depth values found")
    except Exception as e:
        logger.error(f"Error analyzing depth data: {e}")
    
    # 出力画像の初期化
    output_h = rows * cell_size
    output_w = cols * cell_size
    output = np.zeros((output_h, output_w, 3), dtype=np.uint8) + 30  # ダークグレー背景
    
    # テキスト設定
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 1
    
    # 各セルの平均深度を計算して可視化
    for i in range(rows):
        for j in range(cols):
            # セルの領域を取得
            r_start, r_end = row_indices[i], row_indices[i+1]
            c_start, c_end = col_indices[j], col_indices[j+1]
            
            # セルの深度値を取得 - より多くのピクセルを取得するため、境界を少し拡張
            r_start_exp = max(0, r_start - 5)
            r_end_exp = min(h, r_end + 5)
            c_start_exp = max(0, c_start - 5)
            c_end_exp = min(w, c_end + 5)
            
            cell_depth = absolute_depth[r_start_exp:r_end_exp, c_start_exp:c_end_exp]
            
            # 有効な深度値の条件を緩和（0.1m〜15mの範囲で有効と判断）
            valid_depth = cell_depth[(cell_depth > 0.1) & (cell_depth < 15.0)]
            
            cell_y_start = i * cell_size
            cell_x_start = j * cell_size
            
            # データ量の閾値を下げる（より少ないデータでも有効と判断）