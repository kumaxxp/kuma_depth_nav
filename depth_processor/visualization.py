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

def create_depth_grid_visualization(depth_map, absolute_depth=None, grid_size=(8, 6), max_distance=10.0, cell_size=60):

    rows, cols = grid_size


    """深度マップの可視化を行う"""
    try:
        if depth_map is None or depth_map.size == 0:
            logger.warning("Empty depth map received for visualization")
            return create_default_depth_image()
            
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
        depth_feature = np.nan_to_num(depth_feature, nan=0.5, posinf=1.0, neginf=0.01)

        # depth_featureをgrid_sizeのサイズに畳み込んだdepth_convを作成
        depth_conv = np.zeros((rows, cols), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                # 各グリッドの範囲を計算
                row_start = int(i * depth_feature.shape[0] / rows)
                row_end = int((i + 1) * depth_feature.shape[0] / rows)
                col_start = int(j * depth_feature.shape[1] / cols)
                col_end = int((j + 1) * depth_feature.shape[1] / cols)

                # グリッド内の深度値を平均化
                depth_conv[i, j] = np.mean(depth_feature[row_start:row_end, col_start:col_end])


        # depth_convを正規化（無効値を除外）
        valid_depth = depth_conv[depth_conv > 0.01]
        if len(valid_depth) > 0:
            min_depth = np.percentile(valid_depth, 5)  # 外れ値を除外
            max_depth = np.percentile(valid_depth, 95) # 外れ値を除外
        else:
            logger.warning("No valid depth values found")
            min_depth = 0.1
            max_depth = 0.9
        logger.debug(f"Using depth range for normalization: {min_depth:.4f} to {max_depth:.4f}")

        # 正規化して0-1範囲にする
        normalized = np.zeros_like(depth_conv, dtype=np.float32)
        valid_mask = depth_conv > 0.01
        if np.any(valid_mask) and (max_depth > min_depth):
            normalized[valid_mask] = np.clip(
                (depth_conv[valid_mask] - min_depth) / (max_depth - min_depth + 1e-6), 
                0, 1
            )
                    
        # colormap適用
        depth_colored = cv2.applyColorMap(
            (normalized * 255).astype(np.uint8), 
            cv2.COLORMAP_MAGMA
        )

        return depth_colored
        
    except Exception as e:
        logger.error(f"Error in create_depth_visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # エラー時はデフォルト画像を返す
        return create_default_depth_image()

    
    return output_with_colorbar