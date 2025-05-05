"""
深度マップの可視化関連機能
"""

import cv2
import numpy as np

def create_depth_visualization(depth_map, original_shape):
    """
    深度マップの可視化を行う
    
    Args:
        depth_map: 深度データ
        original_shape: オリジナル画像のサイズ
        
    Returns:
        可視化されたカラー深度画像
    """
    depth_feature = depth_map.reshape(depth_map.shape[-2:])
    
    # 正規化と色付け
    normalized = (depth_feature - depth_feature.min()) / (depth_feature.max() - depth_feature.min() + 1e-6)
    depth_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    
    # 元の画像サイズにリサイズ
    if original_shape is not None and len(original_shape) >= 2:
        depth_resized = cv2.resize(depth_colored, (original_shape[1], original_shape[0]))
        return depth_resized
    
    return depth_colored

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