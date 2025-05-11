"""
深度マップの可視化関連機能
"""

import cv2
import numpy as np
import logging

# ロガーの取得
logger = logging.getLogger("kuma_depth_nav.visualization")
# --- ここから追加 ---
# logger のレベルを DEBUG に設定
logger.setLevel(logging.DEBUG)
# ハンドラが設定されていなければ、標準出力へのハンドラを追加
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.info("Logger for 'kuma_depth_nav.visualization' INITIALIZED. This message should appear if the module is imported and logger is working.")
# --- ここまで追加 ---

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

def create_depth_grid_visualization(depth_grid_map, absolute_depth=None, cell_size=60): # 引数変更
    """
    圧縮済みの深度グリッドマップの可視化を行う
    
    Args:
        depth_grid_map: 圧縮済みの深度グリッドマップ (rows, cols)
        absolute_depth: 絶対深度マップ（オプション、現在はグリッドセルごとの絶対深度表示に使用）
        cell_size: セルサイズ（ピクセル）

    Returns:
        グリッド可視化画像
    """
    if depth_grid_map is None or depth_grid_map.size == 0:
        logger.warning("Empty depth_grid_map received for visualization.")
        return create_default_depth_image() # デフォルト画像を返す

    rows, cols = depth_grid_map.shape # grid_size は depth_grid_map.shape から取得

    try:
        # depth_grid_map を depth_conv として使用
        depth_conv = depth_grid_map

        # 以下、可視化用の処理
        valid_depth = depth_conv[depth_conv > 0.01]
        if len(valid_depth) > 0:
            min_depth = np.percentile(valid_depth, 5)
            max_depth = np.percentile(valid_depth, 95)
        else:
            logger.warning("No valid depth values found in depth_conv for grid visualization")
            min_depth = 0.1
            max_depth = 0.9

        normalized = np.zeros_like(depth_conv, dtype=np.float32)
        valid_mask = depth_conv > 0.01
        if np.any(valid_mask) and (max_depth > min_depth):
            normalized[valid_mask] = np.clip(
                (depth_conv[valid_mask] - min_depth) / (max_depth - min_depth + 1e-6), 
                0, 1
            )
            
        depth_colored = cv2.applyColorMap(
            (normalized * 255).astype(np.uint8), 
            cv2.COLORMAP_MAGMA
        )
        
        output = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)

        for i in range(rows):
            for j in range(cols):
                y_start, y_end = i * cell_size, (i + 1) * cell_size
                x_start, x_end = j * cell_size, (j + 1) * cell_size
                cell_color = depth_colored[i, j]
                output[y_start:y_end, x_start:x_end] = cell_color
                
                # セルに深度値を表示（絶対深度がある場合）
                # 注意: absolute_depth が渡された場合、それがグリッドセルに対応したデータであることを期待します。
                # もし元の高解像度 absolute_depth の場合、別途圧縮処理が必要です。
                # ここでは、depth_conv（圧縮済みの相対深度）から簡易的に絶対深度を計算して表示する例のままにします。
                if absolute_depth is not None: 
                    if depth_conv[i, j] > 1e-5: # ゼロ除算を避ける
                        # この変換は仮のものです。実際の絶対深度の計算方法に合わせてください。
                        depth_val = 15.0 / depth_conv[i, j] 
                        text = f"{depth_val:.1f}m"
                        cv2.putText(output, text, (x_start + 5, y_start + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # グリッド線を描画
        for i in range(rows + 1):
            y = i * cell_size
            cv2.line(output, (0, y), (cols * cell_size, y), (50, 50, 50), 1)
        for j in range(cols + 1):
            x = j * cell_size
            cv2.line(output, (x, 0), (x, rows * cell_size), (50, 50, 50), 1)

        return output
        
    except Exception as e:
        logger.error(f"Error in create_depth_grid_visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return create_default_depth_image()