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
    深度マップをグリッド形式で可視化（深度マップと同じカラーマップを使用）
    
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
    
    # 出力画像の初期化
    output_h = rows * cell_size
    output_w = cols * cell_size
    output = np.zeros((output_h, output_w, 3), dtype=np.uint8) + 30  # ダークグレー背景
    
    # グリッド分割のためのインデックス計算
    row_indices = np.linspace(0, h-1, rows+1).astype(int)
    col_indices = np.linspace(0, w-1, cols+1).astype(int)
    
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
            
            # セルの領域を拡張（境界線上の点を含める）
            r_start_ext = max(0, r_start - 2)
            r_end_ext = min(h, r_end + 2)
            c_start_ext = max(0, c_start - 2)
            c_end_ext = min(w, c_end + 2)
            
            # セルの深度値を取得
            if absolute_depth is not None:
                # 絶対深度から直接取得
                cell_depth = absolute_depth[r_start_ext:r_end_ext, c_start_ext:c_end_ext]
                # 無効な値をフィルタ（0.1m〜30mの範囲）
                valid_depth = cell_depth[(cell_depth >= 0.1) & (cell_depth <= 30.0)]
            else:
                # 相対深度から計算
                cell_depth = depth_map[r_start_ext:r_end_ext, c_start_ext:c_end_ext]
                # 無効な値をフィルタ
                valid_depth = cell_depth[cell_depth > 0.01]
                # 絶対深度に変換（簡易的なスケーリング）
                if valid_depth.size > 0:
                    valid_depth = 15.0 / valid_depth
                    # 範囲外の値をフィルタ
                    valid_depth = valid_depth[(valid_depth >= 0.1) & (valid_depth <= 30.0)]
            
            # セルの位置
            cell_y_start = i * cell_size
            cell_x_start = j * cell_size
            
            # 有効なデータがあるかどうか
            if valid_depth.size > 5:  # 最低5ピクセルは必要
                # 外れ値を除外するため、25%トリミーン平均を使用
                sorted_depths = np.sort(valid_depth.flatten())
                trim_size = max(1, int(sorted_depths.size * 0.25))
                if sorted_depths.size > trim_size*2:
                    trimmed = sorted_depths[trim_size:-trim_size]
                    avg_depth = np.mean(trimmed)
                else:
                    avg_depth = np.mean(sorted_depths)
                
                # 深度に応じた色を設定（深度マップと同じMAGMAカラーマップを使用）
                norm_depth = min(avg_depth / max_distance, 1.0)
                
                # MAGMA カラーマップを適用（深度マップと同じ）
                # 0-255の範囲に変換して、MAGMAカラーマップを適用
                depth_as_uint8 = np.array([(1.0 - norm_depth) * 255], dtype=np.uint8)
                color_pixel = cv2.applyColorMap(depth_as_uint8, cv2.COLORMAP_MAGMA)[0][0]
                color = (int(color_pixel[0]), int(color_pixel[1]), int(color_pixel[2]))
                
                # セルを描画
                cv2.rectangle(
                    output, 
                    (cell_x_start + 2, cell_y_start + 2), 
                    (cell_x_start + cell_size - 2, cell_y_start + cell_size - 2),
                    color, 
                    -1
                )
                
                # 数値を表示
                text = f"{avg_depth:.1f}m"
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = cell_x_start + (cell_size - text_size[0]) // 2
                text_y = cell_y_start + (cell_size + text_size[1]) // 2
                
                # テキストの縁取り（読みやすさ向上）
                cv2.putText(
                    output, 
                    text, 
                    (text_x, text_y), 
                    font, 
                    font_scale, 
                    (0, 0, 0),  # 黒（縁取り）
                    font_thickness + 2
                )
                
                # テキスト本体
                cv2.putText(
                    output, 
                    text, 
                    (text_x, text_y), 
                    font, 
                    font_scale, 
                    (255, 255, 255),  # 白
                    font_thickness
                )
            else:
                # データが不足している場合は灰色のセル
                cv2.rectangle(
                    output, 
                    (cell_x_start + 2, cell_y_start + 2), 
                    (cell_x_start + cell_size - 2, cell_y_start + cell_size - 2),
                    (80, 80, 80),  # 暗い灰色
                    -1
                )
                
                # 数値がないことを示す
                text = "---"
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = cell_x_start + (cell_size - text_size[0]) // 2
                text_y = cell_y_start + (cell_size + text_size[1]) // 2
                
                cv2.putText(
                    output, 
                    text, 
                    (text_x, text_y), 
                    font, 
                    font_scale, 
                    (150, 150, 150),  # 明るい灰色
                    font_thickness
                )
    
    # グリッド線を描画
    for i in range(rows+1):
        y = i * cell_size
        cv2.line(output, (0, y), (output_w, y), (50, 50, 50), 1)
    
    for j in range(cols+1):
        x = j * cell_size
        cv2.line(output, (x, 0), (x, output_h), (50, 50, 50), 1)
    
    # カラーバーを追加（オプション）
    colorbar_height = 20
    colorbar_width = output_w
    colorbar = np.zeros((colorbar_height, colorbar_width, 3), dtype=np.uint8)
    
    for x in range(colorbar_width):
        normalized_value = x / colorbar_width
        color_value = np.array([int((1 - normalized_value) * 255)], dtype=np.uint8)
        color_pixel = cv2.applyColorMap(color_value, cv2.COLORMAP_MAGMA)[0][0]
        colorbar[:, x] = color_pixel
    
    # カラーバーラベルを追加
    cv2.putText(colorbar, "近い", (10, 15), font, 0.5, (255, 255, 255), 1)
    cv2.putText(colorbar, f"{max_distance}m", (colorbar_width - 50, 15), font, 0.5, (255, 255, 255), 1)
    
    # カラーバーを出力画像に結合
    output_with_colorbar = np.vstack([output, colorbar])
    
    return output_with_colorbar