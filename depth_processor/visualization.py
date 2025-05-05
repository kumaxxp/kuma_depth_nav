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

def create_depth_grid_visualization(depth_map, grid_cols=8, grid_rows=6, text_color=(255, 255, 255)):
    """
    深度マップをグリッドに分割し、各セルの平均深度値を表示する可視化を作成
    
    Args:
        depth_map: 深度マップ
        grid_cols: 横方向の分割数
        grid_rows: 縦方向の分割数
        text_color: テキスト色 (BGR形式)
        
    Returns:
        可視化された画像
    """
    # 深度マップの形状を取得
    if len(depth_map.shape) > 2:
        depth_map = depth_map.squeeze()  # バッチ次元や不要な次元を除去
        
    height, width = depth_map.shape
    
    # 出力画像を作成（やや暗めの背景）
    visualization = np.zeros((480, 640, 3), dtype=np.uint8) + 30
    
    # セルの幅と高さを計算
    cell_width = width // grid_cols
    cell_height = height // grid_rows
    
    # ビジュアライゼーションのセルサイズ
    vis_cell_width = 640 // grid_cols
    vis_cell_height = 480 // grid_rows
    
    # 各セルの平均深度を計算して表示
    for row in range(grid_rows):
        for col in range(grid_cols):
            # 元の深度マップのセル領域
            y_start = row * cell_height
            y_end = min((row + 1) * cell_height, height)
            x_start = col * cell_width
            x_end = min((col + 1) * cell_width, width)
            
            # この領域の深度値を抽出
            cell_depth = depth_map[y_start:y_end, x_start:x_end]
            
            # 有効な深度値の平均を計算（0に近い値は無効として除外）
            valid_depth = cell_depth[cell_depth > 0.001]
            if len(valid_depth) > 0:
                avg_depth = np.mean(valid_depth)
            else:
                avg_depth = 0
            
            # ビジュアライゼーションのセル座標
            vis_y_start = row * vis_cell_height
            vis_y_end = (row + 1) * vis_cell_height
            vis_x_start = col * vis_cell_width
            vis_x_end = (col + 1) * vis_cell_width
            
            # 深度値に応じた色を決定
            if avg_depth > 0:
                # 遠いほど赤、近いほど青
                normalized_depth = min(avg_depth / 5.0, 1.0)  # 5mを最大とする
                b = int(255 * (1.0 - normalized_depth))
                r = int(255 * normalized_depth)
                color = (b, 80, r)  # BGR形式
                
                # セルを塗りつぶし
                visualization[vis_y_start:vis_y_end, vis_x_start:vis_x_end] = color
                
                # 平均深度値をテキストとして表示
                text = f"{avg_depth:.2f}m"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = vis_x_start + (vis_cell_width - text_size[0]) // 2
                text_y = vis_y_start + (vis_cell_height + text_size[1]) // 2
                cv2.putText(visualization, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            else:
                # データがない領域はダークグレー
                visualization[vis_y_start:vis_y_end, vis_x_start:vis_x_end] = (30, 30, 30)
                
                # "N/A"と表示
                text = "N/A"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = vis_x_start + (vis_cell_width - text_size[0]) // 2
                text_y = vis_y_start + (vis_cell_height + text_size[1]) // 2
                cv2.putText(visualization, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)
    
    # グリッド線を描画
    for row in range(1, grid_rows):
        y = row * vis_cell_height
        cv2.line(visualization, (0, y), (640, y), (100, 100, 100), 1)
    
    for col in range(1, grid_cols):
        x = col * vis_cell_width
        cv2.line(visualization, (x, 0), (x, 480), (100, 100, 100), 1)
        
    return visualization