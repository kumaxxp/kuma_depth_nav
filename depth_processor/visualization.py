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
        depth_feature = depth_map.reshape(depth_map.shape[-2:])
        
        # 深度の最小値と最大値（0に近い値は無視）
        valid_depth = depth_feature[depth_feature > 0.01]
        if len(valid_depth) > 0:
            min_depth = valid_depth.min()
            max_depth = valid_depth.max()
        else:
            min_depth = 0.0
            max_depth = 1.0  # デフォルト最大値を小さくする
            print("Warning: No valid depth values for visualization")
        
        # 正規化と色付け
        normalized = (depth_feature - min_depth) / (max_depth - min_depth + 1e-6)
        normalized = np.clip(normalized, 0, 1)  # 0-1の範囲に収める
        
        # デバッグ: 正規化された値を確認
        logger.debug(f"Normalized depth range: {normalized.min()}-{normalized.max()}")
        
        # 強制的に値を設定してテスト
        if normalized.max() < 0.1:  # 値が小さすぎる場合
            logger.warning("Normalized values too small, using default pattern")
            h, w = depth_feature.shape
            normalized = np.zeros((h, w), dtype=np.float32)
            for i in range(h):
                normalized[i, :] = i / h  # テストパターン
        
        depth_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        
        # 元の画像サイズにリサイズ
        if original_shape is not None and len(original_shape) >= 2:
            depth_resized = cv2.resize(depth_colored, (original_shape[1], original_shape[0]))
            
            # グラデーションバーを追加
            if add_colorbar:
                # バーの位置とサイズ
                bar_height = 30
                bar_margin = 30
                bar_width = original_shape[1] - 2 * bar_margin
                
                # キャンバスを拡大（下部にバーを追加するスペースを確保）
                canvas = np.zeros((original_shape[0] + bar_height + bar_margin * 2, original_shape[1], 3), dtype=np.uint8)
                # 深度画像を配置
                canvas[:original_shape[0], :] = depth_resized
                
                # グラデーションバー作成
                for i in range(bar_width):
                    normalized_pos = i / bar_width
                    color_idx = int(normalized_pos * 255)
                    color = cv2.applyColorMap(np.array([[color_idx]], dtype=np.uint8), cv2.COLORMAP_MAGMA)[0, 0]
                    bar_x = bar_margin + i
                    canvas[original_shape[0] + bar_margin:original_shape[0] + bar_margin + bar_height, bar_x:bar_x+1] = color
                
                # バーの境界線
                cv2.rectangle(
                    canvas, 
                    (bar_margin, original_shape[0] + bar_margin),
                    (bar_margin + bar_width, original_shape[0] + bar_margin + bar_height),
                    (255, 255, 255), 1
                )
                
                # 目盛りとラベル（5段階）
                for i in range(6):
                    # 目盛りの位置（均等に5分割）
                    tick_x = bar_margin + int(i * bar_width / 5)
                    tick_y_top = original_shape[0] + bar_margin + bar_height
                    tick_y_bottom = tick_y_top + 5
                    
                    # 目盛り線
                    cv2.line(canvas, (tick_x, tick_y_top), (tick_x, tick_y_bottom), (255, 255, 255), 1)
                    
                    # 深度値ラベル - 生の深度値を使用
                    depth_val = min_depth + (max_depth - min_depth) * i / 5
                    label = f"{depth_val:.2f}"
                    
                    # テキストサイズ取得
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    text_x = tick_x - text_size[0] // 2
                    text_y = tick_y_bottom + text_size[1] + 2
                    
                    # ラベル描画
                    cv2.putText(canvas, label, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                # タイトル
                title = "Depth"
                title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                title_x = (original_shape[1] - title_size[0]) // 2
                title_y = original_shape[0] + 5
                cv2.putText(canvas, title, (title_x, title_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                return canvas
            else:
                return depth_resized
        
        return depth_colored
        
    except Exception as e:
        print(f"Error in create_depth_visualization: {e}")
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

def create_depth_grid_visualization(depth_map, grid_cols=8, grid_rows=6, scaling_factor=15.0):
    """
    深度マップからグリッド表示を生成する
    
    Args:
        depth_map: 深度データ
        grid_cols: グリッドの列数
        grid_rows: グリッドの行数
        scaling_factor: 絶対深度変換のためのスケーリング係数
        
    Returns:
        可視化されたグリッド画像
    """
    if depth_map is None or depth_map.size == 0:
        return create_default_depth_image()
        
    depth_feature = depth_map.reshape(depth_map.shape[-2:])
    
    # 画像全体の大きさを定義
    h, w = depth_feature.shape
    cell_h, cell_w = h // grid_rows, w // grid_cols
    
    # グリッド画像を作成（黒背景）
    grid_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 各グリッドセルごとの処理
    for row in range(grid_rows):
        for col in range(grid_cols):
            # セル範囲の定義
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            
            # セル内の深度の平均値を計算（無効値は除外）
            cell_depth = depth_feature[y1:y2, x1:x2]
            valid_depths = cell_depth[cell_depth > 0.01]
            
            if len(valid_depths) > 0:
                avg_depth = np.mean(valid_depths)
                
                # 相対深度から絶対深度（メートル）へ変換
                abs_depth = scaling_factor / avg_depth if avg_depth > 0.01 else 0
                
                # 深度に基づいて色を決定（近い=赤、遠い=青）
                # 絶対深度を0-10mの範囲で正規化（10m以上は10mとして扱う）
                normalized_depth = min(abs_depth, 10.0) / 10.0
                
                # カラーマップの適用（HSV色空間で青から赤へ）
                color = cv2.applyColorMap(np.array([[int(255 * (1 - normalized_depth))]], dtype=np.uint8), 
                                         cv2.COLORMAP_JET)[0, 0]
                
                # セルを塗りつぶし
                grid_image[y1:y2, x1:x2] = color
                
                # 深度値をテキストで表示（メートル単位）
                text = f"{abs_depth:.1f}m"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                text_x = x1 + (cell_w - text_size[0]) // 2
                text_y = y1 + (cell_h + text_size[1]) // 2
                cv2.putText(grid_image, text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    # グリッド線を描画
    for row in range(grid_rows + 1):
        y = row * cell_h
        cv2.line(grid_image, (0, y), (w, y), (100, 100, 100), 1)
        
    for col in range(grid_cols + 1):
        x = col * cell_w
        cv2.line(grid_image, (x, 0), (x, h), (100, 100, 100), 1)
    
    return grid_image