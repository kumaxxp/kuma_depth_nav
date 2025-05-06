# filepath: c:\work\kuma_depth_nav\source\create_favicon.py
import cv2
import numpy as np
import os

def create_favicon():
    """シンプルなファビコンを作成"""
    # 32x32のキャンバス
    favicon = np.zeros((32, 32, 3), dtype=np.uint8)
    
    # 青い背景
    favicon[:,:] = (0, 0, 128)
    
    # 中央に赤い円
    cv2.circle(favicon, (16, 16), 10, (0, 0, 255), -1)
    
    # 保存先ディレクトリの確認
    target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'source', 'depth_cap')
    os.makedirs(target_dir, exist_ok=True)
    
    # 保存
    favicon_path = os.path.join(target_dir, "favicon.ico")
    cv2.imwrite(favicon_path, favicon)
    print(f"ファビコンを作成しました: {favicon_path}")

if __name__ == "__main__":
    create_favicon()