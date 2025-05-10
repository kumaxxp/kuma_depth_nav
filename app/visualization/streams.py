"""
深度マップの可視化とストリーミングを管理するモジュール
"""
import cv2
import time
import numpy as np
from depth_processor import create_depth_visualization, create_depth_grid_visualization
from ..utils.stats import stats
from ..camera.capture import camera
from ..depth.inference import depth_inference

class VisualizationStreams:
    """可視化ストリームを管理するクラス"""
    
    def get_depth_stream(self):
        """深度マップのストリームジェネレーター
        
        Yields:
            bytes: HTTP multipart形式のJPEGフレーム
        """
        while True:
            # 深度マップとカメラフレームを取得
            current_depth_map = depth_inference.get_depth_map()
            current_frame, _ = camera.get_frame()
            
            if current_depth_map is None or current_frame is None:
                time.sleep(0.01)
                continue
                
            # 深度マップの可視化
            start_time = time.perf_counter()
            vis = create_depth_visualization(current_depth_map, (128, 128))
            vis = cv2.resize(vis, (320, 240), interpolation=cv2.INTER_NEAREST)
            stats.visualization_times.append(time.perf_counter() - start_time)
    
            # FPS計算とテキスト表示
            now = time.time()
            first_frame = stats.update_fps("depth", now)
            
            if not first_frame and len(stats.fps_stats["depth"]) > 0:
                avg_fps = stats.get_median_fps("depth")
                cv2.putText(vis, f"FPS: {avg_fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 遅延情報を表示
                frame_timestamp = camera.frame_timestamp
                delay = (time.time() - frame_timestamp) * 1000
                cv2.putText(vis, f"Delay: {delay:.1f}ms", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
            # JPEG エンコード
            start_time = time.perf_counter()
            ret, buffer = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            stats.encoding_times.append(time.perf_counter() - start_time)
            if not ret:
                continue
    
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.015)  # 約66FPS
    
    def get_depth_grid_stream(self):
        """深度グリッドのストリームジェネレーター
        
        Yields:
            bytes: HTTP multipart形式のJPEGフレーム
        """
        while True:
            # 深度マップとカメラフレームを取得
            current_depth_map = depth_inference.get_depth_map()
            current_frame, _ = camera.get_frame()
            
            if current_depth_map is None or current_frame is None:
                time.sleep(0.01)
                continue
                
            # グリッドの可視化
            start_time = time.perf_counter()
            grid_img = create_depth_grid_visualization(current_depth_map, grid_size=(10, 10), cell_size=18)
            if grid_img is None or len(grid_img.shape) < 2:
                grid_img = 128 * np.ones((240, 320, 3), dtype=np.uint8)
            elif len(grid_img.shape) == 2 or (len(grid_img.shape) == 3 and grid_img.shape[2] == 1):
                grid_img = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)
            grid_img = cv2.resize(grid_img, (320, 240), interpolation=cv2.INTER_NEAREST)
            stats.visualization_times.append(time.perf_counter() - start_time)
    
            # FPS計算とテキスト表示
            now = time.time()
            first_frame = stats.update_fps("grid", now)
            
            if not first_frame and len(stats.fps_stats["grid"]) > 0:
                avg_fps = stats.get_median_fps("grid")
                cv2.putText(grid_img, f"FPS: {avg_fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 遅延情報を表示
                frame_timestamp = camera.frame_timestamp
                delay = (time.time() - frame_timestamp) * 1000
                cv2.putText(grid_img, f"Delay: {delay:.1f}ms", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
            # JPEG エンコード
            start_time = time.perf_counter()
            ret, buffer = cv2.imencode('.jpg', grid_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            stats.encoding_times.append(time.perf_counter() - start_time)
            if not ret:
                continue
    
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.015)  # 約66FPS

# グローバルインスタンス
visualization = VisualizationStreams()
