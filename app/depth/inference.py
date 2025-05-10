"""
深度推論処理を管理するモジュール
"""
import cv2
import time
import threading
import numpy as np
from depth_processor import DepthProcessor
from ..utils.stats import stats
from ..camera.capture import camera

class DepthInference:
    """深度推論を管理するクラス"""
    
    def __init__(self):
        # 深度処理用のリソース初期化
        self.depth_processor = DepthProcessor()
        self.latest_depth_map = None
        self.lock = threading.Lock()
        self.last_inference_time = 0
        self.INFERENCE_INTERVAL = 0.08  # 推論間隔: 0.08秒（約12.5FPS）
        
        # 推論スレッド開始
        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
    
    def _inference_loop(self):
        """推論スレッドのメインループ"""
        while self.running:
            current_time = time.time()
            # 前回の推論から一定時間経過した場合のみ推論実行
            if current_time - self.last_inference_time > self.INFERENCE_INTERVAL:
                # 共有メモリからカメラフレームを取得
                frame, capture_time = camera.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
        # 推論用にリサイズ
                small = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
                start_time = time.perf_counter()
                
                try:
                    # axengineが使用可能な場合のみ推論を実行
                    depth_map, _ = self.depth_processor.predict(small)
                except Exception as e:
                    # 推論エラーの場合はダミーの深度マップを生成
                    print(f"推論エラー: {e}")
                    depth_map = np.zeros((128, 128), dtype=np.float32)
                
                inference_time = time.perf_counter() - start_time
                stats.inference_times.append(inference_time)
                
                # FPS計算
                now = time.time()
                if self.last_inference_time > 0:
                    fps = 1.0 / (now - self.last_inference_time)
                    stats.fps_stats["inference"].append(fps)
                
                # ロックを取得して共有メモリを更新
                with self.lock:
                    self.latest_depth_map = depth_map
                    self.last_inference_time = now
                
                # 遅延を計算して表示
                delay = now - capture_time
                print(f"[Thread] Inference completed in {inference_time:.4f}s, Delay: {delay*1000:.1f}ms")
            else:
                # 推論間隔が来るまで少し待機
                time.sleep(0.01)  # 応答性を保つため短い待機時間
    
    def get_depth_map(self):
        """最新の深度マップを取得する
        
        Returns:
            numpy.ndarray: 深度マップ。マップがない場合はNone。
        """
        with self.lock:
            if self.latest_depth_map is None:
                return None
            return self.latest_depth_map.copy()
    
    def release(self):
        """リソースを解放する"""
        self.running = False
        if self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)

# グローバルインスタンス
depth_inference = DepthInference()
