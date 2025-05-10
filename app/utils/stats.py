"""
パフォーマンス統計情報を収集・管理するモジュール
"""
import time
import numpy as np
import threading
from collections import deque

class PerformanceStats:
    """パフォーマンス統計情報を管理するクラス"""
    
    def __init__(self):
        # 処理時間計測用
        self.camera_times = deque(maxlen=1000)
        self.inference_times = deque(maxlen=1000)
        self.visualization_times = deque(maxlen=1000)
        self.encoding_times = deque(maxlen=1000)

        # パフォーマンス統計用の変数
        self.fps_stats = {
            "camera": deque(maxlen=30),
            "depth": deque(maxlen=30),
            "grid": deque(maxlen=30),
            "inference": deque(maxlen=30)
        }
        
        self.last_frame_times = {
            "camera": 0,
            "depth": 0,
            "grid": 0,
        }
        
        # スレッド開始
        threading.Thread(target=self.log_processing_times, daemon=True).start()
    
    def log_processing_times(self):
        """5秒ごとに平均、最大、最小の処理時間をログ出力"""
        while True:
            time.sleep(5)
            if self.camera_times:
                print(f"[Camera] Avg: {np.mean(self.camera_times):.4f}s, Max: {np.max(self.camera_times):.4f}s, Min: {np.min(self.camera_times):.4f}s")
            if self.inference_times:
                print(f"[Inference] Avg: {np.mean(self.inference_times):.4f}s, Max: {np.max(self.inference_times):.4f}s, Min: {np.min(self.inference_times):.4f}s")
            if self.visualization_times:
                print(f"[Visualization] Avg: {np.mean(self.visualization_times):.4f}s, Max: {np.max(self.visualization_times):.4f}s, Min: {np.min(self.visualization_times):.4f}s")
            if self.encoding_times:
                print(f"[Encoding] Avg: {np.mean(self.encoding_times):.4f}s, Max: {np.max(self.encoding_times):.4f}s, Min: {np.min(self.encoding_times):.4f}s")
    
    def update_fps(self, stream_type, now):
        """FPSを更新する
        
        Args:
            stream_type: ストリームの種類（"camera", "depth", "grid"など）
            now: 現在の時刻
        
        Returns:
            bool: 最初のフレームかどうか
        """
        first_frame = False
        
        # 最初のフレームの場合は初期化
        if self.last_frame_times[stream_type] == 0:
            first_frame = True
        elif (now - self.last_frame_times[stream_type]) < 0.5:  # 0.5秒以内の正常な間隔
            # 正常な間隔の場合のみFPSを計算
            fps = 1.0 / (now - self.last_frame_times[stream_type])
            # 異常値フィルタリング (FPSが200を超える値はエラーと見なす)
            if fps < 200:  
                self.fps_stats[stream_type].append(fps)
        
        # 現在時刻を記録
        self.last_frame_times[stream_type] = now
        return first_frame
    
    def get_median_fps(self, stream_type):
        """中央値FPSを取得する
        
        Args:
            stream_type: ストリームの種類（"camera", "depth", "grid"など）
            
        Returns:
            float: 中央値FPS。データがない場合は0。
        """
        if not self.fps_stats[stream_type]:
            return 0
            
        values = list(self.fps_stats[stream_type])
        values.sort()
        return values[len(values) // 2]
    
    def get_stats_data(self, frame_timestamp):
        """統計情報を取得する
        
        Args:
            frame_timestamp: 最新フレームのタイムスタンプ
            
        Returns:
            dict: 統計情報
        """
        current_delay = (time.time() - frame_timestamp) * 1000 if frame_timestamp > 0 else 0
        
        # 中央値を計算するヘルパー関数
        def median(values):
            if not values:
                return 0
            values_list = list(values)
            values_list.sort()
            return values_list[len(values_list) // 2]
            
        stats = {
            "fps": {
                # 中央値を使用
                "camera": round(median(self.fps_stats["camera"]), 1) if self.fps_stats["camera"] else 0,
                "depth": round(median(self.fps_stats["depth"]), 1) if self.fps_stats["depth"] else 0,
                "grid": round(median(self.fps_stats["grid"]), 1) if self.fps_stats["grid"] else 0,
                "inference": round(median(self.fps_stats["inference"]), 1) if self.fps_stats["inference"] else 0,
            },
            "latency": {
                "camera": round(np.mean(self.camera_times) * 1000, 1) if self.camera_times else 0,
                "inference": round(np.mean(self.inference_times) * 1000, 1) if self.inference_times else 0,
                "visualization": round(np.mean(self.visualization_times) * 1000, 1) if self.visualization_times else 0,
                "encoding": round(np.mean(self.encoding_times) * 1000, 1) if self.encoding_times else 0,
                "total_delay": round(current_delay, 1)  # 現在の総遅延
            }
        }
        return stats

# グローバルなインスタンスを作成
stats = PerformanceStats()
