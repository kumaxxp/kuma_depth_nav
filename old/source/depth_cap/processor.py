"""
画像処理モジュール

カメラから取得した映像フレームに対する処理を実装します。
"""
import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Dict, Any, Callable, List, Tuple

# ロガー設定
logger = logging.getLogger(__name__)

class ImageProcessor:
    """画像処理クラス"""
    
    def __init__(self, camera_manager):
        """
        画像処理クラスの初期化
        
        Args:
            camera_manager: カメラマネージャーインスタンス
        """
        self.camera_manager = camera_manager
        self.is_running = False
        self.thread = None
        self.processed_frame = None
        self.processed_frame_time = 0
        self.processing_fps = 0
        self.last_fps_update = time.time()
        self.frame_count = 0
        self.lock = threading.RLock()
        
        # 処理結果データ
        self.analysis_data = {}
        
    def start_processing(self) -> bool:
        """
        処理スレッドを開始します
        
        Returns:
            bool: 開始成功時True
        """
        if self.thread and self.thread.is_alive():
            logger.warning("既に処理スレッドが実行中です")
            return True
            
        self.is_running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        logger.info("画像処理スレッド開始")
        return True
        
    def stop_processing(self):
        """処理スレッドを停止します"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            logger.info("画像処理スレッド終了")
            
    def _processing_loop(self):
        """画像処理ループ (内部スレッド用)"""
        fps_update_interval = 1.0  # FPS計算間隔(秒)
        frames_processed = 0
        
        while self.is_running:
            try:
                # カメラマネージャーから処理用フレームを取得
                frame_data = self.camera_manager.get_frame_for_processing()
                if frame_data is None:
                    time.sleep(0.01)  # フレームがない場合は少し待つ
                    continue
                    
                frame, timestamp = frame_data
                
                # フレームの処理を実行
                processed_frame, analysis_data = self._process_frame(frame)
                
                # 結果を保存 (スレッドセーフに)
                with self.lock:
                    self.processed_frame = processed_frame
                    self.processed_frame_time = timestamp
                    self.analysis_data = analysis_data
                    self.frame_count += 1
                    frames_processed += 1
                
                # FPSの計算
                current_time = time.time()
                if current_time - self.last_fps_update >= fps_update_interval:
                    elapsed = current_time - self.last_fps_update
                    self.processing_fps = frames_processed / elapsed
                    frames_processed = 0
                    self.last_fps_update = current_time
                    logger.debug(f"画像処理FPS: {self.processing_fps:.1f}")
                
            except Exception as e:
                logger.error(f"画像処理中エラー: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        フレームに対する実際の処理を行います
        このメソッドをオーバーライドして実際の処理を実装します
        
        Args:
            frame: 処理対象のフレーム
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (処理後フレーム, 分析データ)
        """
        # この実装はシンプルな例です。派生クラスでオーバーライドします。
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 二値化
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 輪郭検出
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 輪郭を描画
        result = frame.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        # テキスト表示
        cv2.putText(
            result, 
            f"Processed: {time.strftime('%H:%M:%S')}",
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0,
            (0, 0, 255),
            2
        )
        
        # 分析データの作成
        analysis_data = {
            "contour_count": len(contours),
            "timestamp": time.time()
        }
        
        return result, analysis_data
    
    def get_latest_processed_frame(self) -> Optional[np.ndarray]:
        """
        最新の処理済みフレームを取得します
        
        Returns:
            Optional[np.ndarray]: 処理済みフレーム、未処理時はNone
        """
        with self.lock:
            if self.processed_frame is not None:
                return self.processed_frame.copy()
            return None
            
    def get_analysis_data(self) -> Dict[str, Any]:
        """
        最新の分析データを取得します
        
        Returns:
            Dict[str, Any]: 分析データの辞書
        """
        with self.lock:
            return self.analysis_data.copy()
    
    def get_processor_info(self) -> Dict[str, Any]:
        """
        処理情報を取得します
        
        Returns:
            Dict[str, Any]: 処理情報の辞書
        """
        with self.lock:
            return {
                "is_running": self.is_running,
                "processing_fps": self.processing_fps,
                "frames_processed": self.frame_count,
                "last_frame_time": self.processed_frame_time
            }