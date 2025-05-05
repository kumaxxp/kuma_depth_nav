"""
カメラ管理モジュール

USBカメラの初期化、フレーム取得、リソース管理を担当します。
"""
import cv2
import numpy as np
import time
import logging
import threading
import queue
from typing import Optional, Tuple, List, Dict, Any

# ロガー設定
logger = logging.getLogger(__name__)

class CameraManager:
    """USBカメラ管理クラス"""
    
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        """
        カメラマネージャーの初期化
        
        Args:
            device_index: カメラデバイスのインデックス
            width: 要求する映像の幅
            height: 要求する映像の高さ
        """
        self.device_index = device_index
        self.width = width
        self.height = height
        self.camera = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)  # 最新フレームを保持するキュー
        self.process_queue = queue.Queue(maxsize=10)  # 処理用フレームキュー
        self.frame_count = 0
        self.last_frame_time = 0
        self.fps = 0
        self._lock = threading.RLock()  # カメラアクセス用ロック
        self._thread = None
        
    def initialize_camera(self, max_attempts: int = 3) -> bool:
        """
        USBカメラを初期化します
        
        Args:
            max_attempts: 初期化試行回数
            
        Returns:
            bool: 初期化成功時True
        """
        with self._lock:
            # 既存カメラのクリーンアップ
            if self.camera is not None:
                self.camera.release()
                self.camera = None
                
            for attempt in range(max_attempts):
                try:
                    # Linux環境でV4L2を使用
                    self.camera = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
                    
                    if not self.camera.isOpened():
                        logger.warning(f"カメラ初期化試行 {attempt+1}/{max_attempts} 失敗")
                        if attempt < max_attempts - 1:
                            time.sleep(1.0)
                            continue
                        else:
                            logger.error(f"カメラindex={self.device_index}を開けませんでした")
                            return False
                    
                    # カメラプロパティ設定
                    self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    
                    # 実際に設定された値を確認
                    actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    
                    # テストフレームを読み込んで動作確認
                    success, _ = self.camera.read()
                    if not success:
                        logger.warning("カメラはオープンできましたが、フレーム読込に失敗")
                        self.camera.release()
                        self.camera = None
                        if attempt < max_attempts - 1:
                            time.sleep(1.0)
                            continue
                        else:
                            return False
                            
                    logger.info(f"カメラ初期化成功: {actual_width}x{actual_height}")
                    return True
                    
                except Exception as e:
                    logger.error(f"カメラ初期化エラー: {e}")
                    if self.camera:
                        self.camera.release()
                        self.camera = None
                    if attempt < max_attempts - 1:
                        time.sleep(1.0)
                    else:
                        return False
            
            return False
    
    def start_capture(self) -> bool:
        """
        フレーム取得スレッドを開始します
        
        Returns:
            bool: 開始成功時True
        """
        if self._thread and self._thread.is_alive():
            logger.warning("既にキャプチャスレッドが実行中です")
            return True
            
        if not self.camera or not self.camera.isOpened():
            if not self.initialize_camera():
                return False
        
        # キューをクリア
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.process_queue.empty():
            try:
                self.process_queue.get_nowait()
            except queue.Empty:
                break
        
        self.is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("カメラキャプチャスレッド開始")
        return True
        
    def stop_capture(self):
        """キャプチャスレッドを停止します"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        
    def _capture_loop(self):
        """フレーム取得ループ (内部スレッド用)"""
        consecutive_failures = 0
        max_consecutive_failures = 10
        frame_time_accumulator = 0
        frame_count_for_fps = 0
        fps_update_interval = 1.0  # FPS計算間隔(秒)
        fps_last_update = time.time()
        
        while self.is_running:
            try:
                if not self.camera or not self.camera.isOpened():
                    logger.warning("カメラ接続が失われました。再初期化します")
                    if not self.initialize_camera():
                        time.sleep(1.0)
                        continue
                
                # フレーム取得
                success, frame = self.camera.read()
                current_time = time.time()
                
                if not success or frame is None:
                    consecutive_failures += 1
                    logger.warning(f"フレーム読み取り失敗 ({consecutive_failures}/{max_consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning("連続フレーム取得失敗。カメラを再初期化します")
                        self.initialize_camera()
                        consecutive_failures = 0
                    
                    time.sleep(0.1)
                    continue
                
                # 成功したらカウンタリセット
                consecutive_failures = 0
                self.frame_count += 1
                frame_count_for_fps += 1
                self.last_frame_time = current_time
                
                # FPS計算
                frame_time = current_time - fps_last_update
                if frame_time >= fps_update_interval:
                    self.fps = frame_count_for_fps / frame_time
                    frame_count_for_fps = 0
                    fps_last_update = current_time
                
                # ストリーミング用キューに最新フレームを追加（古いフレームは破棄）
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass
                
                # 処理用キューにもフレームを追加（キューがいっぱいなら追加しない）
                try:
                    if not self.process_queue.full():
                        self.process_queue.put_nowait((frame.copy(), current_time))
                except queue.Full:
                    pass
                    
                # スリープ (フレームレート制御)
                time.sleep(0.005)
                
            except Exception as e:
                logger.error(f"フレームキャプチャ中エラー: {e}")
                time.sleep(0.1)
        
        # スレッド終了時のクリーンアップ
        logger.info("キャプチャスレッド終了")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        最新のフレームを取得します（ブロックしない）
        
        Returns:
            Optional[np.ndarray]: 最新フレーム、取得失敗時はNone
        """
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_frame_for_processing(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        処理用フレームを取得します（ブロックする場合あり）
        
        Returns:
            Optional[Tuple[np.ndarray, float]]: (フレーム, タイムスタンプ)、キューが空ならNone
        """
        try:
            return self.process_queue.get(block=True, timeout=0.5)
        except queue.Empty:
            return None
    
    def release(self):
        """カメラリソースを解放します"""
        self.stop_capture()
        with self._lock:
            if self.camera:
                self.camera.release()
                self.camera = None
                logger.info("カメラリソースを解放しました")
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        現在のカメラ情報を取得します
        
        Returns:
            Dict[str, Any]: カメラ情報の辞書
        """
        info = {
            "status": "disconnected",
            "fps": self.fps,
            "frame_count": self.frame_count
        }
        
        if self.camera and self.camera.isOpened():
            info.update({
                "status": "connected",
                "width": self.camera.get(cv2.CAP_PROP_FRAME_WIDTH),
                "height": self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
                "device_fps": self.camera.get(cv2.CAP_PROP_FPS),
                "format": "MJPG",
                "device_index": self.device_index
            })
            
        return info