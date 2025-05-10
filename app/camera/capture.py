"""
カメラキャプチャ関連の機能を提供するモジュール
"""
import cv2
import time
import threading
from typing import Optional, Tuple
from ..utils.stats import stats

class CameraCapture:
    """カメラキャプチャを管理するクラス"""
    
    def __init__(self, camera_id=0, width=320, height=240):
        """
        Args:
            camera_id: カメラID（デフォルト：0）
            width: キャプチャ幅（デフォルト：320）
            height: キャプチャ高さ（デフォルト：240）
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.latest_frame = None
        self.frame_timestamp = 0
        self.lock = threading.Lock()
        
        # カメラキャリブレーション関連
        self.calibration = None
        self.use_calibration = False
        
        # カメラを初期化
        self._initialize_camera()
        
        # キャプチャスレッド開始
        self.running = True
        self.capture_thread = threading.Thread(target=self._camera_capture_loop, daemon=True)
        self.capture_thread.start()
    
    def _initialize_camera(self):
        """カメラを初期化する"""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファサイズを最小に
        self.cap.set(cv2.CAP_PROP_FPS, 30)        # カメラのFPS設定
        
        # カメラの接続状態を確認
        if not self.cap.isOpened():
            raise RuntimeError("エラー: カメラに接続できません")
        else:
            print(f"カメラ接続成功: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
      def set_calibration(self, calibration):
        """カメラキャリブレーションを設定する
        
        Args:
            calibration: カメラキャリブレーションインスタンス
        """
        # キャリブレーションデータの検証
        if calibration is None:
            print("警告: キャリブレーションインスタンスがNoneです。キャリブレーションは適用されません。")
            return
            
        if calibration.camera_matrix is None or calibration.dist_coeffs is None:
            print("警告: キャリブレーションデータが無効です。キャリブレーションは適用されません。")
            return
            
        self.calibration = calibration
        self.use_calibration = True
        print("カメラキャリブレーションが適用されました")
    
    def disable_calibration(self):
        """カメラキャリブレーションを無効化する"""
        self.use_calibration = False
        print("カメラキャリブレーションが無効化されました")
    
    def _camera_capture_loop(self):
        """カメラキャプチャスレッドのメインループ"""
        consecutive_errors = 0
        max_errors = 5
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    # キャリブレーションが有効な場合は補正を適用
                    if self.use_calibration and self.calibration is not None:
                        frame = self.calibration.undistort_image(frame)
                    
                    with self.lock:
                        self.latest_frame = frame.copy()
                        self.frame_timestamp = time.time()
                    consecutive_errors = 0  # エラーカウンタをリセット
                else:
                    consecutive_errors += 1
                    print(f"カメラ読み取りエラー ({consecutive_errors}/{max_errors})")
                    
                    if consecutive_errors >= max_errors:
                        print("カメラをリセットします...")
                        self.cap.release()
                        time.sleep(1.0)
                        self._initialize_camera()
                        consecutive_errors = 0
            except Exception as e:
                print(f"カメラ例外: {e}")
                time.sleep(0.5)
                
            time.sleep(0.05)  # 20FPSを維持
    
    def get_frame(self):
        """最新のフレームを取得する
        
        Returns:
            tuple: (フレーム, タイムスタンプ)。フレームがない場合は(None, 0)。
        """
        with self.lock:
            if self.latest_frame is None:
                return None, 0
            return self.latest_frame.copy(), self.frame_timestamp
    
    def release(self):
        """カメラリソースを解放する"""
        self.running = False
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        if self.cap is not None:
            self.cap.release()
            print("カメラリソースを解放しました")
    
    def get_camera_stream(self):
        """カメラストリームのジェネレーター（FastAPI用）"""
        # ストリーム開始時のタイムスタンプをリセット
        stats.last_frame_times["camera"] = 0
        stats.fps_stats["camera"].clear()  # FPS統計をクリア
        first_frame = True  # 最初のフレームかどうかを追跡
        
        while True:
            # 最新のカメラフレームを取得
            frame, _ = self.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # FPS計算 - 改善版
            now = time.time()
            if first_frame:
                # 最初のフレームはFPS計算をスキップ、タイムスタンプのみ記録
                first_frame = False
            elif (now - stats.last_frame_times["camera"]) < 0.5:  # 0.5秒以内の正常な間隔
                # 正常な間隔の場合のみFPSを計算
                fps = 1.0 / (now - stats.last_frame_times["camera"])
                # 異常値フィルタリング (FPSが200を超える値はエラーと見なす)
                if fps < 200:  
                    stats.fps_stats["camera"].append(fps)
            
            # 現在時刻を常に記録
            stats.last_frame_times["camera"] = now
    
            # 画面上にFPS表示
            if len(stats.fps_stats["camera"]) > 0:
                # 中央値を使用 (平均値より外れ値の影響を受けにくい)
                camera_fps_values = list(stats.fps_stats["camera"])
                camera_fps_values.sort()
                median_fps = camera_fps_values[len(camera_fps_values) // 2]
                cv2.putText(frame, f"FPS: {median_fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
            # エンコーディング
            start_time = time.perf_counter()
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            stats.encoding_times.append(time.perf_counter() - start_time)
            if not ret:
                continue
    
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)  # 20FPSに制限

# グローバルインスタンス
camera = CameraCapture()
