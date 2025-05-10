"""
カメラキャリブレーションUIを管理するモジュール
"""
import cv2
import numpy as np
import os
import glob
import time
import threading
from typing import List, Tuple, Optional

from .calibration import CameraCalibration
from ..camera.capture import camera

class CalibrationApp:
    """カメラキャリブレーションUIアプリケーション"""
    
    def __init__(self, calibration: CameraCalibration):
        """キャリブレーションアプリケーションの初期化
        
        Args:
            calibration: キャリブレーションインスタンス
        """
        self.calibration = calibration
        self.images_folder = "calibration_images"
        self.result_folder = "calibration_results"
        
        # 保存ディレクトリの準備
        os.makedirs(self.images_folder, exist_ok=True)
        os.makedirs(self.result_folder, exist_ok=True)
        
        # 状態変数
        self.is_capturing = False
        self.is_calibrating = False
        self.captured_images: List[np.ndarray] = []
        self.calibration_success = False
        self.comparison_view = None
        
        # UIテキスト
        self.status_text = "準備完了"
        
    def capture_images(self, num_images: int = 10, delay_seconds: int = 2) -> bool:
        """一連のキャリブレーション用画像をキャプチャします
        
        Args:
            num_images: キャプチャする画像数
            delay_seconds: キャプチャ間の遅延（秒）
            
        Returns:
            キャプチャ成功フラグ
        """
        if self.is_capturing:
            return False
            
        self.is_capturing = True
        self.captured_images = []
        
        try:
            for i in range(num_images):
                self.status_text = f"画像キャプチャ中... ({i+1}/{num_images})"
                
                # カメラからフレームを取得
                frame, _ = camera.get_frame()
                if frame is None:
                    self.status_text = "カメラからのキャプチャに失敗しました"
                    continue
                    
                # 画像を保存
                filename = f"{self.images_folder}/calibration_{time.strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
                cv2.imwrite(filename, frame)
                self.captured_images.append(frame)
                
                # 遅延
                time.sleep(delay_seconds)
            
            self.status_text = f"{len(self.captured_images)}枚の画像をキャプチャしました"
            return True
            
        except Exception as e:
            self.status_text = f"キャプチャエラー: {str(e)}"
            return False
        finally:
            self.is_capturing = False
    
    def load_images_from_folder(self) -> bool:
        """フォルダから画像を読み込みます
        
        Returns:
            読み込み成功フラグ
        """
        image_paths = glob.glob(os.path.join(self.images_folder, '*.jpg'))
        
        if not image_paths:
            self.status_text = f"画像が見つかりません: {self.images_folder}"
            return False
            
        self.captured_images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                self.captured_images.append(img)
                
        self.status_text = f"{len(self.captured_images)}枚の画像を読み込みました"
        return len(self.captured_images) > 0
    
    def run_calibration(self) -> bool:
        """キャリブレーション処理を実行します
        
        Returns:
            キャリブレーション成功フラグ
        """
        if self.is_calibrating or not self.captured_images:
            return False
            
        self.is_calibrating = True
        self.status_text = "キャリブレーション処理中..."
        
        try:
            # キャリブレーション実行
            success, rms, mtx, dist = self.calibration.calibrate(self.captured_images)
            
            if success:
                self.calibration_success = True
                self.status_text = f"キャリブレーション成功! RMS誤差: {rms:.4f}"
                
                # 結果を保存
                save_path = os.path.join(self.result_folder, "calibration.json")
                self.calibration.save_calibration(save_path)
                
                # 補正結果のプレビューを作成
                self.create_comparison_view()
                
                return True
            else:
                self.status_text = "キャリブレーション失敗: 十分な画像がないか、チェスボードが検出できません"
                return False
                
        except Exception as e:
            self.status_text = f"キャリブレーションエラー: {str(e)}"
            return False
        finally:
            self.is_calibrating = False
    
    def create_comparison_view(self) -> bool:
        """補正前後の比較ビューを作成します
        
        Returns:
            作成成功フラグ
        """
        if not self.calibration_success or not self.captured_images:
            return False
            
        try:
            # テスト用画像（最後のキャプチャ画像を使用）
            test_image = self.captured_images[-1].copy()
            
            # 補正後の画像
            corrected_image = self.calibration.undistort_image(test_image)
            
            # 2つの画像を並べて表示するために結合
            h, w = test_image.shape[:2]
            comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
            comparison[:, :w] = test_image
            comparison[:, w:] = corrected_image
            
            # テキスト追加
            cv2.putText(comparison, "元画像", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(comparison, "補正後", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 結果を保存
            self.comparison_view = comparison
            cv2.imwrite(os.path.join(self.result_folder, "comparison.jpg"), comparison)
            
            return True
            
        except Exception as e:
            print(f"比較ビュー作成エラー: {str(e)}")
            return False
    def apply_calibration_to_camera(self) -> bool:
        """キャリブレーション結果をカメラキャプチャに適用します
        
        Returns:
            適用成功フラグ
        """
        if not self.calibration_success:
            self.status_text = "キャリブレーションが完了していません"
            print("警告: キャリブレーションが完了していないため適用できません")
            return False
            
        if self.calibration.camera_matrix is None or self.calibration.dist_coeffs is None:
            self.status_text = "有効なキャリブレーションデータがありません"
            print("警告: キャリブレーションデータが無効です")
            return False
            
        try:
            # カメラ補正用のグローバル変数を設定
            from ..camera.capture import camera
            camera.set_calibration(self.calibration)
            self.status_text = "キャリブレーションがカメラに適用されました"
            print("情報: キャリブレーションがカメラに適用されました")
            return True
        except Exception as e:
            self.status_text = f"キャリブレーション適用エラー: {str(e)}"
            print(f"エラー: キャリブレーション適用中に例外が発生しました: {str(e)}")
            return False

# キャリブレーションアプリケーションを作成
from .calibration import calibration
calibration_app = CalibrationApp(calibration)
