"""
キャリブレーションファイルがなくてもアプリケーションが正常に動作することを確認するテストスクリプト
"""
import os
import shutil
import time
import unittest
import sys
import cv2
import numpy as np

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestNoCalibration(unittest.TestCase):
    """キャリブレーション無しでもアプリケーションが動作するかテストするクラス"""
    
    def setUp(self):
        """テスト前の準備"""
        # 既存のキャリブレーションファイルをバックアップ
        self.backup_dir = "calibration_backup_" + str(int(time.time()))
        os.makedirs(self.backup_dir, exist_ok=True)
        
        if os.path.exists("calibration_data"):
            for file in os.listdir("calibration_data"):
                if file.endswith(".json") or file.endswith(".pkl"):
                    src = os.path.join("calibration_data", file)
                    dst = os.path.join(self.backup_dir, file)
                    try:
                        shutil.copy2(src, dst)
                        os.remove(src)
                        print(f"バックアップ: {src} -> {dst}")
                    except Exception as e:
                        print(f"バックアップエラー: {e}")
    
    def tearDown(self):
        """テスト後の後片付け"""
        # バックアップからキャリブレーションファイルを復元
        if os.path.exists(self.backup_dir):
            for file in os.listdir(self.backup_dir):
                src = os.path.join(self.backup_dir, file)
                dst = os.path.join("calibration_data", file)
                try:
                    shutil.copy2(src, dst)
                    print(f"復元: {src} -> {dst}")
                except Exception as e:
                    print(f"復元エラー: {e}")
                    
            # バックアップディレクトリを削除
            shutil.rmtree(self.backup_dir)
    
    def test_camera_without_calibration(self):
        """キャリブレーションなしでのカメラ動作テスト"""
        from app.camera.capture import camera
        from app.calibration.calibration import calibration
        
        # カメラが起動しているか確認
        self.assertIsNotNone(camera.cap, "カメラが初期化されていません")
        self.assertTrue(camera.cap.isOpened(), "カメラがオープンされていません")
        
        # キャリブレーションが適用されていないことを確認
        self.assertFalse(camera.use_calibration, "キャリブレーションが適用されています")
        
        # カメラからフレームを取得できるか確認
        ret, frame = camera.read()
        self.assertTrue(ret, "カメラからフレームを取得できません")
        self.assertIsNotNone(frame, "取得したフレームがNoneです")
        
        # 画像の形状と型を確認
        self.assertIsInstance(frame, np.ndarray, "フレームがnumpy配列ではありません")
        self.assertEqual(len(frame.shape), 3, "フレームが3チャンネルではありません")
        
        print("キャリブレーションなしでのカメラテスト成功")
    
    def test_depth_inference_without_calibration(self):
        """キャリブレーションなしでの深度推論テスト"""
        from app.camera.capture import camera
        from app.depth.inference import depth_inference
        
        # カメラからフレームを取得
        ret, frame = camera.read()
        self.assertTrue(ret, "カメラからフレームを取得できません")
        
        # 深度推論を実行
        depth_map = depth_inference.predict(frame)
        
        # 深度マップが生成されるか確認
        self.assertIsNotNone(depth_map, "深度マップがNoneです")
        self.assertIsInstance(depth_map, np.ndarray, "深度マップがnumpy配列ではありません")
        
        print("キャリブレーションなしでの深度推論テスト成功")
    
    def test_visualization_without_calibration(self):
        """キャリブレーションなしでの可視化テスト"""
        from app.visualization.streams import visualization
        
        # 深度ストリームを取得
        depth_stream_gen = visualization.get_depth_stream()
        
        # ジェネレータから最初のフレームを取得
        frame = next(depth_stream_gen)
        
        # バイナリデータが返されることを確認
        self.assertIsNotNone(frame, "深度ストリームがNoneです")
        self.assertTrue(frame.startswith(b'--frame'), "深度ストリームが正しい形式ではありません")
        
        print("キャリブレーションなしでの可視化テスト成功")

if __name__ == "__main__":
    unittest.main()
