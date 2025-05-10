"""
カメラキャリブレーション処理を管理するモジュール
"""
import cv2
import numpy as np
import os
import pickle
import json
import time
from typing import Tuple, List, Dict, Any, Optional

class CameraCalibration:
    """カメラキャリブレーション処理を管理するクラス"""
    
    def __init__(self, chessboard_size=(10, 7), square_size_mm=23.0):
        """カメラキャリブレーションクラスを初期化します

        Args:
            chessboard_size: チェスボードのコーナー数 (width, height)
            square_size_mm: チェスボードの正方形一辺のサイズ (mm単位)
        """
        self.chessboard_size = chessboard_size
        self.square_size_mm = square_size_mm
        
        # キャリブレーション結果
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.rms_error = None
        self.frame_size = None
        
        # コーナー検出のための終了条件
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # オブジェクトポイントの準備（3D空間の点）
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp = self.objp * square_size_mm  # 実際のサイズ（mm）を適用
        
        # 保存ディレクトリの準備
        os.makedirs("calibration_data", exist_ok=True)
    
    def detect_chessboard(self, image: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        """画像からチェスボードのコーナーを検出します

        Args:
            image: 入力画像

        Returns:
            (検出成功フラグ, 元画像に描画した結果, 検出されたコーナー)
        """
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # チェスボードコーナーを検出
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        # コーナーが検出されればサブピクセル精度に改善
        vis_img = image.copy()
        if ret:
            refined_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), self.criteria
            )
            # コーナーを描画
            cv2.drawChessboardCorners(vis_img, self.chessboard_size, refined_corners, ret)
            return True, vis_img, refined_corners
        
        return False, vis_img, None
    
    def calibrate(self, images: List[np.ndarray]) -> Tuple[bool, float, np.ndarray, np.ndarray]:
        """画像セットからカメラキャリブレーションを実行します

        Args:
            images: キャリブレーション用の画像リスト

        Returns:
            (成功フラグ, RMS誤差, カメラ行列, 歪み係数)
        """
        if not images:
            return False, 0, None, None
        
        # フレームサイズを取得（最初の画像から）
        self.frame_size = (images[0].shape[1], images[0].shape[0])
        
        objpoints = []  # 3D空間の点
        imgpoints = []  # 画像上の点
        
        successful_count = 0
        for image in images:
            ret, _, corners = self.detect_chessboard(image)
            if ret:
                objpoints.append(self.objp)
                imgpoints.append(corners)
                successful_count += 1
        
        # キャリブレーション実行（最低5枚の画像が必要）
        if successful_count < 5:
            return False, 0, None, None
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, self.frame_size, None, None
        )
        
        # 再投影誤差を計算
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        rms_error = total_error / len(objpoints)
        
        # 結果を保存
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.rms_error = float(rms_error)
        
        return True, rms_error, mtx, dist
    
    def save_calibration(self, filepath: str = "calibration_data/calibration.json") -> bool:
        """キャリブレーション結果をJSONファイルとして保存します

        Args:
            filepath: 保存先のファイルパス

        Returns:
            保存成功フラグ
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            return False
        
        # 保存用データ辞書の作成
        calibration_data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "frame_size": self.frame_size,
            "chessboard_size": self.chessboard_size,
            "square_size_mm": self.square_size_mm,
            "rms_error": self.rms_error,
            "calibration_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        try:
            # JSONファイルとして保存
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=4)
              # 互換性のためにPickleファイルも保存
            pickle_dir = os.path.dirname(filepath)
            pickle.dump((self.camera_matrix, self.dist_coeffs), 
                      open(f"{pickle_dir}/calibration.pkl", "wb"))
            pickle.dump(self.camera_matrix, open(f"{pickle_dir}/cameraMatrix.pkl", "wb"))
            pickle.dump(self.dist_coeffs, open(f"{pickle_dir}/dist.pkl", "wb"))
            return True
        except Exception as e:
            print(f"保存エラー: {e}")
            return False
    
    def load_calibration(self, filepath: str = "calibration_data/calibration.json") -> bool:
        """保存されたキャリブレーション結果を読み込みます

        Args:
            filepath: JSONファイルのパス

        Returns:
            読み込み成功フラグ
        """
        try:
            # 拡張子でフォーマットを判断
            if filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # 必須フィールド
                if "camera_matrix" not in data or "dist_coeffs" not in data:
                    print(f"警告: キャリブレーションファイルに必須フィールドがありません: {filepath}")
                    return False
                    
                self.camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
                self.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float32)
                
                # オプションフィールド
                self.frame_size = data.get("frame_size")
                self.chessboard_size = data.get("chessboard_size", (10, 7))
                self.square_size_mm = data.get("square_size_mm", 23.0)
                self.rms_error = data.get("rms_error", 0.0)
                
            elif filepath.endswith('.pkl'):
                # 古いPickleフォーマットを処理
                try:
                    self.camera_matrix, self.dist_coeffs = pickle.load(open(filepath, "rb"))
                    self.rms_error = 0.0  # 古いフォーマットではRMS誤差が保存されていない
                except Exception as pkl_err:
                    print(f"Pickleファイル読み込みエラー: {pkl_err}")
                    return False
                
            print(f"キャリブレーションデータを読み込みました: カメラ行列={self.camera_matrix.shape}, 歪み係数={self.dist_coeffs.shape}")
            return True
            
        except Exception as e:
            print(f"キャリブレーション読み込みエラー: {e}")
            return False
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """画像の歪みを補正します

        Args:
            image: 入力画像

        Returns:
            歪み補正された画像
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            return image
        
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        
        # 歪み補正
        dst = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
        
        # ROIで切り取り
        x, y, w, h = roi
        if x != 0 and y != 0 and w > 0 and h > 0:
            dst = dst[y:y+h, x:x+w]
        
        return dst

# シングルトンインスタンス
calibration = CameraCalibration()
