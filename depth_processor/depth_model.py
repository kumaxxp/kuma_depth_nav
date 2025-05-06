"""
深度推定モデル関連の処理
"""

import cv2
import numpy as np
import time
import os

# axengine をインポート
try:
    import axengine as axe
    HAS_AXENGINE = True
except ImportError:
    HAS_AXENGINE = False

# デフォルトのモデルパス
DEFAULT_MODEL_PATH = '/opt/m5stack/data/depth_anything/compiled.axmodel'

class DepthProcessor:
    """深度推定処理クラス"""
    
    def __init__(self, model_path=None):
        """
        初期化
        
        Args:
            model_path: モデルファイルのパス。Noneの場合はデフォルトパスを使用
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.model = self._initialize_model()
        self.input_name = None
        if self.model:
            self.input_name = self.model.get_inputs()[0].name
            
    def _initialize_model(self):
        """モデルを初期化"""
        if not HAS_AXENGINE:
            print("[WARN] axengine not installed. Cannot initialize depth model.")
            return None
            
        try:
            print(f"[INFO] Loading model from {self.model_path}")
            session = axe.InferenceSession(self.model_path)
            print("[INFO] Model loaded successfully")
            return session
        except Exception as e:
            print(f"[ERROR] Failed to initialize depth model: {e}")
            return None
            
    def process_frame(self, frame, target_size=(384, 256)):
        """
        フレームを処理用に前処理
        
        Args:
            frame: 入力画像
            target_size: リサイズ後のサイズ
            
        Returns:
            前処理済みの画像テンソル
        """
        if frame is None:
            raise ValueError("フレームの読み込みに失敗しました")
        
        resized_frame = cv2.resize(frame, target_size)
        # RGB -> BGR の変換とバッチ次元の追加
        return np.expand_dims(resized_frame[..., ::-1], axis=0)
        
    def predict(self, frame):
        """
        深度推定の実行
        
        Args:
            frame: 入力画像
            
        Returns:
            (depth_map, inference_time): 深度マップと推論時間
        """
        if not self.model or not self.input_name:
            raise ValueError("モデルが初期化されていません")
            
        # 画像の前処理
        start_time = time.time()
        input_tensor = self.process_frame(frame)
        
        # 深度推論
        output = self.model.run(None, {self.input_name: input_tensor})
        inference_time = time.time() - start_time
        
        # 深度マップを取得
        depth_map = output[0]
        
        return depth_map, inference_time
        
    def is_available(self):
        """モデルが利用可能かどうかを返す"""
        return self.model is not None

def initialize_depth_model(model_path=None):
    """
    深度推定モデルを初期化する便利関数
    
    Args:
        model_path: モデルファイルのパス
        
    Returns:
        DepthProcessor インスタンス
    """
    return DepthProcessor(model_path)

def convert_to_absolute_depth(depth_map, scaling_factor=15.0):
    """
    相対深度マップを絶対深度マップ（メートル単位）に変換します
    
    Args:
        depth_map (numpy.ndarray): 相対深度マップ（0-1の範囲）
        scaling_factor (float): スケーリング係数（キャリブレーションで決定）
        
    Returns:
        numpy.ndarray: 絶対深度マップ（メートル単位）
    """
    # 深度マップがゼロに近い値を持つ場所を処理（ゼロ除算防止）
    valid_mask = depth_map > 0.01
    
    # 絶対深度マップの初期化
    absolute_depth = np.zeros_like(depth_map)
    
    # スケーリング係数を用いて相対深度から絶対深度を計算
    absolute_depth[valid_mask] = scaling_factor / depth_map[valid_mask]
    
    return absolute_depth