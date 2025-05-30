"""
深度推定モデル関連の処理
"""

import cv2
import numpy as np
import time
import os
import logging

# ロガーの取得
logger = logging.getLogger("kuma_depth_nav.depth_model")

# axengine をインポート
try:
    import axengine as axe
    HAS_AXENGINE = True
except ImportError:
    HAS_AXENGINE = False
    logger.warning("axengine is not installed. Running in basic mode without depth estimation.")

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
            logger.warning("axengine not installed. Cannot initialize depth model.")
            return None
            
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # ファイルの存在確認
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return None
                 
            # セッション作成 - エラー処理を強化
            try:
                session = axe.InferenceSession(self.model_path)
                logger.info("Model session created successfully")
                
                # モデル入力情報の取得は成功した場合のみ
                try:
                    inputs = session.get_inputs()
                    if inputs and len(inputs) > 0:
                        logger.info(f"Model has {len(inputs)} inputs")
                        logger.info(f"First input name: {inputs[0].name}")
                except Exception as e:
                    logger.warning(f"Could not get input details: {e}, but continuing")
                    
                # エラーがあっても、セッションは返す（推論は可能な場合があるため）
                logger.info("Model loaded successfully")
                return session
            except Exception as e:
                logger.error(f"Failed to create inference session: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
            
        except Exception as e:
            logger.error(f"Failed to initialize depth model: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
        """深度推定を実行"""
        if not self.is_available():
            # ダミーデータを使用
            logger.info("Using dummy depth data (model not available)")
            return self._generate_dummy_depth(frame), 0.01
            
        start_time = time.time()
        
        try:
            # フレーム前処理
            input_tensor = self.process_frame(frame)
            
            # 入力テンソルの形状ログ出力
            logger.debug(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
            
            # 推論実行
            outputs = self.model.run(None, {self.input_name: input_tensor})
            if outputs is None or len(outputs) == 0:
                logger.error("Inference returned empty outputs")
                return self._generate_dummy_depth(frame), time.time() - start_time
                
            depth = outputs[0]
            logger.debug(f"Raw depth output shape: {depth.shape}, size: {depth.size}")
            
            # 後処理 - 形状の明示的指定
            try:
                # 深度データを整形
                if depth.size == 384*256:  # 期待サイズの場合
                    depth_map = depth.reshape(1, 256, 384, 1)
                else:
                    # 形状が異なる場合、ログに出力して可能な限り調整
                    logger.warning(f"Unexpected depth size: {depth.size}, expected: {384*256}")
                    h = int(np.sqrt(depth.size / 384))
                    w = 384 if h > 0 else int(np.sqrt(depth.size))
                    h = h or 256
                    depth_map = depth.reshape(1, h, w, 1)
                    
                depth_map = np.ascontiguousarray(depth_map)
                
                # 正規化して値の範囲をチェック
                min_val = np.min(depth_map)
                max_val = np.max(depth_map)
                logger.debug(f"Depth range: {min_val:.4f} to {max_val:.4f}")
                
                # 異常値チェック（NaNやInf）
                if np.isnan(depth_map).any() or np.isinf(depth_map).any():
                    logger.warning("Depth map contains NaN or Inf values")
                    depth_map = np.nan_to_num(depth_map, nan=0.5, posinf=1.0, neginf=0.0)
                
                inference_time = time.time() - start_time
                return depth_map, inference_time
                
            except Exception as e:
                logger.error(f"Error in depth post-processing: {e}")
                return self._generate_dummy_depth(frame), time.time() - start_time
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._generate_dummy_depth(frame), time.time() - start_time
    
    def _generate_dummy_depth(self, frame):
        """テスト用のダミー深度マップを生成"""
        if frame is None:
            h, w = 480, 640
        else:
            h, w = frame.shape[:2]
            
        # モデルの出力サイズに合わせる
        output_h, output_w = 256, 384
        
        # グラデーションパターン: 下に行くほど深度値が大きくなる（近い）
        dummy_depth = np.zeros((1, output_h, output_w, 1), dtype=np.float32)
        for y in range(output_h):
            value = 0.1 + 0.8 * (y / output_h)
            dummy_depth[0, y, :, 0] = value
            
        logger.debug(f"Generated dummy depth map: {dummy_depth.shape}")
        return dummy_depth
    
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