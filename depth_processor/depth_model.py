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
                
            # セッション作成
            session = axe.InferenceSession(self.model_path)

            # モデル情報の出力を安全に行う
            try:
                inputs = session.get_inputs()
                logger.info(f"Model inputs: {[x.name for x in inputs]}")
                if inputs:
                    logger.info(f"Input name: {inputs[0].name}")
                    # shape と type へのアクセスで例外が発生する可能性があるためtry-except内に移動
                    try:
                        if hasattr(inputs[0], 'shape'):
                            logger.info(f"Input shape: {inputs[0].shape}")
                        if hasattr(inputs[0], 'type'):
                            logger.info(f"Input type: {inputs[0].type}")
                    except Exception as e:
                        logger.warning(f"Could not get input details: {e}")
                
                outputs = session.get_outputs()
                logger.info(f"Model outputs: {[x.name for x in outputs]}")
                if outputs:
                    logger.info(f"Output name: {outputs[0].name}")
                    try:
                        if hasattr(outputs[0], 'shape'):
                            logger.info(f"Output shape: {outputs[0].shape}")
                        if hasattr(outputs[0], 'type'):
                            logger.info(f"Output type: {outputs[0].type}")
                    except Exception as e:
                        logger.warning(f"Could not get output details: {e}")
            except Exception as e:
                logger.warning(f"Error getting model details: {e}")
                # ここでは例外を握りつぶす（モデルのメタデータを取得できなくても実行は続行）

            logger.info("Model loaded successfully")
            return session
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
            # テスト用のダミー深度マップを生成
            if frame is not None:
                h, w = frame.shape[:2]
                # 距離に応じた深度を生成（上部ほど遠く、下部ほど近く）
                dummy_depth = np.zeros((1, 256, 384, 1), dtype=np.float32)
                for y in range(256):
                    # 下部ほど近く（値が大きい）
                    value = 0.1 + 0.9 * (y / 256)
                    dummy_depth[0, y, :, 0] = value
                logger.info("Generated dummy depth map for testing")
                return dummy_depth, 0.01
            return None, 0.0
            
        start_time = time.time()
        
        try:
            # フレーム前処理
            input_tensor = self.process_frame(frame)
            logger.info(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
            logger.info(f"Input tensor range: min={np.min(input_tensor)}, max={np.max(input_tensor)}")
            
            # 推論実行
            logger.info(f"Starting inference with input name: {self.input_name}")
            depth = self.model.run(None, {self.input_name: input_tensor})[0]
            logger.info(f"Inference completed successfully")
            
            # 後処理 - reshape操作を修正
            # オリジナルのモデル出力形状を確認
            logger.info(f"Raw depth output shape: {depth.shape}, size: {depth.size}")
            logger.info(f"Depth output range: min={np.min(depth)}, max={np.max(depth)}")
            
            # 適切な形状に変換
            # 一般的な単眼深度推定モデルは(H,W)または(1,H,W)の形状で出力する
            if len(depth.shape) == 2:
                # すでに2次元の場合
                depth_map = depth.reshape(1, depth.shape[0], depth.shape[1], 1)
            elif len(depth.shape) == 3 and depth.shape[0] == 1:
                # バッチ次元がある3次元の場合
                depth_map = depth.reshape(1, depth.shape[1], depth.shape[2], 1)
            else:
                # それ以外の場合は推論された深度マップの形状を元にreshape
                h = int(np.sqrt(depth.size / 256))
                w = int(np.sqrt(depth.size / 384))
                if h * w * 384 == depth.size:
                    depth_map = depth.reshape(1, h, w, 1)
                else:
                    # 384×256に強制変換
                    depth_map = depth.reshape(1, 256, 384, 1)
            
            logger.info(f"Reshaped depth map shape: {depth_map.shape}")
            
            depth_map = np.ascontiguousarray(depth_map)
            
            inference_time = time.time() - start_time
            return depth_map, inference_time
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            logger.error(f"推論エラー詳細: {type(e).__name__}, 深度配列サイズ: {depth.size if 'depth' in locals() else 'unknown'}")
            import traceback
            logger.error(traceback.format_exc())
            return None, time.time() - start_time
        
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