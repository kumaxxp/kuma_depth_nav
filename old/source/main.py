"""
デプスナビゲーションカメラアプリケーション

USBカメラからの映像を取得し、処理して表示するアプリケーションのメインモジュール。
"""
import os
import sys
import time
import logging
import signal
import argparse
import uvicorn
import cv2
import numpy as np
from typing import Optional

# ライブラリパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 自作ライブラリのインポート
from depth_cap.camera import CameraManager
from depth_cap.processor import ImageProcessor
from depth_cap.web_server import create_app

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# グローバル変数
camera_manager = None
image_processor = None

class DepthImageProcessor(ImageProcessor):
    """デプスカメラ用の拡張画像処理クラス"""
    
    def _process_frame(self, frame: np.ndarray):
        """
        デプス情報を用いたフレーム処理
        
        Args:
            frame: 処理対象のフレーム
            
        Returns:
            Tuple[np.ndarray, Dict]: (処理後フレーム, 分析データ)
        """
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # エッジ検出
        edges = cv2.Canny(gray, 50, 150)
        
        # カラーマップの適用
        colored_edges = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
        
        # 元画像と合成
        alpha = 0.7
        result = cv2.addWeighted(frame, alpha, colored_edges, 1-alpha, 0)
        
        # 情報テキスト追加
        cv2.putText(
            result, 
            f"Depth Analysis: {time.strftime('%H:%M:%S')}",
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8,
            (255, 255, 255),
            2
        )
        
        # 分析情報
        analysis_data = {
            "edge_percentage": np.count_nonzero(edges) / edges.size * 100,
            "timestamp": time.time()
        }
        
        return result, analysis_data

def cleanup():
    """アプリケーションのクリーンアップ処理"""
    global camera_manager, image_processor
    
    logger.info("アプリケーションをシャットダウンしています...")
    
    if image_processor:
        image_processor.stop_processing()
        
    if camera_manager:
        camera_manager.stop_capture()
        camera_manager.release()
        
    logger.info("リソースを解放しました。")

def signal_handler(sig, frame):
    """シグナルハンドラ - 正常終了処理"""
    logger.info(f"シグナル {sig} を受信。シャットダウンします。")
    cleanup()
    sys.exit(0)

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='デプスカメラナビゲーションサーバー')
    parser.add_argument('-p', '--port', type=int, default=8888, help='サーバーポート (デフォルト: 8888)')
    parser.add_argument('-c', '--camera', type=int, default=0, help='カメラインデックス (デフォルト: 0)')
    parser.add_argument('-W', '--width', type=int, default=640, help='解像度幅 (デフォルト: 640)')
    parser.add_argument('-H', '--height', type=int, default=480, help='解像度高 (デフォルト: 480)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='ホストアドレス (デフォルト: 0.0.0.0)')
    parser.add_argument('--no-processing', action='store_true', help='画像処理を無効化')
    return parser.parse_args()

def main():
    """メイン関数"""
    global camera_manager, image_processor
    
    # コマンドライン引数の解析
    args = parse_args()
    
    try:
        # シグナルハンドラの登録
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # カメラマネージャーの初期化
        camera_manager = CameraManager(
            device_index=args.camera,
            width=args.width,
            height=args.height
        )
        
        # カメラの起動
        if not camera_manager.initialize_camera():
            logger.error("カメラ初期化に失敗しました")
            return 1
            
        camera_manager.start_capture()
        
        # 画像処理クラスの初期化
        if not args.no_processing:
            image_processor = DepthImageProcessor(camera_manager)
            image_processor.start_processing()
        
        # FastAPIアプリケーションの作成
        app = create_app(camera_manager, image_processor)
        
        # サーバー起動
        logger.info(f"サーバー起動: http://{args.host}:{args.port}")
        logger.info(f"カメラID: {args.camera}, 解像度: {args.width}x{args.height}")
        logger.info(f"画像処理: {'無効' if args.no_processing else '有効'}")
        logger.info(f"Ctrl+Cで終了")
        
        uvicorn.run(app, host=args.host, port=args.port)
        
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}")
        return 1
    finally:
        cleanup()
        
    return 0

if __name__ == "__main__":
    sys.exit(main())