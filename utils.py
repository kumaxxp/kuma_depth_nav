"""
ユーティリティ関数モジュール
"""

import json
import os
import logging
import sys
from typing import Dict, Any, Optional

def setup_logger(name: str, log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    ロギングシステムをセットアップします
    
    Args:
        name: ロガー名
        log_level: ログレベル（"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"）
        log_file: ログファイルパス（Noneの場合はコンソールのみ）
        
    Returns:
        logging.Logger: 設定されたロガーインスタンス
    """
    # ログレベルの設定
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    level = level_map.get(log_level.upper(), logging.INFO)
    
    # ロガーの作成
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # すでにハンドラーがあれば一旦クリア
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # フォーマッタの作成
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # コンソールハンドラーを追加
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラーを追加（指定があれば）
    if log_file:
        # ディレクトリがなければ作成
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    設定ファイルを読み込みます
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        Dict: 設定データの辞書
    """
    # デフォルト設定
    default_config = {
        "camera": {
            "device_index": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
            "use_v4l2": True
        },
        "depth": {
            "model_type": "DINOv2",
            "width": 640,
            "height": 480,
            "use_gpu": True,
            "visualization_min": 0.1,
            "visualization_max": 0.9
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8888,
            "debug": False
        },
        "logging": {
            "level": "INFO",
            "file": "logs/kuma_depth_nav.log"
        }
    }
    
    # 設定ファイルの読み込み
    config = default_config.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                # 深いマージ
                for key, value in loaded_config.items():
                    if key in config and isinstance(value, dict) and isinstance(config[key], dict):
                        config[key].update(value)
                    else:
                        config[key] = value
        except Exception as e:
            print(f"設定ファイル読み込みエラー: {e}")
    
    # 環境変数から上書き
    # KUMA_CAMERA_DEVICE_INDEX, KUMA_DEPTH_USE_GPU など
    for section in config:
        for key in config[section]:
            env_var = f"KUMA_{section.upper()}_{key.upper()}"
            if env_var in os.environ:
                env_value = os.environ[env_var]
                # 型変換を試みる
                if isinstance(config[section][key], bool):
                    config[section][key] = env_value.lower() in ('true', 'yes', '1')
                elif isinstance(config[section][key], int):
                    try:
                        config[section][key] = int(env_value)
                    except ValueError:
                        pass
                elif isinstance(config[section][key], float):
                    try:
                        config[section][key] = float(env_value)
                    except ValueError:
                        pass
                else:
                    config[section][key] = env_value
    
    return config

def optimize_linux_performance():
    """
    Linux パフォーマンス最適化関数
    実際の最適化は run.sh で行うため、ここでは何もしない
    """
    # run.sh で既に実行されているので何もしない
    pass