#!/usr/bin/env python3
"""
深度カメラアプリケーションのエントリポイント
このスクリプトは、リファクタリングされたアプリケーションを起動します
"""
import uvicorn

def main():
    try:
        print("リファクタリングされた深度カメラアプリケーションを起動します...")
        # デバッグログを有効化
        import logging
        logging.basicConfig(level=logging.INFO)
        
        # 必要なディレクトリが存在することを確認
        import os
        for directory in ["calibration_data", "calibration_images", "calibration_results"]:
            os.makedirs(directory, exist_ok=True)
            print(f"ディレクトリを確認: {directory}")
        
        # 正確なパスを確認
        import sys
        print(f"Pythonパス: {sys.path}")
        print(f"Pythonバージョン: {sys.version}")
        
        # アプリケーションを実行
        uvicorn.run("app.main:app", host="0.0.0.0", port=8888, reload=False, log_level="debug")
    except KeyboardInterrupt:
        print("Ctrl+Cが押されました。アプリケーションを終了します。")
    except Exception as e:
        print(f"予期せぬエラーでアプリケーションが終了しました: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
