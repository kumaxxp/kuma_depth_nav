#!/usr/bin/env python3
"""
カメラとキャリブレーションの簡単なテストスクリプト
"""
import cv2
import numpy as np
import os
import sys
import time

# アプリケーションのルートディレクトリをPythonパスに追加
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 必要なモジュールをインポート
from app.calibration.calibration import calibration
from app.camera.capture import camera

def main():
    """カメラキャプチャとキャリブレーションテスト"""
    print("カメラとキャリブレーションのテスト")
    
    # キャリブレーションデータをロード
    calib_file = "calibration_data/calibration.json"
    if os.path.exists(calib_file):
        if calibration.load_calibration(calib_file):
            print(f"キャリブレーションデータを読み込みました: {calib_file}")
            
            # ユーザーに選択を提示
            response = input("キャリブレーションをカメラに適用しますか? (y/n): ")
            if response.lower() == 'y':
                camera.set_calibration(calibration)
                print("キャリブレーションが適用されました")
            else:
                print("キャリブレーションは適用されませんでした")
        else:
            print("キャリブレーションデータを読み込めませんでした")
    else:
        print("キャリブレーションデータが見つかりません")
      # GUIなしテストモード
    print("\nカメラテスト中... Ctrl+Cで終了")
    print("現在のカメラの解像度:", camera.width, "x", camera.height)
    
    try:
        # キャリブレーション情報を表示
        print("\nキャリブレーション情報:")
        if calibration.camera_matrix is not None:
            print("カメラ行列:")
            print(calibration.camera_matrix)
        if calibration.dist_coeffs is not None:
            print("歪み係数:")
            print(calibration.dist_coeffs)
        if hasattr(calibration, "rms_error") and calibration.rms_error is not None:
            print(f"RMS誤差: {calibration.rms_error:.4f}")
            
        # キャリブレーションのON/OFF切り替え
        while True:
            # カメラからフレームを取得
            frame, timestamp = camera.get_frame()
            
            if frame is None:
                print("フレームの取得に失敗しました")
                time.sleep(0.1)
                continue
                
            # 1秒ごとに状態を表示
            time.sleep(1)
            
            # 現在の状態を表示
            calib_status = "ON" if camera.use_calibration else "OFF"
            print(f"現在の状態: タイムスタンプ={timestamp:.3f}s, キャリブレーション={calib_status}")
            
            # ユーザー入力をチェック
            print("\nコマンド: [t] キャリブレーション切り替え, [q] 終了")
            user_input = input("> ")
            
            if user_input.lower() == 'q':
                break
            elif user_input.lower() == 't':
                # キャリブレーションのON/OFFを切り替え
                if camera.use_calibration:
                    camera.disable_calibration()
                    print("キャリブレーション: OFF")
                else:
                    camera.set_calibration(calibration)
                    print("キャリブレーション: ON")
      finally:
        print("\nテスト終了")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nユーザーにより中断されました")
    except Exception as e:
        import traceback
        print(f"エラー: {str(e)}")
        print(traceback.format_exc())
