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
    
    # ウィンドウ作成
    cv2.namedWindow('Camera Test', cv2.WINDOW_NORMAL)
    
    print("\nカメラテスト中... Escキーで終了")
    print("現在のカメラの解像度:", camera.width, "x", camera.height)
    
    try:
        while True:
            # カメラからフレームを取得
            frame, timestamp = camera.get_frame()
            
            if frame is None:
                print("フレームの取得に失敗しました")
                time.sleep(0.1)
                continue
                
            # タイムスタンプとキャリブレーション状態を表示
            cv2.putText(frame, f"Time: {timestamp:.3f}s", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            calib_status = "ON" if camera.use_calibration else "OFF" 
            cv2.putText(frame, f"Calibration: {calib_status}", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 表示
            cv2.imshow('Camera Test', frame)
            
            # キー入力確認
            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # Escキー
                break
            elif key == 32:  # スペースキー
                # キャリブレーションのON/OFFを切り替え
                if camera.use_calibration:
                    camera.disable_calibration()
                    print("キャリブレーション: OFF")
                else:
                    camera.set_calibration(calibration)
                    print("キャリブレーション: ON")
    
    finally:
        cv2.destroyAllWindows()
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
