#!/usr/bin/env python3
"""
カメラキャリブレーションアプリを実行するためのスクリプト
"""
import cv2
import numpy as np
import argparse
import os
import sys
import time

# アプリケーションのルートディレクトリをPythonパスに追加
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 必要なモジュールをインポート
from app.calibration.calibration import calibration
from app.calibration.app_ui import calibration_app
from app.camera.capture import camera

def display_usage():
    """使用方法を表示する"""
    print("カメラキャリブレーションツール")
    print("使用方法:")
    print("  load: 既存の画像からキャリブレーションを実行")
    print("  capture: 新しい画像をキャプチャしてキャリブレーション")
    print("  view: キャリブレーション効果を確認")
    print("  apply: キャリブレーションをカメラに適用")
    print("Escキーで終了")

def main():
    parser = argparse.ArgumentParser(description='カメラキャリブレーションツール')
    parser.add_argument('--mode', choices=['load', 'capture', 'view', 'apply'], 
                        default='view', help='実行モード')
    parser.add_argument('--images', type=int, default=10, 
                        help='キャプチャする画像の数 (captureモード時)')
    parser.add_argument('--delay', type=int, default=2, 
                        help='キャプチャ間の遅延（秒） (captureモード時)')
    args = parser.parse_args()
    
    # 既存のキャリブレーションデータを読み込む
    calib_file = "calibration_data/calibration.json"
    if os.path.exists(calib_file):
        calibration.load_calibration(calib_file)
        print(f"キャリブレーションデータを読み込みました: {calib_file}")
        calibration_app.calibration_success = True
    
    # モードに応じた処理を実行
    if args.mode == 'load':
        # 既存の画像を読み込んでキャリブレーション
        if calibration_app.load_images_from_folder():
            calibration_app.run_calibration()
            print(calibration_app.status_text)
    
    elif args.mode == 'capture':
        # 新しい画像をキャプチャしてキャリブレーション
        print(f"{args.images}枚の画像をキャプチャします。{args.delay}秒間隔で撮影します。")
        print("各ポーズでチェスボードが明確に見えるようにしてください。")
        
        # カウントダウン
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        # キャプチャ開始
        if calibration_app.capture_images(args.images, args.delay):
            calibration_app.run_calibration()
        
        print(calibration_app.status_text)
    
    elif args.mode == 'apply':
        # キャリブレーションをカメラに適用
        if calibration_app.apply_calibration_to_camera():
            print("キャリブレーションがカメラに適用されました")
        else:
            print("キャリブレーションの適用に失敗しました")
      else:  # view mode
        # キャリブレーション効果を確認するためのライブビュー
        if not calibration_app.calibration_success:
            print("キャリブレーションデータが見つかりません。")
            print("まず 'load' または 'capture' モードでキャリブレーションを実行してください。")
            sys.exit(1)
        
        # キャリブレーションレベルを切り替えるフラグ
        calibration_level = 1  # 0: 補正なし, 1: 補正あり, 2: 比較ビュー
        
        # ウィンドウ作成
        cv2.namedWindow('Calibration View', cv2.WINDOW_NORMAL)
        
        print("ライブビューモード（スペースキーで補正モード切替、Escキーで終了）")
        print("補正モード: 0=補正なし, 1=補正あり, 2=比較ビュー")
          while True:
            # カメラからフレームを取得
            frame, _ = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # キャリブレーションモードに応じた表示
            if calibration_level == 0:  # 補正なし
                display = frame.copy()
                cv2.putText(display, "Original (No Calibration)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            elif calibration_level == 1:  # 補正あり
                display = calibration.undistort_image(frame)
                cv2.putText(display, "Calibrated", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            else:  # 比較ビュー
                # 補正画像を準備
                corrected = calibration.undistort_image(frame)
                
                # 画像を並べて表示
                h, w = frame.shape[:2]
                display = np.zeros((h, w * 2, 3), dtype=np.uint8)
                display[:, :w] = frame
                display[:, w:] = corrected
                
                # テキスト追加
                cv2.putText(display, "Original", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, "Calibrated", (w + 10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # 中央線を描画
                cv2.line(display, (w, 0), (w, h), (0, 255, 255), 2)
            
            # 表示
            cv2.imshow('Calibration View', display)
            
            # キー入力確認
            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # Escキー
                break
            elif key == 32:  # スペースキー
                calibration_level = (calibration_level + 1) % 3
                print(f"補正モード: {calibration_level}")
        
        # ウィンドウ閉じる
        cv2.destroyAllWindows()
    
    print("終了しました")

if __name__ == "__main__":
    try:
        display_usage()
        main()
    except KeyboardInterrupt:
        print("\nユーザーにより中断されました")
    except Exception as e:
        import traceback
        print(f"エラー: {str(e)}")
        print(traceback.format_exc())
