# キャリブレーション実行用アプリケーション
import cv2
import glob
import os
from calibration.camera_calibration import CameraCalibration

def main():
    # 画像ディレクトリのパス
    image_dir = "calibration_images"
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
    if not image_paths:
        print(f"画像が見つかりません: {image_dir}")
        return

    images = [cv2.imread(p) for p in image_paths]
    calib = CameraCalibration()
    success, rms, mtx, dist = calib.calibrate(images)
    if success:
        print(f"キャリブレーション成功 RMS誤差: {rms}")
        calib.save_calibration()
    else:
        print("キャリブレーション失敗: 十分な画像がありません")

if __name__ == "__main__":
    main()
