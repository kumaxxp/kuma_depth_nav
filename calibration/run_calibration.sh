#!/bin/bash
# キャリブレーション実行用シェルスクリプト

cd "$(dirname "$0")"

# calibration_images ディレクトリがなければ作成
[ -d calibration_images ] || mkdir calibration_images

python3 calibration_app.py
