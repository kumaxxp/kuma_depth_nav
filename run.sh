#!/bin/bash
# Depth Navigation Camera起動スクリプト
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR" || exit 1

# 設定ファイルのチェック
if [ ! -f "config.json" ]; then
    echo "[WARNING] config.jsonが見つかりません。デフォルト設定を使用します。"
fi

# CPUガバナーを設定（M5Stackなどハードウェア固有の最適化）
if [ -f "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor" ]; then
    echo "パフォーマンス最適化を適用中..."
    sudo sh -c 'echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor' || \
    echo "パフォーマンス設定に失敗しました（rootが必要です）"
fi

# V4L2パラメータ設定（カメラ最適化）
if command -v v4l2-ctl > /dev/null; then
    echo "カメラパラメータを設定中..."
    v4l2-ctl --set-fmt-video=width=640,height=480,pixelformat=MJPG
    v4l2-ctl --set-parm=30
fi

# Python実行
echo "プログラムを起動中..."
exec python fast_camera_streaming.py "$@"