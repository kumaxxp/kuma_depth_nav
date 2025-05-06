#!/bin/bash
# カメラストリーミングサーバー起動スクリプト

echo "Depth Navigation System を起動しています..."

# 仮想環境がある場合はアクティベート
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# サーバー起動
python fast_camera_streaming.py "$@"