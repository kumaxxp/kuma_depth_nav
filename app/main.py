"""
アプリケーションのエントリーポイント
"""
import os
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .camera.capture import camera
from .depth.inference import depth_inference
from .web.routes import setup_routes
from .calibration.calibration import calibration

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""    # 起動時の処理
    print("アプリケーション起動: カメラとスレッドを初期化します")
    # カメラと深度推論は既に初期化済み
    
    # 必要なディレクトリを作成
    for directory in ["calibration_data", "calibration_images", "calibration_results"]:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"ディレクトリを確認: {directory}")
        except Exception as e:
            print(f"ディレクトリ作成中のエラー ({directory}): {e}")
    
    # キャリブレーションデータがあれば読み込む
    try:
        calib_file = "calibration_data/calibration.json"
        if os.path.exists(calib_file):
            if calibration.load_calibration(calib_file):
                print(f"キャリブレーションデータを読み込みました: {calib_file}")
                # カメラにキャリブレーション適用
                try:
                    camera.set_calibration(calibration)
                    print("キャリブレーションをカメラに適用しました")
                except Exception as e:
                    print(f"キャリブレーション適用中のエラー: {e}")
                    print("キャリブレーションなしで続行します")
            else:
                print(f"警告: キャリブレーションデータの読み込みに失敗しました: {calib_file}")
                print("キャリブレーションなしで続行します")
        else:
            print("情報: キャリブレーションデータが見つかりません。キャリブレーションなしで続行します")
            print("キャリブレーションを実行するには、run_camera_calibration.py を使用してください")
    except Exception as e:
        print(f"キャリブレーション初期化中のエラー: {e}")
        print("キャリブレーションなしで続行します")
    
    yield  # アプリケーション実行中
    
    # 終了時の処理
    print("アプリケーション終了: リソースを解放します")
    try:
        # カメラのクリーンアップ
        camera.release()
    except Exception as e:
        print(f"終了処理中のエラー: {e}")

# FastAPIアプリケーションを初期化
app = FastAPI(lifespan=lifespan)

# ルートを設定
setup_routes(app)

# グローバルな例外ハンドラーを追加
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """すべての未処理例外をキャッチするグローバル例外ハンドラー"""
    error_msg = f"予期せぬエラーが発生しました: {str(exc)}"
    print(f"[エラー] {error_msg}")
    print(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={"detail": error_msg}
    )
