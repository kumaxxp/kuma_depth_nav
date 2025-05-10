"""
FastAPIのルート定義を管理するモジュール
"""
from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from ..camera.capture import camera
from ..visualization.streams import visualization
from ..utils.stats import stats
from .templates import get_index_html

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def index():
    """インデックスページを返す"""
    return get_index_html()

@router.get("/video")
async def video():
    """カメラビデオストリームを返す"""
    return StreamingResponse(camera.get_camera_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/depth_video")
async def depth_video():
    """深度マップストリームを返す"""
    return StreamingResponse(visualization.get_depth_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/depth_grid")
async def depth_grid():
    """深度グリッドストリームを返す"""
    return StreamingResponse(visualization.get_depth_grid_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/stats")
async def get_stats():
    """統計情報を取得するAPIエンドポイント"""
    # カメラからフレームタイムスタンプを取得して統計を計算
    return stats.get_stats_data(camera.frame_timestamp)

@router.get("/calibration_status")
async def calibration_status():
    """キャリブレーション状態を取得するAPIエンドポイント"""
    from ..calibration.calibration import calibration
    
    status = {
        "calibrated": calibration.camera_matrix is not None,
        "applied": camera.use_calibration,
        "rms_error": calibration.rms_error if calibration.rms_error else 0,
        "calibration_time": None
    }
    
    # キャリブレーションデータがあれば追加情報を設定
    if calibration.camera_matrix is not None:
        import os
        calib_file = "calibration_data/calibration.json"
        if os.path.exists(calib_file):
            import json
            try:
                with open(calib_file, 'r') as f:
                    data = json.load(f)
                    status["calibration_time"] = data.get("calibration_time")
            except:
                pass
    
    return status

def setup_routes(app: FastAPI):
    """FastAPIアプリにルートを設定する"""
    app.include_router(router)
