from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import axengine as axe
import time

app = FastAPI()
MODEL_PATH = '/opt/m5stack/data/depth_anything/compiled.axmodel'

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Depth SLAM Stream</title>
        </head>
        <body>
            <h1>Depth SLAM ナビゲーション</h1>
            <img src="/video" width="768" height="256" />
        </body>
    </html>
    """

def initialize_model(model_path: str):
    return axe.InferenceSession(model_path)

def process_frame(frame: np.ndarray, target_size=(384, 256)) -> np.ndarray:
    resized_frame = cv2.resize(frame, target_size)
    return np.expand_dims(resized_frame[..., ::-1], axis=0)

def create_depth_visualization(depth_map: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
    depth_feature = depth_map.reshape(depth_map.shape[-2:])
    normalized = (depth_feature - depth_feature.min()) / (depth_feature.max() - depth_feature.min())
    depth_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    depth_resized = cv2.resize(depth_colored, (original_frame.shape[1], original_frame.shape[0]))
    return depth_resized


def visualize_navigation(original_frame: np.ndarray, direction_point):
    vis_frame = original_frame.copy()
    h, w = vis_frame.shape[:2]

    # 中心から進行方向へ矢印を描画
    cv2.arrowedLine(vis_frame, (w//2, h), direction_point, (0, 255, 0), 5)
    cv2.circle(vis_frame, direction_point, 8, (255, 0, 0), -1)
    return vis_frame

def detect_navigation_direction(depth_map: np.ndarray, threshold: float = 1.5):
    depth_map = depth_map.squeeze()  # 不要な次元を削除 (1, 1, H, W) -> (H, W)

    free_space = (depth_map > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(free_space)

    if num_labels <= 1:
        return None  # 自由空間なし

    largest_area = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cx, cy = centroids[largest_area]
    return int(cx), int(cy)


def get_video_stream():
    session = initialize_model(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    camera = cv2.VideoCapture(0)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            input_tensor = process_frame(frame)
            depth_output = session.run(None, {input_name: input_tensor})[0]
            depth_output = depth_output.squeeze()  # 追加（重要！）

            # 形状を確認
            print("Depth Output shape:", depth_output.shape)

            depth_vis = create_depth_visualization(depth_output, frame)

            direction_point = detect_navigation_direction(depth_output, threshold=50)  # 整数値に調整
            if direction_point:
                nav_vis = visualize_navigation(frame, direction_point)
            else:
                nav_vis = frame

            # shape確認（結合前）
            print("nav_vis shape:", nav_vis.shape, "depth_vis shape:", depth_vis.shape)

            combined_vis = np.concatenate([nav_vis, depth_vis], axis=1)

            _, buffer = cv2.imencode('.jpg', combined_vis)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            time.sleep(0.005)

    finally:
        camera.release()

@app.get("/video")
async def video_endpoint():
    return StreamingResponse(
        get_video_stream(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7777)
