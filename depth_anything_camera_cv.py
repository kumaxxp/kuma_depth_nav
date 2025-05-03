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
    if frame is None or frame.size == 0:
        raise ValueError("Empty frame received.")
    resized_frame = cv2.resize(frame, target_size)
    rgb_frame = resized_frame[..., ::-1]  # BGR → RGB
    chw_frame = np.transpose(rgb_frame, (2, 0, 1))  # HWC → CHW
    input_tensor = np.expand_dims(chw_frame, axis=0).astype(np.uint8)  # (1, 3, H, W)
    return input_tensor

def create_depth_visualization(depth_map: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
    depth_feature = depth_map.reshape(depth_map.shape[-2:])
    normalized = (depth_feature - depth_feature.min()) / (depth_feature.max() - depth_feature.min())
    depth_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    depth_resized = cv2.resize(depth_colored, (original_frame.shape[1], original_frame.shape[0]))
    return depth_resized

def detect_navigation_direction(depth_map: np.ndarray, threshold: float = 1.5):
    depth_map = depth_map.squeeze()
    free_space = (depth_map > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(free_space)

    if num_labels <= 1:
        return None  # 自由空間なし

    largest_area = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cx, cy = centroids[largest_area]
    return int(cx), int(cy)

def visualize_navigation(original_frame: np.ndarray, direction_point):
    vis_frame = original_frame.copy()
    h, w = vis_frame.shape[:2]

    cx, cy = direction_point
    cx = np.clip(cx, 0, w - 1)
    cy = np.clip(cy, 0, h - 1)

    center = (w // 2, h)
    cv2.arrowedLine(vis_frame, center, (cx, cy), (0, 255, 0), 3, tipLength=0.2)
    cv2.circle(vis_frame, (cx, cy), 6, (0, 0, 255), -1)
    cv2.putText(vis_frame, f\"Direction: ({cx}, {cy})\", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis_frame


def get_video_stream():
    session = initialize_model(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    camera = cv2.VideoCapture(0)

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("Failed to read frame.")
                continue  # skip this frame

            try:
                input_tensor = process_frame(frame)
                depth_output = session.run(None, {input_name: input_tensor})[0]
                depth_output = depth_output.squeeze()
            except Exception as e:
                print(f"[ERROR] Processing frame failed: {e}")
                continue  # skip to next frame

            # 残りの可視化・描画処理はそのまま

            depth_vis = create_depth_visualization(depth_output, frame)

            direction_point = detect_navigation_direction(depth_output)
            if direction_point:
                nav_vis = visualize_navigation(frame, direction_point)
            else:
                nav_vis = frame

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
