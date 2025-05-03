from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import axengine as axe
import time
import traceback

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
    session = axe.InferenceSession(model_path)
    input_info = session.get_inputs()[0]
    print("[INFO] Model Input Shape:", input_info.shape)
    print("[INFO] Model Input Dtype:", input_info.dtype)
    return session

def process_frame(frame: np.ndarray, target_size=(384, 256)) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ValueError("Empty frame received.")

    resized = cv2.resize(frame, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    if MODEL_PATH.endswith(".axmodel"):
        tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)  # NHWC uint8 for axmodel
        if tensor.nbytes % np.dtype(np.uint8).itemsize != 0:
            raise ValueError("[ERROR] Tensor buffer size not aligned with dtype")
        return tensor
    rgb = rgb.astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0)

def create_depth_visualization(depth_map: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
    depth_feature = depth_map.reshape(depth_map.shape[-2:])
    normalized = (depth_feature - depth_feature.min()) / (depth_feature.max() - depth_feature.min() + 1e-6)
    depth_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    depth_resized = cv2.resize(depth_colored, (original_frame.shape[1], original_frame.shape[0]))
    return depth_resized

def detect_navigation_direction(depth_map: np.ndarray, threshold: float = 1.5):
    depth_map = depth_map.squeeze()
    free_space = (depth_map > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(free_space)

    if num_labels <= 1:
        return None

    largest_area = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cx, cy = centroids[largest_area]
    return int(cx), int(cy)

def visualize_navigation(original_frame: np.ndarray, direction_point):
    vis_frame = original_frame.copy()
    h, w = vis_frame.shape[:2]

    if direction_point is None:
        return vis_frame

    cx, cy = direction_point
    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))

    center = (w // 2, h - 1)
    try:
        cv2.arrowedLine(vis_frame, center, (cx, cy), (0, 255, 0), 3, tipLength=0.2)
        cv2.circle(vis_frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(vis_frame, f"Direction: ({cx}, {cy})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    except Exception as e:
        print(f"[WARN] Visualization failed: {e}")

    return vis_frame

def get_video_stream():
    session = initialize_model(MODEL_PATH)
    input_info = session.get_inputs()[0]
    input_name = input_info.name

    camera = cv2.VideoCapture(0)

    try:
        while True:
            if not camera.isOpened():
                print("[WARN] Camera not open. Reinitializing...")
                camera.release()
                time.sleep(0.5)
                camera = cv2.VideoCapture(0)
                continue

            success, frame = camera.read()
            if not success or frame is None:
                print("[WARN] Frame read failed")
                continue

            try:
                input_tensor = process_frame(frame)
                if input_tensor is None or input_tensor.size == 0:
                    print("[WARN] Empty input_tensor")
                    continue
                print(f"[INFO] Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

                try:
                    depth_output_raw = session.run(None, {input_name: input_tensor})
                except Exception as e:
                    print("[ERROR] AX run failed:", e)
                    traceback.print_exc()
                    continue

                if not depth_output_raw or not isinstance(depth_output_raw[0], np.ndarray):
                    print("[ERROR] Invalid output from model")
                    continue
                depth_output = depth_output_raw[0].squeeze()

                if depth_output.ndim != 2:
                    print(f"[ERROR] Unexpected depth output shape: {depth_output.shape}")
                    continue

            except Exception as e:
                print(f"[ERROR] Processing frame failed: {e}")
                traceback.print_exc()
                continue

            depth_vis = create_depth_visualization(depth_output, frame)
            direction_point = detect_navigation_direction(depth_output)
            nav_vis = visualize_navigation(frame, direction_point)

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