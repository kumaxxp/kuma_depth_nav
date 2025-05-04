from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import time
import axengine as axe

app = FastAPI()
MODEL_PATH = '/opt/m5stack/data/depth_anything/compiled.axmodel'

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>USBカメラ ストリーム</title>
        </head>
        <body>
            <h1>USBカメラの映像を表示中</h1>
            <img src="/video" width="1280" height="480" />
        </body>
    </html>
    """

def initialize_camera(index=0, width=640, height=480):
    cam = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cam

def initialize_model(model_path: str):
    session = axe.InferenceSession(model_path)
    print("[INFO] Depth model loaded")
    return session

def process_for_depth(frame: np.ndarray, target_size=(384, 256)) -> np.ndarray:
    resized = cv2.resize(frame, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)  # NHWC uint8
    return tensor

def create_depth_visualization(depth_map: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
    depth_feature = depth_map.squeeze()
    normalized = (depth_feature - depth_feature.min()) / (depth_feature.max() - depth_feature.min() + 1e-6)
    depth_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    depth_resized = cv2.resize(depth_colored, (original_frame.shape[1], original_frame.shape[0]))
    return depth_resized

def get_video_stream():
    camera = initialize_camera()
    model = initialize_model(MODEL_PATH)
    input_name = model.get_inputs()[0].name
    times = []
    last_report = time.time()

    cam_times, prep_times, infer_times, encode_times = [], [], [], []

    try:
        while True:
            frame_start = time.perf_counter()

            if not camera.isOpened():
                print("[WARN] Camera not open. Retrying...")
                camera.release()
                time.sleep(0.5)
                camera = initialize_camera()
                continue

            # Skip stale frames from buffer
            #for _ in range(3):
            #    camera.grab()
            #success, frame = camera.retrieve()
            success, frame = camera.read()

            start = time.perf_counter()
            cam_time = time.perf_counter() - start
            cam_times.append(cam_time)

            if not success or frame is None:
                print("[WARN] Failed to read frame. Skipping...")
                continue

            try:
                start = time.perf_counter()
                input_tensor = process_for_depth(frame)
                prep_time = time.perf_counter() - start
                prep_times.append(prep_time)

                start = time.perf_counter()
                output = model.run(None, {input_name: input_tensor})[0]
                infer_time = time.perf_counter() - start
                infer_times.append(infer_time)

                depth_vis = create_depth_visualization(output, frame)
            except Exception as e:
                print(f"[ERROR] Depth model failed: {e}")
                depth_vis = frame

            combined = np.concatenate([frame, depth_vis], axis=1)

            start = time.perf_counter()
            ret, buffer = cv2.imencode('.jpg', combined)
            encode_time = time.perf_counter() - start
            encode_times.append(encode_time)

            if not ret:
                print("[WARN] JPEG encode failed.")
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            total_time = time.perf_counter() - frame_start
            times.append(total_time)

            if time.time() - last_report >= 5.0:
                if times:
                    def stats(arr):
                        return (sum(arr) / len(arr), max(arr), min(arr)) if arr else (0, 0, 0)
                    fps = len(times) / 5.0
                    c_avg, c_max, c_min = stats(cam_times)
                    p_avg, p_max, p_min = stats(prep_times)
                    i_avg, i_max, i_min = stats(infer_times)
                    e_avg, e_max, e_min = stats(encode_times)
                    t_avg, t_max, t_min = stats(times)
                    print(f"[PERF] FPS: {fps:.1f} | Total Avg: {t_avg:.4f}s Max: {t_max:.4f}s Min: {t_min:.4f}s")
                    print(f"       Camera Avg: {c_avg:.4f}s, Prep Avg: {p_avg:.4f}s, Infer Avg: {i_avg:.4f}s, Encode Avg: {e_avg:.4f}s")
                    times.clear()
                    cam_times.clear()
                    prep_times.clear()
                    infer_times.clear()
                    encode_times.clear()
                last_report = time.time()

            time.sleep(0.005)

    finally:
        camera.release()

@app.get("/video")
async def video_endpoint():
    return StreamingResponse(get_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
