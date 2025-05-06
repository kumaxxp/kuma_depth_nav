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
            <title>Depth Stream</title>
        </head>
        <body>
            <h1>Depth Anything ストリーム</h1>
            <img src="/video" width="512" height="256" />
        </body>
    </html>
    """

def initialize_model(model_path: str):
    session = axe.InferenceSession(model_path)
    input_info = session.get_inputs()[0]
    print("[INFO] Model Input Shape:", input_info.shape)
    print("[INFO] Model Input Dtype:", input_info.dtype)
    return session

def initialize_camera(index=0, width=640, height=480):
    cam = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cam

def process_frame(frame: np.ndarray, target_size=(384, 256)) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ValueError("Empty frame received.")

    resized = cv2.resize(frame, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)
    if tensor.nbytes % np.dtype(np.uint8).itemsize != 0:
        raise ValueError("[ERROR] Tensor buffer size not aligned with dtype")
    return tensor

def create_depth_visualization(depth_map: np.ndarray, original_frame: np.ndarray) -> np.ndarray:
    depth_feature = depth_map.reshape(depth_map.shape[-2:])
    normalized = (depth_feature - depth_feature.min()) / (depth_feature.max() - depth_feature.min() + 1e-6)
    depth_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    depth_resized = cv2.resize(depth_colored, (original_frame.shape[1], original_frame.shape[0]))
    return np.concatenate([original_frame, depth_resized], axis=1)

def get_video_stream():
    session = initialize_model(MODEL_PATH)
    input_info = session.get_inputs()[0]
    input_name = input_info.name

    camera = initialize_camera()

    try:
        while True:
            success, frame = camera.read()

            if not success or frame is None:
                print("[WARN] Frame read failed")
                continue

            try:
                input_tensor = process_frame(frame)
                depth_output_raw = session.run(None, {input_name: input_tensor})

                if not depth_output_raw or not isinstance(depth_output_raw[0], np.ndarray):
                    print("[ERROR] Invalid output from model")
                    continue

                depth_output = depth_output_raw[0].squeeze()
                if depth_output.ndim != 2:
                    print(f"[ERROR] Unexpected depth output shape: {depth_output.shape}")
                    continue

                visualization = create_depth_visualization(depth_output, frame)

                _, buffer = cv2.imencode('.jpg', visualization)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            except Exception as e:
                print(f"[ERROR] Frame processing failed: {e}")
                traceback.print_exc()
                continue

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
