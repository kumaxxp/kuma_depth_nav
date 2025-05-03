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
            <title>Depth Stream</title>
        </head>
        <body>
            <h1>Depth Anything ストリーム</h1>
            <img src="/video" width="512" height="256" />
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
    return np.concatenate([original_frame, depth_resized], axis=1)

def get_video_stream():
    session = initialize_model(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    camera = cv2.VideoCapture(0)

    try:
        while True:
            # フレーム取得時間の計測
            start = time.perf_counter()
            success, frame = camera.read()
            camera_time = time.perf_counter() - start

            if not success:
                break

            # 前処理時間の計測
            start = time.perf_counter()
            input_tensor = process_frame(frame)
            preprocess_time = time.perf_counter() - start

            # 推論時間の計測
            start = time.perf_counter()
            output = session.run(None, {input_name: input_tensor})
            inference_time = time.perf_counter() - start

            # 可視化時間の計測
            start = time.perf_counter()
            visualization = create_depth_visualization(output[0], frame)
            visualization_time = time.perf_counter() - start

            total_time = camera_time + preprocess_time + inference_time + visualization_time

            print(f"Frame Times - Camera: {camera_time:.4f}s, Preprocess: {preprocess_time:.4f}s, "
                  f"Inference: {inference_time:.4f}s, Visualization: {visualization_time:.4f}s, Total: {total_time:.4f}s")

            _, buffer = cv2.imencode('.jpg', visualization)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        #    time.sleep(0.005)

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
