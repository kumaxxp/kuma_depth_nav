import cv2
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse

# Depth Anything用
from depth_processor import DepthProcessor, create_depth_visualization

app = FastAPI()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

depth_processor = DepthProcessor()

def get_depth_stream():
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        depth_map, _ = depth_processor.predict(frame)
        vis = create_depth_visualization(depth_map, frame.shape)
        ret, buffer = cv2.imencode('.jpg', vis)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
    <head><title>Fast Camera Streaming</title></head>
    <body>
        <h2>Camera Stream</h2>
        <img src="/video" />
        <h2>Depth Anything (Depth Map)</h2>
        <img src="/depth_video" />
    </body>
    </html>
    """

def get_camera_stream():
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.01)

@app.get("/video")
async def video():
    return StreamingResponse(get_camera_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/depth_video")
async def depth_video():
    return StreamingResponse(get_depth_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)