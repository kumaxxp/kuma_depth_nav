import cv2
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse

# Depth Anything用
from depth_processor import DepthProcessor, create_depth_visualization, create_depth_grid_visualization

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
            time.sleep(0.001)
            continue
        # 推論用にリサイズ（高速化）
        small = cv2.resize(frame, (224, 224))
        depth_map, _ = depth_processor.predict(small)
        vis = create_depth_visualization(depth_map, small.shape)
        # 表示用に拡大
        vis = cv2.resize(vis, (320, 240))
        # JPEG品質を下げて高速化
        ret, buffer = cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        # sleepを最小限に
        time.sleep(0.001)

def get_camera_stream():
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.001)
            continue
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.001)

def get_depth_grid_stream():
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.001)
            continue
        # 推論用にリサイズ（高速化）
        small = cv2.resize(frame, (224, 224))
        depth_map, _ = depth_processor.predict(small)
        grid_img = create_depth_grid_visualization(depth_map, grid_size=(12, 16), cell_size=18)
        # 表示用に拡大
        grid_img = cv2.resize(grid_img, (320, 240))
        # JPEG品質を下げて高速化
        ret, buffer = cv2.imencode('.jpg', grid_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        # sleepを最小限に
        time.sleep(0.001)

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
        <h2>Depth Grid</h2>
        <img src="/depth_grid" />
    </body>
    </html>
    """

@app.get("/video")
async def video():
    return StreamingResponse(get_camera_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/depth_video")
async def depth_video():
    return StreamingResponse(get_depth_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/depth_grid")
async def depth_grid():
    return StreamingResponse(get_depth_grid_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)