from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from typing import Generator
import time

app = FastAPI()



from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Video Stream</title>
        </head>
        <body>
            <h1>USBカメラの映像</h1>
            <img src="/video" width="640" height="480" />
        </body>
    </html>
    """




def resize_frame(frame: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def get_video_stream() -> Generator[bytes, None, None]:
    camera = cv2.VideoCapture(0)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            frame = resize_frame(frame)
            time.sleep(0.005)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
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
    uvicorn.run(app, host="0.0.0.0", port=8888)
