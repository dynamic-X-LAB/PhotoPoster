from fastapi import FastAPI, Request
import json
import pickle
import base64
import sys
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import cv2
import io
from PIL import Image
from .pose_handler import Handler
app = FastAPI()
app.state.handler = Handler(sys.argv[1])

class ImageResponse(BaseModel):
    info: bytes
    pil: bytes

class ImageRequest(BaseModel):
    image: bytes


@app.post("/process-image/")
async def process_image(request: ImageRequest):
    handler = app.state.handler
    # image_bytes = await request.body()
    image_bytes = base64.b64decode(request.image)

    image = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))[:,:,::-1]
    if image is None:
        return JSONResponse(status_code=400, content={"message": "Invalid image data"})
    print(image.shape)
    info, pil = handler.predict(image)

    infoio = io.BytesIO()
    pickle.dump(info, infoio)
    info_content = base64.b64encode(infoio.getvalue()).decode()

    imgio = io.BytesIO()
    pil.save(imgio, format='JPEG', quality=95)
    pil_content = base64.b64encode(imgio.getvalue()).decode()
    print(sys.getsizeof(info_content), sys.getsizeof(pil_content))
    return ImageResponse(
        info=info_content,
        pil=pil_content
    )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
