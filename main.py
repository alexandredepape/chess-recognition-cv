import base64

import cv2 as cv
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from pieces_cropper import get_fen_from_image
from predictor import detect_2dChessboard, clean_ouput_folder

app = FastAPI()


class Image(BaseModel):
    image_base_64: str


INPUT_IMAGE_FILE = "input_image.png"


def bytes_to_array(image_bytes):
    return cv.imdecode(np.frombuffer(image_bytes, np.uint8), -1)


@app.post("/predict")
async def predict(image: Image):
    clean_ouput_folder()
    image_bytes = base64.b64decode(image.image_base_64)
    input_image = bytes_to_array(image_bytes)
    input_image = cv.rotate(input_image, cv.ROTATE_90_CLOCKWISE)
    cv.imwrite("output/" + INPUT_IMAGE_FILE, input_image)
    cropped = detect_2dChessboard(input_image, INPUT_IMAGE_FILE)
    if cropped is None:
        print("No chessboard found")
        return {"error": "No chessboard found"}
    fen = get_fen_from_image(cropped)
    return {"fen": fen + " w KQkq - 0 1"}


def size(b64string):
    return (len(b64string) * 3) / 4 - b64string.count('=', -2)
