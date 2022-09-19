import base64

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Image(BaseModel):
    image_base_64: str


@app.post("/predict")
async def predict(image: Image):
    print("Image size: ", size(image.image_base_64))
    # convert base64 string to bytes
    image_bytes = base64.b64decode(image.image_base_64)
    # print the size of the image in megabytes
    print("Image size: ", len(image_bytes) / 1000000)
    with open("image.png", "wb") as fh:
        fh.write(image_bytes)
    return {"fen": "k7/8/8/8/8/8/8/1K6 b KQkq e3 0 1"}


def size(b64string):
    return (len(b64string) * 3) / 4 - b64string.count('=', -2)
