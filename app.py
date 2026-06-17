from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

import numpy as np
from PIL import Image
import io

from model import predict, preprocess_image, load_model


model = load_model()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Fixed: passed the dictionary explicitly to the context keyword argument
    return templates.TemplateResponse(name="index.html", context={"request": request})


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()

    img = Image.open(io.BytesIO(contents)).convert('L')
    img = img.resize((28, 28))
    img = np.array(img)

    img = 255 - img
    img = (img - 127.5) / 127.5

    prediction = predict(model, img)

    return {"prediction": int(prediction)}