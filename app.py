from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import base64
from contextlib import asynccontextmanager

# Nossos novos módulos do Cassandra
from db.seeder import seed_mnist_data
from model import predict, load_model


# Gerenciador de inicialização do FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Executa quando o servidor está ligando
    print("🚀 Verificando banco de dados Cassandra e populando MNIST...")
    seed_mnist_data() 
    
    # Carrega o modelo (agora lerá os dados direto do Cassandra)
    global model
    model = load_model()
    yield
    # Executa quando o servidor está desligando (se quiser fechar conexões)

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# Modelo Pydantic para capturar a string em Base64 enviada pelo JavaScript
class ImageData(BaseModel):
    image: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Fixed: passed the dictionary explicitly to the context keyword argument
    return templates.TemplateResponse(name="index.html", context={"request": request})


@app.post("/predict")
async def predict_image(data: ImageData):
    header, encoded = data.image.split(",", 1)
    image_bytes = base64.b64decode(encoded)

    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((28, 28), Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32)

    img_array[img_array < 50] = 0.0
    img_array = np.clip(img_array * 1.5, 0.0, 255.0)
    img_array = (img_array - 127.5) / 127.5
    img_array = img_array.reshape(1, 784)

    prediction = predict(model, img_array)

    return {"prediction": int(prediction)}