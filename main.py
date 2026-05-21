from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

modelo = tf.keras.models.load_model("modelo_prediccion.keras")

class Entrada(BaseModel):
    c1: float
    c2: float

@app.post("/predecir")
def predecir(datos: Entrada):
    entrada = np.array([[datos.c1, datos.c2]], dtype=float)
    resultado = modelo.predict(entrada)
    return {"prediccion": float(resultado[0][0])}

@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", "r") as f:
        return f.read()