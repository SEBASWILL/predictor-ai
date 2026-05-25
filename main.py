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

modelo = tf.keras.models.load_model("hipotenusa_model.keras")

SCALER_MAX = np.sqrt(2)  # ← debe ser el mismo valor que usaste al entrenar

class Entrada(BaseModel):
    c1: float
    c2: float

@app.post("/predecir")
def predecir(datos: Entrada):
    a = abs(datos.c1)
    b = abs(datos.c2)

    mayor = max(a, b)
    menor = min(a, b)

    r = menor / mayor                               # razón siempre entre 0 y 1

    entrada = np.array([[r]], dtype=float)
    pred_norm = modelo.predict(entrada, verbose=0)[0][0]

    factor = pred_norm * SCALER_MAX                 # desnormalizar
    resultado = mayor * factor                      # escalar al tamaño real

    return {
        "prediccion": round(float(resultado), 6),
        "real": round(float(np.sqrt(a**2 + b**2)), 6)   # opcional, para comparar
    }

@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/2", response_class=HTMLResponse)
def root():
    with open("index1.html", "r") as f:
        return f.read()
