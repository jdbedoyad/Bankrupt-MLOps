from pandas import DataFrame
from joblib import load
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException


model = load("pipeline.joblib")


app = FastAPI()

# Definición de las columnas usadas por el modelo
MODEL_COLUMNS = [
    ' ROA(C) before interest and depreciation before interest',
    ' ROA(A) before interest and % after tax',
    ' ROA(B) before interest and depreciation after tax',
    ' Persistent EPS in the Last Four Seasons',
    ' Per Share Net profit before tax (Yuan ¥)',
    ' Debt ratio %',
    ' Net worth/Assets',
    ' Net profit before tax/Paid-in capital',
    ' Retained Earnings to Total Assets',
    ' Net Income to Total Assets'
]



# Clase para la validación del payload de entrada
class DataPredict(BaseModel):
    data_to_predict: list[list] = [
        [0.370594257300249, 0.424389446140427, 0.40574977247176, 0.16914058806845, 0.311664426681757, 0.998969203197885,
         0.808809360876843, 0.302646433889668, 0.780984850207341, 0.808809360876843],
        [0.464290937454297, 0.53821412996075, 0.516730017666899, 0.2089439349532, 0.318136804131004, 0.998945978205482,
         0.809300725667939, 0.303556430290771, 0.781505974330882, 0.809300725667939]
    ]

# Endpoint para realizar predicciones
@app.post("/predict")
def predict(request: DataPredict):
    """
    Endpoint para la Predicción de Bancarrota de las empresas.
    """
    try:
        # Convertir los datos validados a un DataFrame
        df_data = DataFrame(request.data_to_predict, columns=MODEL_COLUMNS)
        
        # Generar predicciones
        prediction = model.predict(df_data)
        
        # Retornar las predicciones como una lista
        return {"prediction": prediction.tolist()}
    except ValidationError as ve:
        # Manejar errores de validación
        raise HTTPException(status_code=400, detail=ve.errors())
    except Exception as e:
        # Manejar errores generales
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint auxiliar para realizar sumas
@app.get("/sum")
def sum(param1: float, param2: float):
    """
    Endpoint para sumar dos números, útil para probar la conexión con la API.
    """
    try:
        result = param1 + param2
        return {"param1": param1, "param2": param2, "sum": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint raíz para comprobar que la API está activa
@app.get("/")
def home():
    """
    Endpoint raíz de la API.
    """
    return {'Universidad EIA': 'MLOps'}
