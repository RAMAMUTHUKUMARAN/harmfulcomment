from fastapi import FastAPI
from pydantic import BaseModel
from predict import prediction

app = FastAPI()

# Request body schema
class CommentRequest(BaseModel):
    comment: str

# Response schema
class PredictionResponse(BaseModel):
    result: bool

@app.post("/predict", response_model=PredictionResponse)
def get_prediction(request: CommentRequest):
    result = prediction(request.comment)
    return result
