from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from mangum import Mangum

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("ramamuthukumaran/harmfulupdatedmodel")
model = AutoModelForSequenceClassification.from_pretrained("ramamuthukumaran/harmfulupdatedmodel")

class CommentRequest(BaseModel):
    comment: str

@app.post("/predict")
def predict(request: CommentRequest):
    inputs = tokenizer(request.comment, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    return {
        "result": prediction == 1,
        "confidence": round(probs[0][prediction].item(), 4)
    }

# Needed to make it work in Vercel
handler = Mangum(app)
