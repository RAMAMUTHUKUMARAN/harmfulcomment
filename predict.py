from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer from Hugging Face Hub
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("ramamuthukumaran/harmfulupdatedmodel")
model = AutoModelForSequenceClassification.from_pretrained("ramamuthukumaran/harmfulupdatedmodel")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    return prediction, probs[0][prediction].item()

# Async prediction function
def prediction(text):
    label, confidence = predict(text)
    # True if harmful (label 1), else False
    print(label==1)
    return {
        "result": label == 1,
    }

# Example usage
# import asyncio
# print(asyncio.run(prediction("giveaway")))

# while True:
#     try:
#         text = input("Enter a text: ").strip()
#         label, confidence = predict(text)
#         if label == 0:
#             label = "Not Scam"
#         else:
#             label = "Scam"
#         print(f"Predicted label: {label}, Confidence: {confidence:.2f}")
#         if not text:
#             break
#     except EOFError:
#         break