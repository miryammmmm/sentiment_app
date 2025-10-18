from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="API d'Analyse de Sentiments")

# --- Charger le modèle et le vectorizer ---
model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# --- Modèle de données attendu ---
class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Bienvenue dans l'API d'analyse de sentiments !"}

@app.post("/predict")
def predict_sentiment(data: TextInput):
    text = data.text
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    label = "positif" if prediction == 1 else "négatif"
    return {"texte": text, "sentiment": label}
