import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# ====== Chargement du modèle ======
model_path = r"C:\Users\ASUS\Desktop\sentiment_app\sentiment_model.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Fichier introuvable : {model_path}")

print("✅ Chargement du modèle...")
model = joblib.load(model_path)
print("➡️ Modèle chargé :", type(model))

# ====== Exemple de texte à tester ======
text = ["J'adore ce film, il est génial !"]

try:
    # On essaie de prédire directement (cas où le modèle contient déjà un TfidfVectorizer)
    prediction = model.predict(text)
    print("\n✅ Prédiction réussie (le modèle contient un vectorizer)")
    print("Texte :", text[0])
    print("Prédiction :", prediction[0])

except Exception as e:
    print("\n⚠️ Le modèle ne contient pas de vectorizer.")
    print("Erreur :", e)
    print("Tentative de chargement d'un vectorizer séparé...")

    vectorizer_path = r"C:\Users\ASUS\Desktop\sentiment_app\tfidf_vectorizer.joblib"
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            "❌ Aucun vectorizer.joblib trouvé. "
            "Le modèle a sans doute été entraîné séparément sans être mis dans un pipeline."
        )

    # Charger le vectorizer et transformer le texte
    vectorizer = joblib.load(vectorizer_path)
    X = vectorizer.transform(text)

    prediction = model.predict(X)
    print("\n✅ Prédiction réussie après vectorisation manuelle")
    print("Texte :", text[0])
    print("Prédiction :", prediction[0])
