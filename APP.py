import streamlit as st
import requests

# -------------------------
# ⚙️ Configuration de la page
# -------------------------
st.set_page_config(
    page_title="Analyse de Sentiments - TP IA",
    page_icon="💬",
    layout="centered",
)

# -------------------------
# 💬 Titre principal
# -------------------------
st.title("💬 Application d’Analyse de Sentiments")
st.markdown(
    """
    <style>
        .stTextArea textarea {
            border: 2px solid #6c63ff;
            border-radius: 10px;
        }
        .stButton button {
            background-color: #6c63ff;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6em 1.2em;
        }
        .stButton button:hover {
            background-color: #4b47c0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("🧠 Cette application permet de prédire le **sentiment d’un texte** (positif ou négatif) à l’aide d’un modèle de Machine Learning entraîné sur des données textuelles.")

# -------------------------
# 🔗 URL de l'API FastAPI
# -------------------------
API_URL = API_URL = "https://sentiment-app-dko7.onrender.com/predict"

# -------------------------
# 📝 Zone de texte utilisateur
# -------------------------
st.subheader("✍️ Saisissez un texte à analyser :")
text = st.text_area("Votre texte :", height=130, placeholder="Exemple : J'adore ce film, il est incroyable !")

# -------------------------
# 🧮 Analyse du sentiment
# -------------------------
if st.button("🔍 Analyser le sentiment"):
    if text.strip():
        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(API_URL, json={"text": text})
                if response.status_code == 200:
                    data = response.json()
                    sentiment = data["sentiment"]

                    st.subheader("📊 Résultat de l’analyse :")

                    if sentiment == "positif":
                        st.success("😊 **Sentiment positif détecté !**")
                    elif sentiment == "négatif":
                        st.error("😞 **Sentiment négatif détecté.**")
                    else:
                        st.warning("😐 **Sentiment neutre ou incertain.**")

                else:
                    st.error("⚠️ Erreur : impossible d’obtenir une réponse de l’API.")
            except Exception as e:
                st.error(f"Erreur de connexion à l’API : {e}")
    else:
        st.warning("⚠️ Veuillez saisir un texte avant de lancer l’analyse.")

# -------------------------
# ℹ️ Section d’informations
# -------------------------
st.markdown("---")
st.subheader("ℹ️ À propos du modèle")

with st.expander("Voir les détails du modèle entraîné"):
    st.markdown("""
    - **Type de modèle :** Régression Logistique (`LogisticRegression`)
    - **Vectorisation :** TF-IDF (`TfidfVectorizer`)
    - **Langue :** Français 🇫🇷  
    - **But :** Déterminer si un texte exprime une opinion **positive** ou **négative**
    - **Entrée :** Phrase ou texte libre  
    - **Sortie :** Label de sentiment (`positif` ou `négatif`)
    """)

st.caption("💡 Projet développé dans le cadre du TP Analyse de Sentiments (IA).")

