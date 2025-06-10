import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Charger le modèle et les encodeurs
@st.cache_data
def load_model():
    with open('model_assurance.pkl', 'rb') as f:
        return pickle.load(f)

model, label_encoders = load_model()

# Fonction pour prédire et décoder
def predict_assurance(input_data):
    df = pd.DataFrame([input_data])

    # Encoder catégoriques
    for col in ['prix_vehicule_categorie', 'parking', 'profil_conducteur', 'valeur_vehicule']:
        le = label_encoders[col]
        df[col] = le.transform(df[col])

    pred_encoded = model.predict(df)[0]
    assurance_pred = label_encoders['assurance_recommandee'].inverse_transform([pred_encoded])[0]
    return assurance_pred

# Messages personnalisés selon matrice et profil

def generate_message(profil, valeur_vehicule, assurance):
    # Messages par profil
    messages_profil = {
        "risque élevé": "Votre profil présente un risque élevé en raison de votre jeune âge ou de vos antécédents d'accidents. Il est important de choisir une assurance adaptée pour vous protéger efficacement.",
        "modéré": "Vous avez un profil modéré avec une expérience correcte et un historique d'accidents limité. Vous pouvez bénéficier d'une couverture équilibrée.",
        "expérimenté": "Votre profil expérimenté reflète votre maîtrise et prudence sur la route, ce qui vous permet d'accéder à des couvertures plus larges à meilleur prix."
    }
    # Messages véhicule
    messages_vehicule = {
        "peu coûteux": "Votre véhicule peu coûteux limite les coûts de réparation, l'assurance adaptée peut être plus économique.",
        "intermédiaire": "Votre véhicule de valeur intermédiaire nécessite une couverture adaptée pour un bon équilibre qualité/prix.",
        "cher": "Votre véhicule cher demande une couverture renforcée pour éviter de lourdes pertes en cas de sinistre."
    }
    # Messages assurance
    messages_assurance = {
        "Tiers": "L'assurance Tiers est la couverture minimale obligatoire. Elle protège principalement les tiers en cas de dommages.",
        "Tiers Plus": "L'assurance Tiers Plus offre une protection supplémentaire, notamment contre le vol et l'incendie.",
        "Tous Risques": "L'assurance Tous Risques est la couverture la plus complète, protégeant votre véhicule et vous-même dans presque toutes les situations."
    }

    return f"{messages_profil[profil]}\n{messages_vehicule[valeur_vehicule]}\n{messages_assurance[assurance]}"

# Interface utilisateur
st.title("Recommandation d'assurance auto")

age_conducteur = st.number_input("Âge du conducteur", min_value=18, max_value=100, value=30)
annees_experience = st.number_input("Années d'expérience", min_value=0, max_value=80, value=5)
accidents = st.number_input("Nombre d'accidents", min_value=0, max_value=10, value=0)

age_vehicule = st.number_input("Âge du véhicule (années)", min_value=0, max_value=30, value=5)

prix_vehicule_categorie = st.selectbox("Catégorie prix véhicule", ["peu coûteux", "intermédiaire", "cher"])
parking = st.selectbox("Le véhicule est-il stationné dans un parking sécurisé ?", [True, False])

# Déterminer profil conducteur (la même logique que pour le dataset)

def determine_profile(age, exp, acc):
    ratio_acc = acc / max(exp,1)
    score_age = 0
    if age < 25:
        score_age = 1
    elif age > 55:
        score_age = 0.5
    risk_score = ratio_acc + score_age
    if risk_score > 0.3:
        return "risque élevé"
    elif risk_score > 0.1:
        return "modéré"
    else:
        return "expérimenté"

profil_conducteur = determine_profile(age_conducteur, annees_experience, accidents)

# Déterminer valeur véhicule (simplifiée)

def determine_valeur_vehicule(age_vehicule, prix_cat):
    if prix_cat == 'cher' and age_vehicule <= 10:
        return "cher"
    elif prix_cat == 'peu coûteux' or age_vehicule > 15:
        return "peu coûteux"
    else:
        return "intermédiaire"

valeur_vehicule = determine_valeur_vehicule(age_vehicule, prix_vehicule_categorie)

if st.button("Recommander assurance"):
    input_dict = {
        "age_conducteur": age_conducteur,
        "annees_experience": annees_experience,
        "accidents": accidents,
        "age_vehicule": age_vehicule,
        "prix_vehicule_categorie": [prix_vehicule_categorie],
        "parking": [parking],
        "profil_conducteur": [profil_conducteur],
        "valeur_vehicule": [valeur_vehicule]
    }

    # Flatten single values in dict for prediction
    input_for_pred = {
        "age_conducteur": age_conducteur,
        "annees_experience": annees_experience,
        "accidents": accidents,
        "age_vehicule": age_vehicule,
        "prix_vehicule_categorie": prix_vehicule_categorie,
        "parking": parking,
        "profil_conducteur": profil_conducteur,
        "valeur_vehicule": valeur_vehicule,
    }

    assurance_pred = predict_assurance(input_for_pred)
    message = generate_message(profil_conducteur, valeur_vehicule, assurance_pred)

    st.success(f"💡 Assurance recommandée : **{assurance_pred}**")
    st.info(message)
