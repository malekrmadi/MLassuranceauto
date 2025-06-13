import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Recommandation d'Assurance Auto",
    page_icon="🚗",
    layout="wide"
)

# Chargement du modèle et des encodeurs
@st.cache_resource
def load_model():
    try:
        model = joblib.load('insurance_model.pkl')
        marque_encoder = joblib.load('marque_encoder.pkl')
        type_vehicule_encoder = joblib.load('type_vehicule_encoder.pkl')
        model_info = joblib.load('model_info.pkl')
        return model, marque_encoder, type_vehicule_encoder, model_info
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        st.stop()

# Fonctions de calcul des scores (reprises du code de génération)
def calculer_score_vehicule(marque_score, valeur, age, kilometrage, type_score, chevaux_fiscaux, parking):
    """Calcule le score du véhicule selon la formule définie"""
    # Score valeur
    if valeur <= 25000:
        score_valeur = 1
    elif valeur <= 60000:
        score_valeur = 2
    elif valeur <= 120000:
        score_valeur = 3
    else:
        score_valeur = 4
    
    # Score âge
    if age <= 2:
        score_age = 4
    elif age <= 5:
        score_age = 3
    elif age <= 10:
        score_age = 2
    else:
        score_age = 1
    
    # Score kilométrage
    if kilometrage <= 30000:
        score_km = 4
    elif kilometrage <= 80000:
        score_km = 3
    elif kilometrage <= 150000:
        score_km = 2
    else:
        score_km = 1
    
    # Score chevaux fiscaux
    if chevaux_fiscaux <= 4:
        score_cv = 1
    elif chevaux_fiscaux <= 7:
        score_cv = 2
    elif chevaux_fiscaux <= 12:
        score_cv = 3
    else:
        score_cv = 4
    
    # Calcul du score final
    score = (marque_score * 0.30 + 
             score_valeur * 0.25 + 
             score_age * 0.20 + 
             score_km * 0.10 + 
             type_score * 0.10 + 
             score_cv * 0.05)
    
    # Bonus parking
    if parking:
        score += 0.3
    
    return score

def calculer_score_conducteur(accidents, experience, age, voiture_entreprise):
    """Calcule le score du conducteur selon la formule définie"""
    # Ratio accidents/expérience
    if experience == 0:
        ratio = 0 if accidents == 0 else 1
    else:
        ratio = accidents / experience
    
    if ratio == 0:
        score_ratio = 4
    elif ratio <= 0.1:
        score_ratio = 3
    elif ratio <= 0.3:
        score_ratio = 2
    else:
        score_ratio = 1
    
    # Score âge
    if age < 22:
        score_age = 2
    elif 22 <= age <= 35:
        score_age = 4
    elif 36 <= age <= 50:
        score_age = 4
    elif 51 <= age <= 65:
        score_age = 3
    else:
        score_age = 2
    
    # Calcul du score final
    score = score_ratio * 0.80 + score_age * 0.20
    
    # Malus entreprise
    if voiture_entreprise:
        score -= 0.4
    
    return score

def classifier_vehicule(score):
    """Classifie le véhicule selon son score"""
    if score <= 2.0:
        return "Low Cost"
    elif score <= 3.0:
        return "Medium Cost"
    else:
        return "High Cost"

def classifier_conducteur(score):
    """Classifie le conducteur selon son score"""
    if score <= 2.2:
        return "À risque"
    elif score <= 3.1:
        return "Modéré"
    else:
        return "Expert"

def generer_texte_personnalise(type_conducteur, type_vehicule, assurance_recommandee, 
                              age_conducteur, experience, marque, valeur_vehicule):
    """Génère un texte personnalisé pour la recommandation"""
    
    # Messages selon le profil conducteur
    profil_conducteur_msg = {
        "Expert": f"Félicitations ! Avec vos {experience} années d'expérience, vous êtes considéré comme un conducteur expert.",
        "Modéré": f"Votre profil de {experience} années d'expérience vous classe comme conducteur modéré.",
        "À risque": "Votre profil nécessite une attention particulière en termes d'assurance."
    }
    
    # Messages selon le type de véhicule
    vehicule_msg = {
        "Low Cost": f"Votre {marque} est classée dans la catégorie économique.",
        "Medium Cost": f"Votre {marque} est une voiture de gamme standard.",
        "High Cost": f"Votre {marque} est un véhicule haut de gamme d'une valeur de {valeur_vehicule:,} DT."
    }
    
    # Messages selon l'assurance recommandée
    assurance_msg = {
        "Tiers": "L'assurance au Tiers est recommandée pour votre profil. Elle couvre votre responsabilité civile et représente l'option la plus économique.",
        "Tiers Plus": "L'assurance Tiers Plus est idéale pour vous. Elle inclut la responsabilité civile plus des garanties additionnelles comme le vol et l'incendie.",
        "Tout Risque": "L'assurance Tous Risques est recommandée pour votre profil. Elle offre une couverture complète incluant les dommages à votre propre véhicule."
    }
    
    # Construction du message final
    message = f"""
    🎯 **Analyse de votre profil :**
    
    👤 **Conducteur :** {profil_conducteur_msg[type_conducteur]}
    
    🚗 **Véhicule :** {vehicule_msg[type_vehicule]}
    
    📋 **Recommandation :** {assurance_msg[assurance_recommandee]}
    
    💡 **Conseil personnalisé :** Basé sur votre âge de {age_conducteur} ans et votre véhicule {marque}, 
    cette recommandation optimise le rapport couverture/prix pour votre situation.
    """
    
    return message

# Interface utilisateur
def main():
    model, marque_encoder, type_vehicule_encoder, model_info = load_model()
    
    st.title("🚗 Système de Recommandation d'Assurance Auto")
    st.markdown("---")
    
    # Affichage des performances du modèle
    st.sidebar.markdown("### 📊 Performances du Modèle")
    st.sidebar.metric("Précision", f"{model_info['accuracy']*100:.1f}%")
    
    # Formulaire de saisie
    st.header("📝 Informations sur votre profil")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚗 Informations Véhicule")
        
        marque = st.selectbox("Marque du véhicule", model_info['marques'])
        type_vehicule = st.selectbox("Type de véhicule", model_info['types_vehicules'])
        age_vehicule = st.slider("Âge du véhicule (années)", 0, 20, 5)
        valeur_vehicule = st.number_input("Valeur du véhicule (DT)", 5000, 500000, 50000, step=5000)
        kilometrage = st.number_input("Kilométrage", 0, 400000, 80000, step=5000)
        chevaux_fiscaux = st.slider("Chevaux fiscaux", 1, 25, 7)
        place_parking = st.checkbox("Place de parking sécurisée")
        voiture_entreprise = st.checkbox("Voiture d'entreprise")
    
    with col2:
        st.subheader("👤 Informations Conducteur")
        
        age_conducteur = st.slider("Âge du conducteur", 18, 80, 35)
        annees_experience = st.slider("Années d'expérience", 0, min(age_conducteur-18, 60), 10)
        nombre_accidents = st.slider("Nombre d'accidents déclarés", 0, 10, 0)
    
    # Bouton de prédiction
    if st.button("🎯 Obtenir ma recommandation", type="primary"):
        try:
            # Préparation des données pour la prédiction
            marque_encoded = marque_encoder.transform([marque])[0]
            type_vehicule_encoded = type_vehicule_encoder.transform([type_vehicule])[0]
            
            # Création du vecteur de features
            features_vector = np.array([[
                marque_encoded, type_vehicule_encoded, age_vehicule,
                valeur_vehicule, kilometrage, chevaux_fiscaux,
                int(place_parking), int(voiture_entreprise),
                age_conducteur, annees_experience, nombre_accidents
            ]])
            
            # Prédiction
            prediction = model.predict(features_vector)[0]
            prediction_proba = model.predict_proba(features_vector)[0]
            
            # Calcul des scores pour affichage
            marques_scores = {
                'Dacia': 1, 'Fiat': 1, 'Hyundai': 1, 'Kia': 1, 'Chevrolet': 1, 'Chery': 1,
                'Renault': 2, 'Peugeot': 2, 'Citroën': 2, 'Volkswagen': 2, 'Ford': 2, 'Toyota': 2, 'Nissan': 2,
                'Audi': 3, 'BMW': 3, 'Mercedes': 3, 'Volvo': 3, 'Lexus': 3, 'Infiniti': 3,
                'Porsche': 4, 'Jaguar': 4, 'Land Rover': 4, 'Maserati': 4, 'Bentley': 4, 'Ferrari': 4, 'Lamborghini': 4
            }
            
            types_scores = {
                'Citadine': 1, 'Voiture populaire': 1,
                'Berline compact': 2, 'Break familial': 2,
                'SUV': 3, 'Berline executive': 3,
                'Voiture de luxe': 4, 'Sportive': 4
            }
            
            marque_score = marques_scores.get(marque, 2)
            type_score = types_scores.get(type_vehicule, 2)
            
            score_vehicule = calculer_score_vehicule(marque_score, valeur_vehicule, age_vehicule, 
                                                   kilometrage, type_score, chevaux_fiscaux, place_parking)
            score_conducteur = calculer_score_conducteur(nombre_accidents, annees_experience, 
                                                       age_conducteur, voiture_entreprise)
            
            type_vehicule_classe = classifier_vehicule(score_vehicule)
            type_conducteur_classe = classifier_conducteur(score_conducteur)
            
            # Affichage des résultats
            st.markdown("---")
            st.header("🎯 Votre Recommandation Personnalisée")
            
            # Métriques principales
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Type de Conducteur", type_conducteur_classe)
            with col2:
                st.metric("Catégorie Véhicule", type_vehicule_classe)
            with col3:
                st.metric("Assurance Recommandée", prediction)
            
            # Scores détaillés
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Score Conducteur", f"{score_conducteur:.2f}/4.0")
            with col2:
                st.metric("Score Véhicule", f"{score_vehicule:.2f}/4.0")
            
            # Probabilités de prédiction
            st.subheader("📊 Confiance de la prédiction")
            proba_df = pd.DataFrame({
                'Type d\'assurance': model.classes_,
                'Probabilité (%)': [f"{p*100:.1f}%" for p in prediction_proba]
            })
            
            st.dataframe(proba_df, use_container_width=True)
            
            # Message personnalisé
            st.subheader("💬 Analyse Détaillée")
            message_personnalise = generer_texte_personnalise(
                type_conducteur_classe, type_vehicule_classe, prediction,
                age_conducteur, annees_experience, marque, valeur_vehicule
            )
            st.markdown(message_personnalise)
            
            # Conseils additionnels
            st.subheader("💡 Conseils pour optimiser votre assurance")
            conseils = []
            
            if not place_parking:
                conseils.append("• Considérez sécuriser une place de parking pour réduire les risques de vol")
            
            if nombre_accidents > 0:
                conseils.append("• Envisagez un stage de conduite défensive pour améliorer votre profil")
            
            if age_vehicule > 10:
                conseils.append("• Votre véhicule étant ancien, vérifiez si l'assurance tous risques est rentable")
            
            if score_conducteur < 2.5:
                conseils.append("• Avec plus d'expérience sans sinistre, vous pourrez accéder à de meilleures offres")
            
            if conseils:
                for conseil in conseils:
                    st.markdown(conseil)
            else:
                st.success("Votre profil est optimal ! Vous bénéficiez de la meilleure recommandation.")
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {e}")
    
    # Informations sur le système
    with st.expander("ℹ️ À propos du système"):
        st.markdown("""
        Ce système de recommandation d'assurance utilise un modèle Random Forest entraîné sur 1000 profils.
        
        **Critères d'évaluation :**
        - Score véhicule basé sur : marque, valeur, âge, kilométrage, type, chevaux fiscaux
        - Score conducteur basé sur : ratio accidents/expérience, âge
        - Bonus/malus : parking sécurisé, voiture d'entreprise
        
        **Types d'assurance :**
        - **Tiers** : Couverture minimale obligatoire
        - **Tiers Plus** : Tiers + vol, incendie, bris de glace
        - **Tous Risques** : Couverture complète incluant vos propres dommages
        """)

if __name__ == "__main__":
    main()