import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Configuration des données
np.random.seed(42)
random.seed(42)

# Dictionnaires de données
MARQUES = {
    'Dacia': 1, 'Fiat': 1, 'Hyundai': 1, 'Kia': 1, 'Chevrolet': 1, 'Chery': 1,
    'Renault': 2, 'Peugeot': 2, 'Citroën': 2, 'Volkswagen': 2, 'Ford': 2, 'Toyota': 2, 'Nissan': 2,
    'Audi': 3, 'BMW': 3, 'Mercedes': 3, 'Volvo': 3, 'Lexus': 3, 'Infiniti': 3,
    'Porsche': 4, 'Jaguar': 4, 'Land Rover': 4, 'Maserati': 4, 'Bentley': 4, 'Ferrari': 4, 'Lamborghini': 4
}

TYPES_VEHICULE = {
    'Citadine': 1, 'Voiture populaire': 1,
    'Berline compact': 2, 'Break familial': 2,
    'SUV': 3, 'Berline executive': 3,
    'Voiture de luxe': 4, 'Sportive': 4
}

def generer_valeur_vehicule(marque_score, type_score, age):
    """Génère une valeur de véhicule cohérente avec la marque, type et âge"""
    base_value = 0
    
    # Valeur de base selon la marque
    if marque_score == 1:
        base_value = np.random.uniform(8000, 35000)
    elif marque_score == 2:
        base_value = np.random.uniform(20000, 70000)
    elif marque_score == 3:
        base_value = np.random.uniform(50000, 150000)
    else:  # marque_score == 4
        base_value = np.random.uniform(100000, 500000)
    
    # Ajustement selon le type
    type_multiplier = 0.7 + (type_score - 1) * 0.2
    base_value *= type_multiplier
    
    # Dépréciation selon l'âge
    depreciation = max(0.3, 1 - (age * 0.08))
    final_value = base_value * depreciation
    
    return int(final_value)

def generer_kilometrage(age, voiture_entreprise):
    """Génère un kilométrage cohérent avec l'âge et l'usage"""
    km_par_an = 20000 if voiture_entreprise else np.random.uniform(8000, 18000)
    base_km = age * km_par_an
    variation = np.random.uniform(0.7, 1.3)
    return int(base_km * variation)

def generer_chevaux_fiscaux(marque_score, type_score):
    """Génère les chevaux fiscaux selon la marque et le type"""
    base_cv = 0
    
    if marque_score == 1:
        base_cv = np.random.randint(3, 8)
    elif marque_score == 2:
        base_cv = np.random.randint(5, 10)
    elif marque_score == 3:
        base_cv = np.random.randint(8, 15)
    else:
        base_cv = np.random.randint(12, 25)
    
    # Ajustement selon le type
    if type_score >= 3:
        base_cv += np.random.randint(2, 5)
    
    return base_cv

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

def recommander_assurance(type_conducteur, type_vehicule):
    """Recommande l'assurance selon la matrice définie"""
    matrice = {
        ("À risque", "Low Cost"): "Tiers",
        ("À risque", "Medium Cost"): "Tiers",
        ("À risque", "High Cost"): "Tiers Plus",
        ("Modéré", "Low Cost"): "Tiers",
        ("Modéré", "Medium Cost"): "Tiers Plus",
        ("Modéré", "High Cost"): "Tout Risque",
        ("Expert", "Low Cost"): "Tiers Plus",
        ("Expert", "Medium Cost"): "Tout Risque",
        ("Expert", "High Cost"): "Tout Risque"
    }
    
    return matrice.get((type_conducteur, type_vehicule), "Tiers")

# Génération de la base de données
def generer_dataset(n_lignes=1000):
    """Génère un dataset complet de n_lignes"""
    
    data = []
    
    for i in range(n_lignes):
        # Données véhicule
        marque = random.choice(list(MARQUES.keys()))
        marque_score = MARQUES[marque]
        
        type_vehicule = random.choice(list(TYPES_VEHICULE.keys()))
        type_score = TYPES_VEHICULE[type_vehicule]
        
        age_vehicule = np.random.randint(0, 15)
        parking = random.choice([True, False])
        voiture_entreprise = random.choice([True, False])
        
        # Génération cohérente des autres attributs véhicule
        kilometrage = generer_kilometrage(age_vehicule, voiture_entreprise)
        chevaux_fiscaux = generer_chevaux_fiscaux(marque_score, type_score)
        valeur_vehicule = generer_valeur_vehicule(marque_score, type_score, age_vehicule)
        
        # Données conducteur
        age_conducteur = np.random.randint(18, 75)
        experience = min(age_conducteur - 18, np.random.randint(0, age_conducteur - 17))
        
        # Génération cohérente du nombre d'accidents
        if experience <= 2:
            accidents = np.random.choice([0, 1], p=[0.8, 0.2])
        elif experience <= 10:
            accidents = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
        else:
            accidents = np.random.choice([0, 1, 2, 3], p=[0.5, 0.3, 0.15, 0.05])
        
        # Calculs des scores
        score_vehicule = calculer_score_vehicule(
            marque_score, valeur_vehicule, age_vehicule, 
            kilometrage, type_score, chevaux_fiscaux, parking
        )
        
        score_conducteur = calculer_score_conducteur(
            accidents, experience, age_conducteur, voiture_entreprise
        )
        
        # Classifications
        type_vehicule_classe = classifier_vehicule(score_vehicule)
        type_conducteur_classe = classifier_conducteur(score_conducteur)
        
        # Recommandation d'assurance
        assurance_recommandee = recommander_assurance(type_conducteur_classe, type_vehicule_classe)
        
        # Ajout à la base de données
        data.append({
            'ID': i + 1,
            'Marque': marque,
            'Type_Vehicule': type_vehicule,
            'Age_Vehicule': age_vehicule,
            'Valeur_Vehicule_DT': valeur_vehicule,
            'Kilometrage': kilometrage,
            'Chevaux_Fiscaux': chevaux_fiscaux,
            'Place_Parking': parking,
            'Voiture_Entreprise': voiture_entreprise,
            'Age_Conducteur': age_conducteur,
            'Annees_Experience': experience,
            'Nombre_Accidents': accidents,
            'Ratio_Accidents_Experience': round(accidents / max(experience, 1), 3),
            'Score_Vehicule': round(score_vehicule, 2),
            'Score_Conducteur': round(score_conducteur, 2),
            'Type_Vehicule_Classe': type_vehicule_classe,
            'Type_Conducteur_Classe': type_conducteur_classe,
            'Assurance_Recommandee': assurance_recommandee
        })
    
    return pd.DataFrame(data)

# Génération du dataset
print("Génération de la base de données en cours...")
df = generer_dataset(1000)

# Affichage des statistiques
print(f"\nBase de données générée avec {len(df)} lignes")
print("\nRépartition des types de véhicules:")
print(df['Type_Vehicule_Classe'].value_counts())
print("\nRépartition des types de conducteurs:")
print(df['Type_Conducteur_Classe'].value_counts())
print("\nRépartition des assurances recommandées:")
print(df['Assurance_Recommandee'].value_counts())

# Affichage des premières lignes
print("\nPremières lignes du dataset:")
print(df.head())

# Sauvegarde en CSV
df.to_csv('dataset_assurance_auto.csv', index=False, encoding='utf-8')
print("\nDataset sauvegardé dans 'dataset_assurance_auto.csv'")

# Validation de la matrice
print("\nValidation de la matrice de recommandation:")
validation = df.groupby(['Type_Conducteur_Classe', 'Type_Vehicule_Classe'])['Assurance_Recommandee'].first().unstack()
print(validation)