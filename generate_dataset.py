import pandas as pd
import numpy as np
import random

# Fixer une seed pour la reproductibilité
random.seed(42)
np.random.seed(42)

def generate_profile_conducteur(profile_type):
    """Génère l'âge, l'expérience et accidents selon le profil conducteur."""
    if profile_type == "modéré":
        # âge entre 25 et 55
        age = np.random.randint(25, 56)
        # expérience entre 5 et (age-18)
        experience = np.random.randint(5, max(6, age - 18 + 1))
        # accidents 0 ou 1 accident
        accidents = np.random.choice([0, 1], p=[0.7, 0.3])
    elif profile_type == "risque élevé":
        # âge entre 18 et 35 (plus jeune)
        age = np.random.randint(18, 36)
        # expérience faible 0 à 5 ans
        experience = np.random.randint(0, min(6, age - 18 + 1))
        # accidents 1 à 3
        accidents = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
    elif profile_type == "expérimenté":
        # âge entre 40 et 75
        age = np.random.randint(40, 76)
        # expérience entre 15 et (age-18)
        experience = np.random.randint(15, max(16, age - 18 + 1))
        # accidents 0 ou 1 (rare)
        accidents = np.random.choice([0, 1], p=[0.85, 0.15])
    else:
        raise ValueError("Profil inconnu")
    return age, experience, accidents

def generate_vehicle_age():
    """Génère l'âge du véhicule avec une moyenne autour de 7-8 ans, réparti entre 0 et 22 ans,
    avec ~2% sous 4 ans."""
    # 20% chance d'avoir un véhicule < 4 ans
    if np.random.rand() < 0.2:
        age_vehicule = np.random.randint(0, 4)
    else:
        # sinon âge entre 4 et 22 ans, distribué uniformément
        age_vehicule = np.random.randint(4, 23)
    return age_vehicule

def generate_vehicle_price_category(age_vehicule):
    """Catégorise le véhicule en peu coûteux, intermédiaire, cher selon son âge."""
    # Plus la voiture est vieille, plus elle est peu coûteuse
    # Exemple simple :
    if age_vehicule <= 5:
        # neuf ou quasi neuf : cher (30%), intermédiaire (50%), peu coûteux (20%)
        cat = np.random.choice(['cher', 'intermédiaire', 'peu coûteux'], p=[0.3, 0.5, 0.2])
    elif age_vehicule <= 12:
        # intermédiaire (50%), peu coûteux (40%), cher (10%)
        cat = np.random.choice(['cher', 'intermédiaire', 'peu coûteux'], p=[0.1, 0.5, 0.4])
    else:
        # vieux véhicule : peu coûteux (70%), intermédiaire (30%), cher (0%)
        cat = np.random.choice(['cher', 'intermédiaire', 'peu coûteux'], p=[0.0, 0.3, 0.7])
    return cat

def determine_parking():
    """Chance de parking = 60%"""
    return np.random.choice([True, False], p=[0.6, 0.4])

def determine_profile_from_inputs(age, experience, accidents):
    """Logique déterministe pour classer profil conducteur à partir des inputs"""
    ratio_accident = accidents / max(experience, 1)
    score_age = 0
    # Score âge : plus loin de 25-55, plus risque (simple approximation)
    if age < 25:
        score_age = 1
    elif age > 55:
        score_age = 0.5

    risk_score = ratio_accident + score_age
    if risk_score > 0.3:
        return "risque élevé"
    elif risk_score > 0.1:
        return "modéré"
    else:
        return "expérimenté"

def determine_vehicle_value(age_vehicule, price_cat):
    """Détermine la catégorie véhicule selon âge et prix."""
    # On fusionne info âge + prix pour définir une valeur simple : peu coûteux, intermédiaire, cher
    if price_cat == 'cher' and age_vehicule <= 10:
        return "cher"
    elif price_cat == 'peu coûteux' or age_vehicule > 15:
        return "peu coûteux"
    else:
        return "intermédiaire"

def determine_assurance(profil, valeur_vehicule, parking):
    """La matrice d'assurance selon profil conducteur et valeur véhicule + parking"""
    # Cas parking = False, plus de chances d'avoir Tiers Plus que Tiers
    if parking is False:
        if profil == "risque élevé":
            if valeur_vehicule == "peu coûteux":
                return "Tiers Plus"
            elif valeur_vehicule == "intermédiaire":
                return "Tiers Plus"
            else:
                return "Tous Risques"
        elif profil == "modéré":
            if valeur_vehicule == "peu coûteux":
                return "Tiers Plus"
            elif valeur_vehicule == "intermédiaire":
                return "Tiers Plus"
            else:
                return "Tous Risques"
        else:  # expérimenté
            if valeur_vehicule == "peu coûteux":
                return "Tiers Plus"
            else:
                return "Tous Risques"
    else:
        # Parking True : appliquer la matrice classique
        matrice = {
            "risque élevé": {
                "peu coûteux": "Tiers",
                "intermédiaire": "Tiers",
                "cher": "Tiers Plus",
            },
            "modéré": {
                "peu coûteux": "Tiers",
                "intermédiaire": "Tiers Plus",
                "cher": "Tous Risques",
            },
            "expérimenté": {
                "peu coûteux": "Tiers Plus",
                "intermédiaire": "Tous Risques",
                "cher": "Tous Risques",
            },
        }
        return matrice[profil][valeur_vehicule]

def main():
    total = 1000
    repartition = {
        "modéré": int(total * 0.4),
        "risque élevé": int(total * 0.35),
        "expérimenté": int(total * 0.25),
    }

    data = []
    for profil, nb in repartition.items():
        for _ in range(nb):
            age, experience, accidents = generate_profile_conducteur(profil)
            age_vehicule = generate_vehicle_age()
            price_cat = generate_vehicle_price_category(age_vehicule)
            parking = determine_parking()
            profile_determined = determine_profile_from_inputs(age, experience, accidents)
            valeur_vehicule = determine_vehicle_value(age_vehicule, price_cat)
            assurance = determine_assurance(profile_determined, valeur_vehicule, parking)

            data.append({
                "age_conducteur": age,
                "annees_experience": experience,
                "accidents": accidents,
                "age_vehicule": age_vehicule,
                "prix_vehicule_categorie": price_cat,
                "parking": parking,
                "profil_conducteur": profile_determined,
                "valeur_vehicule": valeur_vehicule,
                "assurance_recommandee": assurance
            })

    df = pd.DataFrame(data)
    df.to_csv("assurance_synthetique_1000.csv", index=False)
    print("Dataset 1000 lignes créé dans assurance_synthetique_1000.csv")

if __name__ == "__main__":
    main()
