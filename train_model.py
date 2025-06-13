import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Chargement des données
print("Chargement des données...")
df = pd.read_csv('dataset_assurance_auto.csv')

# Préparation des features
print("Préparation des features...")

# Encodage des variables catégorielles
le_marque = LabelEncoder()
le_type_vehicule = LabelEncoder()

df['Marque_encoded'] = le_marque.fit_transform(df['Marque'])
df['Type_Vehicule_encoded'] = le_type_vehicule.fit_transform(df['Type_Vehicule'])

# Conversion des booléens en entiers
df['Place_Parking_int'] = df['Place_Parking'].astype(int)
df['Voiture_Entreprise_int'] = df['Voiture_Entreprise'].astype(int)

# Sélection des features pour l'entraînement
features = [
    'Marque_encoded', 'Type_Vehicule_encoded', 'Age_Vehicule', 
    'Valeur_Vehicule_DT', 'Kilometrage', 'Chevaux_Fiscaux',
    'Place_Parking_int', 'Voiture_Entreprise_int',
    'Age_Conducteur', 'Annees_Experience', 'Nombre_Accidents'
]

X = df[features]
y = df['Assurance_Recommandee']

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Taille du jeu d'entraînement: {X_train.shape[0]}")
print(f"Taille du jeu de test: {X_test.shape[0]}")

# Entraînement du modèle Random Forest
print("\nEntraînement du modèle Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

# Prédictions
y_pred = rf_model.predict(X_test)

# Évaluation du modèle
print("\n" + "="*50)
print("ÉVALUATION DU MODÈLE")
print("="*50)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nRapport de classification détaillé:")
print(classification_report(y_test, y_pred))

print("\nMatrice de confusion:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualisation de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Tiers', 'Tiers Plus', 'Tout Risque'],
            yticklabels=['Tiers', 'Tiers Plus', 'Tout Risque'])
plt.title('Matrice de Confusion')
plt.ylabel('Valeurs Réelles')
plt.xlabel('Prédictions')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()  # 🔥 Évite d'afficher la fenêtre

# Importance des features
print("\n" + "="*50)
print("IMPORTANCE DES FEATURES")
print("="*50)

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# Visualisation de l'importance des features
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Importance des Features dans le Modèle Random Forest')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()  # 🔥 Évite d'afficher la fenêtre

# Test de quelques exemples
print("\n" + "="*50)
print("TESTS D'EXEMPLES")
print("="*50)

examples = X_test.head(10)
examples_pred = rf_model.predict(examples)
examples_actual = y_test.head(10).values

print("\nComparaison prédictions vs réalité (10 premiers exemples):")
for i in range(len(examples)):
    print(f"Exemple {i+1}: Prédit={examples_pred[i]}, Réel={examples_actual[i]}, "
          f"Correct={'✓' if examples_pred[i] == examples_actual[i] else '✗'}")

# Analyse par type d'assurance
print("\n" + "="*50)
print("ANALYSE PAR TYPE D'ASSURANCE")
print("="*50)

for assurance_type in ['Tiers', 'Tiers Plus', 'Tout Risque']:
    mask = y_test == assurance_type
    if mask.sum() > 0:
        pred_for_type = y_pred[mask]
        correct = (pred_for_type == assurance_type).sum()
        total = len(pred_for_type)
        accuracy_type = correct / total
        print(f"{assurance_type}: {correct}/{total} ({accuracy_type*100:.1f}% correct)")

# Sauvegarde du modèle et des encodeurs
print("\n" + "="*50)
print("SAUVEGARDE DU MODÈLE")
print("="*50)

joblib.dump(rf_model, 'insurance_model.pkl')
joblib.dump(le_marque, 'marque_encoder.pkl')
joblib.dump(le_type_vehicule, 'type_vehicule_encoder.pkl')

# Vérification que les fichiers sont bien créés
for filename in ['insurance_model.pkl', 'marque_encoder.pkl', 'type_vehicule_encoder.pkl']:
    if os.path.exists(filename):
        print(f"✅ Fichier sauvegardé : {filename}")
    else:
        print(f"❌ Échec de la sauvegarde : {filename}")

# Sauvegarde des informations pour l'application
model_info = {
    'features': features,
    'marques': list(le_marque.classes_),
    'types_vehicules': list(le_type_vehicule.classes_),
    'accuracy': accuracy
}

joblib.dump(model_info, 'model_info.pkl')
if os.path.exists('model_info.pkl'):
    print("✅ Informations du modèle sauvegardées: model_info.pkl")
else:
    print("❌ Échec de la sauvegarde: model_info.pkl")

print("\n" + "="*50)
print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
print("="*50)
print(f"Accuracy finale: {accuracy*100:.2f}%")
print("Fichiers générés :")
print("- insurance_model.pkl")
print("- marque_encoder.pkl")
print("- type_vehicule_encoder.pkl")
print("- model_info.pkl")
print("- confusion_matrix.png")
print("- feature_importance.png")
