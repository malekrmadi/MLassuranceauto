import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# Normalisation des données (importante pour la régression logistique)
print("\nNormalisation des données...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement du modèle Logistic Regression
print("\nEntraînement du modèle Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    multi_class='ovr',  # One-vs-Rest pour classification multiclasse
    solver='liblinear'
)

lr_model.fit(X_train_scaled, y_train)

# Prédictions
y_pred = lr_model.predict(X_test_scaled)
y_pred_proba = lr_model.predict_proba(X_test_scaled)

# Évaluation du modèle
print("\n" + "="*50)
print("ÉVALUATION DU MODÈLE LOGISTIC REGRESSION")
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Tiers', 'Tiers Plus', 'Tout Risque'],
            yticklabels=['Tiers', 'Tiers Plus', 'Tout Risque'])
plt.title('Matrice de Confusion - Logistic Regression')
plt.ylabel('Valeurs Réelles')
plt.xlabel('Prédictions')
plt.tight_layout()
plt.savefig('confusion_matrix_logistic.png', dpi=300, bbox_inches='tight')
plt.close()

# Importance des coefficients (équivalent de l'importance des features)
print("\n" + "="*50)
print("COEFFICIENTS DU MODÈLE")
print("="*50)

# Pour la régression logistique multiclasse, on prend la moyenne des valeurs absolues des coefficients
coef_importance = np.mean(np.abs(lr_model.coef_), axis=0)

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': coef_importance
}).sort_values('importance', ascending=False)

print(feature_importance)

# Visualisation de l'importance des coefficients
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title('Importance des Coefficients - Logistic Regression')
plt.xlabel('Importance (Moyenne des valeurs absolues des coefficients)')
plt.tight_layout()
plt.savefig('feature_importance_logistic.png', dpi=300, bbox_inches='tight')
plt.close()

# Test de quelques exemples avec probabilités
print("\n" + "="*50)
print("TESTS D'EXEMPLES AVEC PROBABILITÉS")
print("="*50)

examples = X_test_scaled[:10]
examples_pred = lr_model.predict(examples)
examples_proba = lr_model.predict_proba(examples)
examples_actual = y_test.head(10).values

print("\nComparaison prédictions vs réalité (10 premiers exemples):")
classes = lr_model.classes_
for i in range(len(examples)):
    max_proba = np.max(examples_proba[i])
    print(f"Exemple {i+1}: Prédit={examples_pred[i]} ({max_proba:.3f}), Réel={examples_actual[i]}, "
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
print("SAUVEGARDE DU MODÈLE LOGISTIC REGRESSION")
print("="*50)

joblib.dump(lr_model, 'insurance_model_logistic.pkl')
joblib.dump(scaler, 'scaler_logistic.pkl')
joblib.dump(le_marque, 'marque_encoder_logistic.pkl')
joblib.dump(le_type_vehicule, 'type_vehicule_encoder_logistic.pkl')

# Vérification que les fichiers sont bien créés
for filename in ['insurance_model_logistic.pkl', 'scaler_logistic.pkl', 
                'marque_encoder_logistic.pkl', 'type_vehicule_encoder_logistic.pkl']:
    if os.path.exists(filename):
        print(f"✅ Fichier sauvegardé : {filename}")
    else:
        print(f"❌ Échec de la sauvegarde : {filename}")

# Sauvegarde des informations pour l'application
model_info_logistic = {
    'features': features,
    'marques': list(le_marque.classes_),
    'types_vehicules': list(le_type_vehicule.classes_),
    'accuracy': accuracy,
    'model_type': 'Logistic Regression',
    'classes': list(lr_model.classes_)
}

joblib.dump(model_info_logistic, 'model_info_logistic.pkl')
if os.path.exists('model_info_logistic.pkl'):
    print("✅ Informations du modèle sauvegardées: model_info_logistic.pkl")
else:
    print("❌ Échec de la sauvegarde: model_info_logistic.pkl")

print("\n" + "="*50)
print("ENTRAÎNEMENT LOGISTIC REGRESSION TERMINÉ AVEC SUCCÈS!")
print("="*50)
print(f"Accuracy finale: {accuracy*100:.2f}%")
print("Fichiers générés :")
print("- insurance_model_logistic.pkl")
print("- scaler_logistic.pkl")
print("- marque_encoder_logistic.pkl")
print("- type_vehicule_encoder_logistic.pkl")
print("- model_info_logistic.pkl")
print("- confusion_matrix_logistic.png")
print("- feature_importance_logistic.png")