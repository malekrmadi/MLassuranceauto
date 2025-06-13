import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Charger le dataset synthétique
df = pd.read_csv('assurance_synthetique_1000.csv')

# Encoder les colonnes catégorielles
label_encoders = {}
for col in ['prix_vehicule_categorie', 'parking', 'profil_conducteur', 'valeur_vehicule', 'assurance_recommandee']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features et label
X = df.drop('assurance_recommandee', axis=1)
y = df['assurance_recommandee']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle RandomForest
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Sauvegarde modèle + encodeurs
with open('model_assurance.pkl', 'wb') as f:
    pickle.dump((model, label_encoders), f)

print("✅ Modèle entraîné et sauvegardé dans model_assurance.pkl")
