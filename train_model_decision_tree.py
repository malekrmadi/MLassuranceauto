import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
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

# Entraînement du modèle Decision Tree
print("\nEntraînement du modèle Decision Tree...")
dt_model = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced',
    criterion='gini'  # Ou 'entropy' pour l'information gain
)

dt_model.fit(X_train, y_train)

# Prédictions
y_pred = dt_model.predict(X_test)

# Évaluation du modèle
print("\n" + "="*50)
print("ÉVALUATION DU MODÈLE DECISION TREE")
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Tiers', 'Tiers Plus', 'Tout Risque'],
            yticklabels=['Tiers', 'Tiers Plus', 'Tout Risque'])
plt.title('Matrice de Confusion - Decision Tree')
plt.ylabel('Valeurs Réelles')
plt.xlabel('Prédictions')
plt.tight_layout()
plt.savefig('confusion_matrix_decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()

# Importance des features
print("\n" + "="*50)
print("IMPORTANCE DES FEATURES - DECISION TREE")
print("="*50)

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# Visualisation de l'importance des features
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='autumn')
plt.title('Importance des Features - Decision Tree')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance_decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualisation de l'arbre de décision (première partie seulement)
print("\nGénération de la visualisation de l'arbre...")
plt.figure(figsize=(20, 12))
plot_tree(dt_model, 
          feature_names=features,
          class_names=['Tiers', 'Tiers Plus', 'Tout Risque'],
          filled=True,
          max_depth=3,  # Limité pour la lisibilité
          fontsize=8)
plt.title('Arbre de Décision (3 premiers niveaux)')
plt.tight_layout()
plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# Export des règles de décision en texte
print("\n" + "="*50)
print("RÈGLES DE DÉCISION (5 premiers niveaux)")
print("="*50)

tree_rules = export_text(dt_model, 
                        feature_names=features,
                        max_depth=5)
print(tree_rules[:2000] + "..." if len(tree_rules) > 2000 else tree_rules)

# Sauvegarde des règles complètes dans un fichier
with open('decision_tree_rules.txt', 'w', encoding='utf-8') as f:
    f.write(tree_rules)
print("\n✅ Règles complètes sauvegardées dans: decision_tree_rules.txt")

# Statistiques de l'arbre
print("\n" + "="*50)
print("STATISTIQUES DE L'ARBRE")
print("="*50)

print(f"Profondeur maximale de l'arbre: {dt_model.get_depth()}")
print(f"Nombre de feuilles: {dt_model.get_n_leaves()}")
print(f"Nombre total de nœuds: {dt_model.tree_.node_count}")

# Test de quelques exemples
print("\n" + "="*50)
print("TESTS D'EXEMPLES")
print("="*50)

examples = X_test.head(10)
examples_pred = dt_model.predict(examples)
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

# Analyse des chemins de décision pour quelques exemples
print("\n" + "="*50)
print("CHEMINS DE DÉCISION (3 premiers exemples)")
print("="*50)

decision_paths = dt_model.decision_path(X_test.head(3))
leaf_ids = dt_model.apply(X_test.head(3))

for sample_id in range(3):
    node_indicator = decision_paths[sample_id]
    leaf_id = leaf_ids[sample_id]
    
    print(f"\nExemple {sample_id + 1}:")
    print(f"Prédiction: {y_pred[sample_id]}")
    print(f"Feuille atteinte: {leaf_id}")
    
    # Affichage du chemin simplifié
    node_index = node_indicator.indices
    print(f"Chemin: {' -> '.join(map(str, node_index[:5]))}{'...' if len(node_index) > 5 else ''}")

# Sauvegarde du modèle et des encodeurs
print("\n" + "="*50)
print("SAUVEGARDE DU MODÈLE DECISION TREE")
print("="*50)

joblib.dump(dt_model, 'insurance_model_decision_tree.pkl')
joblib.dump(le_marque, 'marque_encoder_decision_tree.pkl')
joblib.dump(le_type_vehicule, 'type_vehicule_encoder_decision_tree.pkl')

# Vérification que les fichiers sont bien créés
for filename in ['insurance_model_decision_tree.pkl', 
                'marque_encoder_decision_tree.pkl', 
                'type_vehicule_encoder_decision_tree.pkl']:
    if os.path.exists(filename):
        print(f"✅ Fichier sauvegardé : {filename}")
    else:
        print(f"❌ Échec de la sauvegarde : {filename}")

# Sauvegarde des informations pour l'application
model_info_dt = {
    'features': features,
    'marques': list(le_marque.classes_),
    'types_vehicules': list(le_type_vehicule.classes_),
    'accuracy': accuracy,
    'model_type': 'Decision Tree',
    'tree_depth': dt_model.get_depth(),
    'n_leaves': dt_model.get_n_leaves(),
    'n_nodes': dt_model.tree_.node_count
}

joblib.dump(model_info_dt, 'model_info_decision_tree.pkl')
if os.path.exists('model_info_decision_tree.pkl'):
    print("✅ Informations du modèle sauvegardées: model_info_decision_tree.pkl")
else:
    print("❌ Échec de la sauvegarde: model_info_decision_tree.pkl")

print("\n" + "="*50)
print("ENTRAÎNEMENT DECISION TREE TERMINÉ AVEC SUCCÈS!")
print("="*50)
print(f"Accuracy finale: {accuracy*100:.2f}%")
print(f"Profondeur de l'arbre: {dt_model.get_depth()}")
print(f"Nombre de feuilles: {dt_model.get_n_leaves()}")
print("\nFichiers générés :")
print("- insurance_model_decision_tree.pkl")
print("- marque_encoder_decision_tree.pkl")
print("- type_vehicule_encoder_decision_tree.pkl")
print("- model_info_decision_tree.pkl")
print("- confusion_matrix_decision_tree.png")
print("- feature_importance_decision_tree.png")
print("- decision_tree_visualization.png")
print("- decision_tree_rules.txt")