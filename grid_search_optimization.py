import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🔍 GRID SEARCH OPTIMIZATION FOR INSURANCE MODELS")
print("="*60)

# Chargement des données
print("📊 Chargement des données...")
df = pd.read_csv('dataset_assurance_auto.csv')

# Préparation des features
print("🔧 Préparation des features...")

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

print(f"📈 Taille du jeu d'entraînement: {X_train.shape[0]}")
print(f"📊 Taille du jeu de test: {X_test.shape[0]}")

# Normalisation pour Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 1. GRID SEARCH POUR RANDOM FOREST
# ============================================================================
print("\n" + "="*60)
print("🌲 GRID SEARCH - RANDOM FOREST")
print("="*60)

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf_grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("🔍 Recherche des meilleurs hyperparamètres pour Random Forest...")
rf_grid_search.fit(X_train, y_train)

print(f"✅ Meilleurs paramètres Random Forest: {rf_grid_search.best_params_}")
print(f"🎯 Meilleur score CV: {rf_grid_search.best_score_:.4f}")

# Test du meilleur modèle Random Forest
best_rf = rf_grid_search.best_estimator_
rf_pred = best_rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"📊 Accuracy sur le test set: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")

# ============================================================================
# 2. GRID SEARCH POUR DECISION TREE
# ============================================================================
print("\n" + "="*60)
print("🌳 GRID SEARCH - DECISION TREE")
print("="*60)

dt_param_grid = {
    'max_depth': [3, 5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'class_weight': ['balanced', None]
}

dt_grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("🔍 Recherche des meilleurs hyperparamètres pour Decision Tree...")
dt_grid_search.fit(X_train, y_train)

print(f"✅ Meilleurs paramètres Decision Tree: {dt_grid_search.best_params_}")
print(f"🎯 Meilleur score CV: {dt_grid_search.best_score_:.4f}")

# Test du meilleur modèle Decision Tree
best_dt = dt_grid_search.best_estimator_
dt_pred = best_dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"📊 Accuracy sur le test set: {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")

# ============================================================================
# 3. GRID SEARCH POUR LOGISTIC REGRESSION
# ============================================================================
print("\n" + "="*60)
print("📈 GRID SEARCH - LOGISTIC REGRESSION")
print("="*60)

lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'class_weight': ['balanced', None],
    'max_iter': [1000, 2000]
}

lr_grid_search = GridSearchCV(
    LogisticRegression(random_state=42, multi_class='ovr'),
    lr_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("🔍 Recherche des meilleurs hyperparamètres pour Logistic Regression...")
lr_grid_search.fit(X_train_scaled, y_train)

print(f"✅ Meilleurs paramètres Logistic Regression: {lr_grid_search.best_params_}")
print(f"🎯 Meilleur score CV: {lr_grid_search.best_score_:.4f}")

# Test du meilleur modèle Logistic Regression
best_lr = lr_grid_search.best_estimator_
lr_pred = best_lr.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_pred)
print(f"📊 Accuracy sur le test set: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")

# ============================================================================
# COMPARAISON DES MODÈLES OPTIMISÉS
# ============================================================================
print("\n" + "="*60)
print("🏆 COMPARAISON DES MODÈLES OPTIMISÉS")
print("="*60)

results = {
    'Random Forest': {
        'cv_score': rf_grid_search.best_score_,
        'test_accuracy': rf_accuracy,
        'model': best_rf,
        'predictions': rf_pred
    },
    'Decision Tree': {
        'cv_score': dt_grid_search.best_score_,
        'test_accuracy': dt_accuracy,
        'model': best_dt,
        'predictions': dt_pred
    },
    'Logistic Regression': {
        'cv_score': lr_grid_search.best_score_,
        'test_accuracy': lr_accuracy,
        'model': best_lr,
        'predictions': lr_pred
    }
}

# Affichage des résultats
print("\n📊 RÉSULTATS COMPARATIFS:")
print("-" * 50)
print(f"{'Modèle':<20} {'CV Score':<12} {'Test Accuracy':<15} {'Amélioration':<15}")
print("-" * 50)

# Scores de base (approximatifs basés sur vos scripts)
base_scores = {
    'Random Forest': 0.85,  # Estimation basée sur votre script
    'Decision Tree': 0.80,  # Estimation basée sur votre script
    'Logistic Regression': 0.82  # Estimation basée sur votre script
}

for model_name, result in results.items():
    improvement = result['test_accuracy'] - base_scores.get(model_name, 0)
    print(f"{model_name:<20} {result['cv_score']:<12.4f} {result['test_accuracy']:<15.4f} {improvement:+.4f}")

# Trouver le meilleur modèle
best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
best_model = results[best_model_name]['model']

print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
print(f"🎯 Accuracy: {results[best_model_name]['test_accuracy']:.4f} ({results[best_model_name]['test_accuracy']*100:.2f}%)")

# ============================================================================
# SAUVEGARDE DES MODÈLES OPTIMISÉS
# ============================================================================
print("\n" + "="*60)
print("💾 SAUVEGARDE DES MODÈLES OPTIMISÉS")
print("="*60)

# Sauvegarde du meilleur Random Forest
joblib.dump(best_rf, 'insurance_model_rf_optimized.pkl')
print("✅ Random Forest optimisé sauvegardé: insurance_model_rf_optimized.pkl")

# Sauvegarde du meilleur Decision Tree
joblib.dump(best_dt, 'insurance_model_dt_optimized.pkl')
print("✅ Decision Tree optimisé sauvegardé: insurance_model_dt_optimized.pkl")

# Sauvegarde du meilleur Logistic Regression
joblib.dump(best_lr, 'insurance_model_lr_optimized.pkl')
joblib.dump(scaler, 'scaler_lr_optimized.pkl')
print("✅ Logistic Regression optimisé sauvegardé: insurance_model_lr_optimized.pkl")
print("✅ Scaler optimisé sauvegardé: scaler_lr_optimized.pkl")

# Sauvegarde des encodeurs
joblib.dump(le_marque, 'marque_encoder_optimized.pkl')
joblib.dump(le_type_vehicule, 'type_vehicule_encoder_optimized.pkl')
print("✅ Encodeurs sauvegardés")

# Sauvegarde des informations de comparaison
comparison_info = {
    'best_model': best_model_name,
    'results': results,
    'features': features,
    'marques': list(le_marque.classes_),
    'types_vehicules': list(le_type_vehicule.classes_),
    'grid_search_results': {
        'random_forest': {
            'best_params': rf_grid_search.best_params_,
            'best_score': rf_grid_search.best_score_,
            'cv_results': rf_grid_search.cv_results_
        },
        'decision_tree': {
            'best_params': dt_grid_search.best_params_,
            'best_score': dt_grid_search.best_score_,
            'cv_results': dt_grid_search.cv_results_
        },
        'logistic_regression': {
            'best_params': lr_grid_search.best_params_,
            'best_score': lr_grid_search.best_score_,
            'cv_results': lr_grid_search.cv_results_
        }
    }
}

joblib.dump(comparison_info, 'grid_search_results.pkl')
print("✅ Résultats de grid search sauvegardés: grid_search_results.pkl")

# ============================================================================
# VISUALISATION DES RÉSULTATS
# ============================================================================
print("\n" + "="*60)
print("📊 VISUALISATION DES RÉSULTATS")
print("="*60)

# Graphique de comparaison des accuracies
plt.figure(figsize=(12, 8))

# Sous-graphique 1: Comparaison des accuracies
plt.subplot(2, 2, 1)
models = list(results.keys())
cv_scores = [results[model]['cv_score'] for model in models]
test_accuracies = [results[model]['test_accuracy'] for model in models]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, cv_scores, width, label='CV Score', alpha=0.8)
plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy', alpha=0.8)

plt.xlabel('Modèles')
plt.ylabel('Accuracy')
plt.title('Comparaison des Performances')
plt.xticks(x, models, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Sous-graphique 2: Matrice de confusion du meilleur modèle
plt.subplot(2, 2, 2)
best_predictions = results[best_model_name]['predictions']
cm = confusion_matrix(y_test, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Tiers', 'Tiers Plus', 'Tout Risque'],
            yticklabels=['Tiers', 'Tiers Plus', 'Tout Risque'])
plt.title(f'Matrice de Confusion - {best_model_name}')
plt.ylabel('Valeurs Réelles')
plt.xlabel('Prédictions')

# Sous-graphique 3: Importance des features (pour Random Forest et Decision Tree)
plt.subplot(2, 2, 3)
if best_model_name in ['Random Forest', 'Decision Tree']:
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title(f'Importance des Features - {best_model_name}')
    plt.xlabel('Importance')
else:
    # Pour Logistic Regression, utiliser les coefficients
    coef_importance = np.mean(np.abs(best_model.coef_), axis=0)
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': coef_importance
    }).sort_values('importance', ascending=False).head(10)
    
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title(f'Importance des Coefficients - {best_model_name}')
    plt.xlabel('Importance')

# Sous-graphique 4: Évolution des scores CV
plt.subplot(2, 2, 4)
# Trouver la longueur minimale pour éviter l'erreur
min_length = min(len(rf_grid_search.cv_results_['mean_test_score']),
                len(dt_grid_search.cv_results_['mean_test_score']),
                len(lr_grid_search.cv_results_['mean_test_score']))

cv_evolution = pd.DataFrame({
    'Random Forest': rf_grid_search.cv_results_['mean_test_score'][:min_length],
    'Decision Tree': dt_grid_search.cv_results_['mean_test_score'][:min_length],
    'Logistic Regression': lr_grid_search.cv_results_['mean_test_score'][:min_length]
})

cv_evolution.plot(kind='line', ax=plt.gca())
plt.title('Évolution des Scores CV')
plt.xlabel('Itération de Grid Search')
plt.ylabel('Score CV')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('grid_search_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Graphique de comparaison sauvegardé: grid_search_comparison.png")

# ============================================================================
# RAPPORT FINAL
# ============================================================================
print("\n" + "="*60)
print("📋 RAPPORT FINAL")
print("="*60)

print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
print(f"🎯 Accuracy finale: {results[best_model_name]['test_accuracy']:.4f} ({results[best_model_name]['test_accuracy']*100:.2f}%)")

print(f"\n📊 RAPPORT DE CLASSIFICATION - {best_model_name}:")
print(classification_report(y_test, results[best_model_name]['predictions']))

print(f"\n🔧 MEILLEURS HYPERPARAMÈTRES:")
if best_model_name == 'Random Forest':
    print(f"Random Forest: {rf_grid_search.best_params_}")
elif best_model_name == 'Decision Tree':
    print(f"Decision Tree: {dt_grid_search.best_params_}")
else:
    print(f"Logistic Regression: {lr_grid_search.best_params_}")

print(f"\n💾 FICHIERS GÉNÉRÉS:")
print("- insurance_model_rf_optimized.pkl")
print("- insurance_model_dt_optimized.pkl") 
print("- insurance_model_lr_optimized.pkl")
print("- scaler_lr_optimized.pkl")
print("- marque_encoder_optimized.pkl")
print("- type_vehicule_encoder_optimized.pkl")
print("- grid_search_results.pkl")
print("- grid_search_comparison.png")

print("\n" + "="*60)
print("✅ GRID SEARCH OPTIMIZATION TERMINÉ AVEC SUCCÈS!")
print("="*60) 