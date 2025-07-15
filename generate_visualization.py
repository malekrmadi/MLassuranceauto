import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Charger les résultats du grid search
comparison_info = joblib.load('grid_search_results.pkl')
results = comparison_info['results']
best_model_name = comparison_info['best_model']

# Charger les données de test pour la matrice de confusion
df = pd.read_csv('dataset_assurance_auto.csv')
features = comparison_info['features']

# Préparation des features
from sklearn.preprocessing import LabelEncoder
le_marque = LabelEncoder()
le_type_vehicule = LabelEncoder()

df['Marque_encoded'] = le_marque.fit_transform(df['Marque'])
df['Type_Vehicule_encoded'] = le_type_vehicule.fit_transform(df['Type_Vehicule'])
df['Place_Parking_int'] = df['Place_Parking'].astype(int)
df['Voiture_Entreprise_int'] = df['Voiture_Entreprise'].astype(int)

X = df[features]
y = df['Assurance_Recommandee']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Générer la visualisation
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
    best_model = results[best_model_name]['model']
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title(f'Importance des Features - {best_model_name}')
    plt.xlabel('Importance')
else:
    # Pour Logistic Regression, utiliser les coefficients
    best_model = results[best_model_name]['model']
    coef_importance = np.mean(np.abs(best_model.coef_), axis=0)
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': coef_importance
    }).sort_values('importance', ascending=False).head(10)
    
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title(f'Importance des Coefficients - {best_model_name}')
    plt.xlabel('Importance')

# Sous-graphique 4: Résumé des performances
plt.subplot(2, 2, 4)
performance_data = []
for model_name, result in results.items():
    performance_data.append({
        'Modèle': model_name,
        'CV Score': result['cv_score'],
        'Test Accuracy': result['test_accuracy']
    })

performance_df = pd.DataFrame(performance_data)
performance_df_melted = performance_df.melt(id_vars=['Modèle'], 
                                           value_vars=['CV Score', 'Test Accuracy'],
                                           var_name='Métrique', value_name='Score')

sns.barplot(data=performance_df_melted, x='Modèle', y='Score', hue='Métrique')
plt.title('Résumé des Performances')
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig('grid_search_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Graphique de comparaison généré: grid_search_comparison.png")
print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")
print(f"🎯 Accuracy: {results[best_model_name]['test_accuracy']:.4f} ({results[best_model_name]['test_accuracy']*100:.2f}%)") 