from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Charger modèles et encodeurs (à adapter selon tes chemins)
model = joblib.load('insurance_model.pkl')
marque_encoder = joblib.load('marque_encoder.pkl')
type_vehicule_encoder = joblib.load('type_vehicule_encoder.pkl')

marques = ['Dacia', 'Fiat', 'Hyundai', 'Kia', 'Chevrolet', 'Chery',
           'Renault', 'Peugeot', 'Citroën', 'Volkswagen', 'Ford', 'Toyota', 'Nissan',
           'Audi', 'BMW', 'Mercedes', 'Volvo', 'Lexus', 'Infiniti',
           'Porsche', 'Jaguar', 'Land Rover', 'Maserati', 'Bentley', 'Ferrari', 'Lamborghini']

types_vehicules = ['Citadine', 'Voiture populaire', 'Berline compact', 'Break familial',
                   'SUV', 'Berline executive', 'Voiture de luxe', 'Sportive']

def encode_feature(value, encoder, valid_list):
    if value not in valid_list:
        raise ValueError(f"Valeur invalide: {value}")
    return encoder.transform([value])[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        marque = data['marque']
        type_vehicule = data['type_vehicule']
        age_vehicule = int(data['age_vehicule'])
        valeur_vehicule = float(data['valeur_vehicule'])
        kilometrage = float(data['kilometrage'])
        chevaux_fiscaux = int(data['chevaux_fiscaux'])
        place_parking = bool(data['place_parking'])
        voiture_entreprise = bool(data['voiture_entreprise'])
        age_conducteur = int(data['age_conducteur'])
        annees_experience = int(data['annees_experience'])
        nombre_accidents = int(data['nombre_accidents'])

        # Encodage
        marque_encoded = encode_feature(marque, marque_encoder, marques)
        type_vehicule_encoded = encode_feature(type_vehicule, type_vehicule_encoder, types_vehicules)

        features_vector = np.array([[marque_encoded, type_vehicule_encoded, age_vehicule,
                                     valeur_vehicule, kilometrage, chevaux_fiscaux,
                                     int(place_parking), int(voiture_entreprise),
                                     age_conducteur, annees_experience, nombre_accidents]])

        prediction = model.predict(features_vector)[0]

        return jsonify({
            'assurance_recommandee': prediction
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
