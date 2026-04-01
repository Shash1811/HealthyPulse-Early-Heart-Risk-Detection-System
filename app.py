from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the trained models and metadata
def load_models():
    models = {}
    metadata = {}
    
    try:
        # Load trained models
        model_files = {
            'logistic': 'logistic_model.pkl',
            'random_forest': 'random_forest_model.pkl',
            'svm': 'svm_model.pkl'
        }
        
        for name, filename in model_files.items():
            if os.path.exists(filename):
                models[name] = joblib.load(filename)
                print(f"✅ Loaded {name} model from {filename}")
            else:
                print(f"⚠️  {filename} not found, {name} model will be unavailable")
        
        # Load metadata
        if os.path.exists('model_metadata.pkl'):
            metadata = joblib.load('model_metadata.pkl')
            print(f"✅ Loaded model metadata")
        
        return models, metadata
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return {}, {}

models, metadata = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form
        
        # Convert to DataFrame with proper data types
        input_data = {
            'male': int(data.get('gender', 0)),
            'age': float(data.get('age', 0)),
            'education': int(data.get('education', 1)),
            'currentSmoker': int(data.get('currentSmoker', 0)),
            'cigsPerDay': float(data.get('cigsPerDay', 0)),
            'BPMeds': int(data.get('BPMeds', 0)),
            'prevalentStroke': int(data.get('prevalentStroke', 0)),
            'prevalentHyp': int(data.get('prevalentHyp', 0)),
            'diabetes': int(data.get('diabetes', 0)),
            'totChol': float(data.get('totChol', 0)),
            'sysBP': float(data.get('sysBP', 0)),
            'diaBP': float(data.get('diaBP', 0)),
            'BMI': float(data.get('BMI', 0)),
            'heartRate': float(data.get('heartRate', 0)),
            'glucose': float(data.get('glucose', 0))
        }
        
        # Add derived features
        input_data['cholesterol_risk'] = 1 if input_data['totChol'] > 240 else 0
        
        # Add age group features
        age = input_data['age']
        if age <= 40:
            input_data['age_group_Young'] = 1
            input_data['age_group_Middle'] = 0
            input_data['age_group_Senior'] = 0
        elif age <= 60:
            input_data['age_group_Young'] = 0
            input_data['age_group_Middle'] = 1
            input_data['age_group_Senior'] = 0
        else:
            input_data['age_group_Young'] = 0
            input_data['age_group_Middle'] = 0
            input_data['age_group_Senior'] = 1
        
        # Create DataFrame with all expected features
        if metadata and 'features' in metadata:
            # Use features from metadata
            all_features = metadata['features']
        else:
            # Fallback feature list
            all_features = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 
                          'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 
                          'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose',
                          'cholesterol_risk', 'age_group_Middle', 'age_group_Senior', 'age_group_Young']
        
        # Create DataFrame with all features, filling missing ones with 0
        df = pd.DataFrame([{feature: input_data.get(feature, 0) for feature in all_features}])
        
        # Get predictions from all available models
        predictions = {}
        for name, model in models.items():
            try:
                pred_proba = model.predict_proba(df)[0][1]
                pred_class = model.predict(df)[0]
                predictions[name] = {
                    'probability': float(pred_proba),
                    'prediction': int(pred_class),
                    'risk_level': 'High' if pred_proba > 0.5 else 'Low'
                }
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                predictions[name] = {
                    'error': str(e)
                }
        
        # Calculate ensemble prediction (average of all valid models)
        valid_probs = [pred['probability'] for pred in predictions.values() if 'probability' in pred]
        if valid_probs:
            ensemble_prob = np.mean(valid_probs)
            ensemble_prediction = 1 if ensemble_prob > 0.5 else 0
            ensemble_risk = 'High' if ensemble_prob > 0.5 else 'Low'
        else:
            ensemble_prob = 0.5
            ensemble_prediction = 0
            ensemble_risk = 'Low'
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'ensemble': {
                'probability': float(ensemble_prob),
                'prediction': ensemble_prediction,
                'risk_level': ensemble_risk
            },
            'input_data': input_data,
            'features_used': all_features
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
