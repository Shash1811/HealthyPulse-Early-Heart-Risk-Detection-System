import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def train_and_save_models():
    """
    Train the ML models using the same approach as in the notebook
    and save them for the web application
    """
    
    print("🚀 Training Heart Disease Risk Detection Models...")
    
    # Try to load the processed dataset
    try:
        df = pd.read_csv("processed_dataset.csv")
        print("✅ Loaded processed_dataset.csv")
    except FileNotFoundError:
        print("❌ processed_dataset.csv not found. Creating sample data...")
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'male': np.random.randint(0, 2, n_samples),
            'age': np.random.randint(18, 80, n_samples),
            'education': np.random.randint(1, 5, n_samples),
            'currentSmoker': np.random.randint(0, 2, n_samples),
            'cigsPerDay': np.random.randint(0, 40, n_samples),
            'BPMeds': np.random.randint(0, 2, n_samples),
            'prevalentStroke': np.random.randint(0, 2, n_samples),
            'prevalentHyp': np.random.randint(0, 2, n_samples),
            'diabetes': np.random.randint(0, 2, n_samples),
            'totChol': np.random.normal(200, 40, n_samples),
            'sysBP': np.random.normal(130, 20, n_samples),
            'diaBP': np.random.normal(80, 12, n_samples),
            'BMI': np.random.normal(25, 5, n_samples),
            'heartRate': np.random.normal(75, 10, n_samples),
            'glucose': np.random.normal(90, 25, n_samples),
            'TenYearCHD': np.random.randint(0, 2, n_samples)
        })
        
        # Add age_group feature
        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 100], labels=['Young', 'Middle', 'Senior'])
        df = pd.get_dummies(df, columns=['age_group'], drop_first=True)
        df['cholesterol_risk'] = (df['totChol'] > 240).astype(int)
        
        print("✅ Created sample dataset")
    
    # Drop rows with missing target
    df = df.dropna(subset=["TenYearCHD"])
    
    # Separate features and target
    X = df.drop("TenYearCHD", axis=1)
    y = df["TenYearCHD"]
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
    
    # Identify categorical and numerical features
    categorical_features = ['male', 'education', 'currentSmoker', 'BPMeds', 
                           'prevalentStroke', 'prevalentHyp', 'diabetes']
    
    # Find numerical features
    numerical_features = [col for col in X.columns if col not in categorical_features]
    
    print(f"🔢 Numerical features: {len(numerical_features)}")
    print(f"📋 Categorical features: {len(categorical_features)}")
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models
    models = {
        'logistic': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                penalty='l2',
                solver='liblinear',
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ))
        ]),
        
        'random_forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            ))
        ]),
        
        'svm': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', SVC(
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=42
            ))
        ])
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name.upper()} model...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = model.score(X_test, y_test)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"✅ {name.upper()} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        # Save model
        joblib.dump(model, f'{name}_model.pkl')
        print(f"💾 Saved {name}_model.pkl")
        
        results[name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'model': model
        }
    
    # Save feature names for the web app
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'feature_names.pkl')
    print("💾 Saved feature_names.pkl")
    
    # Save the preprocessor separately
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("💾 Saved preprocessor.pkl")
    
    # Create a model metadata file
    metadata = {
        'features': feature_names,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'target': 'TenYearCHD',
        'models': {name: {'accuracy': results[name]['accuracy'], 
                         'roc_auc': results[name]['roc_auc']} 
                  for name in results.keys()}
    }
    
    joblib.dump(metadata, 'model_metadata.pkl')
    print("💾 Saved model_metadata.pkl")
    
    print("\n🎉 Model training completed successfully!")
    print("\n📈 Model Performance Summary:")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name.upper():15} | Accuracy: {metrics['accuracy']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return results

if __name__ == "__main__":
    results = train_and_save_models()
