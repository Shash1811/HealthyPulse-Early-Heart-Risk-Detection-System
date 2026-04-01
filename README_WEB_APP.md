# HealthyPulse Web Application

An attractive and modern web interface for AI-powered heart disease risk detection, integrating your trained machine learning models.

## 🌟 Features

### **Beautiful Modern UI**
- Responsive design with gradient backgrounds and smooth animations
- Bootstrap 5 with custom styling for a professional healthcare application
- Interactive form with comprehensive health metrics input
- Real-time risk assessment visualization

### **Integrated ML Models**
- **Logistic Regression**: Fast and interpretable baseline model
- **Random Forest**: Ensemble method with robust performance  
- **SVM**: Support Vector Machine with RBF kernel
- **Ensemble Prediction**: Combined model predictions for improved accuracy

### **User Experience**
- Intuitive health assessment form organized by categories:
  - Basic Information (age, gender, education)
  - Lifestyle Factors (smoking, medications)
  - Medical History (stroke, hypertension, diabetes)
  - Clinical Measurements (cholesterol, blood pressure, BMI, etc.)
- Real-time loading indicators and smooth animations
- Comprehensive results display with risk probabilities
- Individual model performance comparison

## 🚀 Quick Start

### **Prerequisites**
```bash
pip install -r requirements.txt
```

### **Train Models**
```bash
python train_models.py
```
This will:
- Load your processed dataset (or create sample data if not available)
- Train Logistic Regression, Random Forest, and SVM models
- Save trained models as pickle files
- Generate feature metadata for the web application

### **Run Web Application**
```bash
python app.py
```

The application will be available at:
- **Local**: http://127.0.0.1:5000
- **Network**: http://192.168.31.84:5000 (or your local IP)

## 📊 Model Integration

### **Features Used**
The web application accepts the following health metrics:

**Basic Information:**
- Gender (Male/Female)
- Age (years)
- Education Level

**Lifestyle Factors:**
- Current Smoker Status
- Cigarettes per Day
- Blood Pressure Medication

**Medical History:**
- Prevalent Stroke
- Prevalent Hypertension  
- Diabetes Status

**Clinical Measurements:**
- Total Cholesterol (mg/dL)
- Systolic Blood Pressure (mmHg)
- Diastolic Blood Pressure (mmHg)
- BMI (Body Mass Index)
- Heart Rate (bpm)
- Glucose (mg/dL)

### **Feature Engineering**
The application automatically creates derived features:
- **Cholesterol Risk**: Binary flag if total cholesterol > 240 mg/dL
- **Age Groups**: One-hot encoded age categories (Young, Middle, Senior)

### **Prediction Pipeline**
1. User inputs health data through the web form
2. Data is validated and preprocessed
3. Features are engineered automatically
4. Each trained model makes predictions
5. Ensemble prediction combines all model outputs
6. Results displayed with risk probabilities and visualizations

## 🎨 Web Interface Design

### **Visual Elements**
- **Gradient Header**: Eye-catching purple gradient with animated heartbeat icon
- **Card-Based Layout**: Organized form sections with hover effects
- **Progress Bars**: Visual representation of risk probabilities
- **Color-Coded Results**: Green for low risk, red for high risk
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile

### **User Flow**
1. **Landing Page**: Attractive header with application introduction
2. **Assessment Form**: Comprehensive health data collection
3. **Loading State**: Animated spinner during model processing
4. **Results Display**: 
   - Ensemble prediction with risk level
   - Individual model performance comparison
   - Probability visualization

### **Technical Stack**
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **Backend**: Flask (Python web framework)
- **ML Integration**: Scikit-learn pipelines with joblib model persistence
- **Styling**: Custom CSS animations and gradient effects

## 🔧 Technical Implementation

### **File Structure**
```
├── app.py                 # Flask web application
├── train_models.py        # Model training script
├── templates/
│   └── index.html        # Main web interface
├── *_model.pkl           # Trained model files
├── model_metadata.pkl    # Feature and model information
├── feature_names.pkl     # Expected input features
└── preprocessor.pkl      # Data preprocessing pipeline
```

### **API Endpoints**
- `GET /`: Main web interface
- `POST /predict`: Model prediction endpoint
- `GET /health`: Application health check

### **Model Loading**
Models are loaded at startup from pickle files:
```python
models = {
    'logistic': joblib.load('logistic_model.pkl'),
    'random_forest': joblib.load('random_forest_model.pkl'), 
    'svm': joblib.load('svm_model.pkl')
}
```

## 📈 Model Performance

The trained models achieve the following performance metrics:

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~0.49 | ~0.43 |
| Random Forest | ~0.48 | ~0.46 |
| SVM | ~0.44 | ~0.57 |

*Note: Performance shown is from sample data. Your actual trained models may have different metrics.*

## 🌐 Deployment

### **Development**
```bash
python app.py
```

### **Production**
For production deployment, consider:
- Using a WSGI server (Gunicorn, uWSGI)
- Adding HTTPS/TLS encryption
- Implementing user authentication
- Setting up database logging
- Containerizing with Docker

### **Environment Variables**
Create a `.env` file for configuration:
```env
FLASK_ENV=production
SECRET_KEY=your-secret-key
DEBUG=False
```

## 🔒 Security Considerations

- Input validation for all health metrics
- Sanitization of user inputs
- Error handling without exposing system details
- HTTPS recommended for production
- No persistent storage of PHI (Protected Health Information)

## 🚀 Future Enhancements

### **Planned Features**
- User authentication and profile management
- Historical predictions tracking
- Additional ML model integration (Neural Networks)
- Export results to PDF/medical records
- Multi-language support
- Mobile app version

### **Technical Improvements**
- Real-time model retraining
- A/B testing framework for model comparison
- Advanced feature engineering
- Integration with electronic health records (EHR)
- Telemedicine API integration

## 📞 Support

For questions or issues:
1. Check the model training logs
2. Verify all pickle files exist
3. Ensure Flask dependencies are installed
4. Test with sample data first

## 🏥 Medical Disclaimer

This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

---

**HealthyPulse** - Making heart disease risk assessment accessible through AI and modern web technology. ❤️
