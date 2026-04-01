# AI-Powered Heart Disease Risk Detection

An **AI-powered machine learning pipeline** for **early detection of heart disease risk**.  
This project walks through the complete workflow:  
 **Raw Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Prediction**

---

# Why This Project?
Heart disease is one of the **leading global health risks**.  
This pipeline helps researchers & healthcare professionals **detect risk early** by leveraging:

✅ Cleaned and standardized datasets  
✅ OCR-based document extraction  
✅ Feature engineering for better insights  
✅ Machine learning models for prediction  

---

# Workflow Overview

| Step | What Happens | Run This |
|------|--------------|----------|
| 1.  Data Upload | Add dataset (`framingham.csv`) + OCR docs to `/data/raw/` | — |
| 2.  Exploratory Data Analysis | Visualize data, spot trends | `notebooks/framingham_day2.ipynb` |
| 3.  Data Cleaning | Handle missing values & outliers | `src/data/preprocess.py` |
| 4.  Feature Engineering | Build informative features | `notebooks/day5_feature_engineering.ipynb` |
| 5.  Data Splitting | Train / Validation / Test sets | `src/pipeline/` |
| 6.  OCR Pipeline | Extract text from medical reports | `src/ocr/ocr_pipeline.py` |
| 7.  Train Models | Logistic Regression, Decision Tree, Random Forest | `notebooks/day6_modeling.ipynb` |
| 8.  Evaluate Models | Confusion Matrix, ROC, AUC | `notebooks/day6_modeling.ipynb` |
| 9.  Prediction | Predict on unseen test data | `src/app/predict.py` |

---

## 📁 Project Structure

heart-risk-day1-starter/
│── data/
│ ├── raw/ #  Original dataset (framingham.csv) & OCR reports
│ ├── interim/ #  Cleaned intermediate data
│ └── processed/ #  Final datasets ready for modeling
│
│── docs/ # Project documentation & reports
│── models/ # Trained machine learning models
│── notebooks/ # Jupyter/Colab notebooks (Day 2–6 tasks)
│── reports/
│ ├── figures/ # Confusion matrix, ROC curves, plots
│ └── eda/ # EDA visualizations
│
│── src/
│ ├── data/ # Data preprocessing scripts
│ ├── features/ # Feature engineering
│ ├── models/ # Model training & evaluation
│ ├── pipeline/ # ML pipeline scripts
│ └── ocr/ # OCR utilities for scanned reports
│
│── tests/ # Unit tests
│── .env # Environment variables (Tesseract path, etc.)
│── requirements.txt # Project dependencies
│── README.md # Project documentation (this file)
│── .gitignore # Ignore unnecessary files in Git

yaml
Copy
Edit

---

## 🛠 Tech Stack

-  **Python 3.9+**
-  **Pandas, NumPy** (Data Processing)
-  **Matplotlib, Seaborn** (Visualization)
-  **Scikit-learn** (ML Models)
-  **Pytesseract, OpenCV, Pillow** (OCR Extraction)
-  **Imbalanced-learn** (Handle class imbalance)

---

##  Example Outputs

✔️ Clean dataset with engineered features  
✔️ Model evaluation reports (Accuracy, Precision, Recall, ROC-AUC)  
✔️ OCR-extracted values from medical reports  
✔️ Predictions for unseen patient data  

---

##  Repository

🔗 [GitHub Repository](https://github.com/your-username/heart-risk-prediction)

---

## 🤝 Contributing

Want to improve this project? Follow these steps:

1. **Fork the repo**  
2. **Create your branch**  
   ```bash
   git checkout -b feature-name