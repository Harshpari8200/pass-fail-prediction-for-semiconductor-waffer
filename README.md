# Semiconductor Wafer Pass/Fail Prediction

An AI-powered system for predicting semiconductor wafer quality control outcomes using machine learning.

## 🎯 Project Overview

This project develops an AI-based prediction model to classify semiconductor wafers as **Pass** or **Fail** during testing. The aim is to improve first-time yield rates by minimizing the number of wafers that require retesting.

## 📁 Project Structure

```
semiconductor-wafer-prediction/
├── main.py                          # Model training script
├── app.py                           # Streamlit web application
├── semiconductor_wafer_dataset.csv   # Training dataset (499 samples)
├── requirements.txt                  # Python dependencies
├── models/                          # Trained model artifacts
│   ├── wafer_model.pkl              # Trained Random Forest model
│   ├── scaler.pkl                   # Feature scaler
│   └── feature_names.pkl            # Feature column names
└── README.md                        # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)

The repository includes pre-trained models, but you can retrain if needed:

```bash
python main.py
```

### 3. Run the Web Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🔧 Features

### Input Parameters (9 Features):
- **Wafer Thickness** (μm): Physical dimension measurement
- **Surface Defect Count**: Number of defects per cm²
- **Resistance** (Ω): Electrical resistance measurement
- **Voltage** (V): Operating voltage
- **Temperature** (°C): Process temperature
- **Process Time** (min): Manufacturing duration
- **Chemical Concentration** (ppm): Impurity levels
- **Surface Roughness** (nm): Surface quality
- **Pattern Alignment Error** (nm): Photolithography accuracy

### Model Performance:
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 99.0%
- **Precision**: 98.7%
- **Recall**: 100.0%
- **F1-Score**: 99.3%

## 📊 Dataset Information

- **Total Samples**: 499 wafers
- **Features**: 9 manufacturing parameters
- **Target Classes**: Pass (74.7%) / Fail (25.3%)
- **Data Quality**: No missing values, realistic parameter ranges

## 🎮 Using the Web App

1. **Enter Parameters**: Input all 9 manufacturing parameters
2. **Get Prediction**: Click "Predict Wafer Quality"
3. **View Results**: See Pass/Fail prediction with confidence scores
4. **Review Details**: Check probability breakdown and input summary

## 🧠 Model Details

The system compares three algorithms:
- Logistic Regression
- Random Forest (selected)
- Gradient Boosting

**Feature Importance Ranking**:
1. Surface Defect Count (49.8%)
2. Pattern Alignment Error (18.7%)
3. Resistance (14.7%)
4. Surface Roughness (8.1%)
5. Temperature (4.9%)

## 📈 Applications

- **Quality Control**: Automated Pass/Fail classification
- **Yield Optimization**: Early defect detection
- **Process Improvement**: Parameter optimization insights
- **Cost Reduction**: Minimize retesting requirements

## 🔄 Retraining the Model

To retrain with new data:

1. Update `semiconductor_wafer_dataset.csv`
2. Run `python main.py`
3. Restart the Streamlit app

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🏆 Results Summary

✅ **99.0% Accuracy** - Production-ready model performance  
⚡ **Real-time Predictions** - Instant classification results  
🎯 **Industry-Standard Features** - Based on semiconductor manufacturing parameters  
🔧 **User-Friendly Interface** - Easy-to-use web application  
📊 **Comprehensive Metrics** - Detailed prediction confidence scores
