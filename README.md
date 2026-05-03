# 🌊 Flood Prediction Web App using Machine Learning

## 📌 Project Overview

This project is a Machine Learning-based Flood Prediction Web Application built using **Streamlit**.  
It predicts flood risk based on geographical and environmental features such as elevation, slope, and other terrain parameters.

The application processes DEM (Digital Elevation Model) data and uses a trained ML model to classify flood-prone areas.

---

## 🛠️ Technologies Used

- Python 3.11
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Rasterio
- Matplotlib
- Joblib

---

## 📂 Project Structure

```
ML_project/
│
├── data/                     # Dataset files
├── geo_features/             # Feature engineering scripts
├── rasters_SRTMGL1/          # DEM raster files
├── models/                   # Trained ML models
├── streamlit_app.py          # Main Streamlit application
├── flood_pred.py             # ML prediction logic
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Installation & Setup

### Step 1: Clone the repository

```
git clone <your-repository-link>
cd ML_project
```

### Step 2: Install dependencies

```
pip install -r requirements.txt
```

### Step 3: Run the application

```
streamlit run streamlit_app.py
```

The app will open in your browser automatically.

---

## 🎯 Features

- Upload flood dataset
- DEM processing
- Feature extraction
- ML-based flood risk prediction
- Interactive Streamlit UI
- Visualization of results

---

## 📊 Machine Learning Model

The project uses a trained classification model built with **Scikit-learn** to predict flood-prone areas based on extracted geographical features.

---

## 👩‍💻 Author

Your Name  
B.Tech Project – Flood Prediction using Machine Learning  
Year: 2026
