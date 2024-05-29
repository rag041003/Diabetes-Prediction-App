# Diabetes Prediction Project

This project is a web application designed to predict whether a patient is diabetic based on various medical attributes. The application is built using Flask for the backend and a machine learning model trained on a diabetes dataset.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model Training](#model-training)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [File Structure](#file-structure)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The Diabetes Prediction Project aims to provide a simple and intuitive web interface where users can input medical information and receive a prediction on their diabetes status. The machine learning model was trained using various classifiers and the best-performing model was selected for deployment.

## Dataset

The dataset used in this project is the Pima Indians Diabetes Database, which is publicly available and contains medical data for female patients of Pima Indian heritage. The dataset includes features such as the number of pregnancies, glucose levels, blood pressure, skin thickness, insulin levels, BMI, diabetes pedigree function, and age.

## Features

- **Machine Learning Model**: Trained using several classifiers including Random Forest, SVM, KNN, Logistic Regression, and a Stacking Classifier.
- **Web Interface**: Built with Flask, allowing users to input data and receive predictions.
- **Data Processing**: Includes handling of missing values, feature scaling, and polynomial feature transformation.

## Model Training

The model training process includes:
1. Loading and cleaning the dataset.
2. Handling missing values using a simple imputer.
3. Creating polynomial features for better model performance.
4. Balancing the dataset using SMOTE.
5. Splitting the data into training and testing sets.
6. Training multiple models and selecting the best one based on accuracy.
7. Saving the trained model and scaler using joblib.

## Setup Instructions

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
# Diabetes Prediction Project

This project is a web application designed to predict whether a patient is diabetic based on various medical attributes. The application is built using Flask for the backend and a machine learning model trained on a diabetes dataset.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model Training](#model-training)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Acknowledgments](#acknowledgments)

## Overview

The Diabetes Prediction Project aims to provide a simple and intuitive web interface where users can input medical information and receive a prediction on their diabetes status. The machine learning model was trained using various classifiers and the best-performing model was selected for deployment.

## Dataset

The dataset used in this project is the Pima Indians Diabetes Database, which is publicly available and contains medical data for female patients of Pima Indian heritage. The dataset includes features such as the number of pregnancies, glucose levels, blood pressure, skin thickness, insulin levels, BMI, diabetes pedigree function, and age.

## Features

- **Machine Learning Model**: Trained using several classifiers including Random Forest, SVM, KNN, Logistic Regression, and a Stacking Classifier.
- **Web Interface**: Built with Flask, allowing users to input data and receive predictions.
- **Data Processing**: Includes handling of missing values, feature scaling, and polynomial feature transformation.

## Model Training

The model training process includes:
1. Loading and cleaning the dataset.
2. Handling missing values using a simple imputer.
3. Creating polynomial features for better model performance.
4. Balancing the dataset using SMOTE.
5. Splitting the data into training and testing sets.
6. Training multiple models and selecting the best one based on accuracy.
7. Saving the trained model and scaler using joblib.

## Setup Instructions

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
   
2. **Create and activate a virtual environment:**
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`

3. **Install the required dependencies:**
    pip install -r requirements.txt

4. **Run the Flask application:**
    python app.py

## Usage

Once the Flask application is running, open your web browser and navigate to 
http://127.0.0.1:5000/. You will see a form where you can input the required medical 
information. After submitting the form, the application will display whether the 
patient is diabetic or not.

## File Structure

diabetes-prediction/
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── app.py
├── train_model.py
├── best_model.pkl
├── scaler.pkl
├── poly_transformer.pkl
├── requirements.txt
└── README.md

## Acknowledgments
The dataset used in this project is provided by the National Institute of Diabetes and Digestive and Kidney Diseases.
The project structure and deployment were inspired by various open-source projects and tutorials.
