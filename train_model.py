import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

# Load the dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Feature Engineering: Adding polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df_imputed.drop(columns=['Outcome']))
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(df_imputed.drop(columns=['Outcome']).columns))

# Splitting the data into features and target variable
X = X_poly_df
y = df_imputed['Outcome']

# Handling class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier()
}

# Initialize base models for stacking
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('log_reg', LogisticRegression(max_iter=2000))
]

# Initialize stacking classifier with logistic regression as meta-classifier
stacking_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())

# Train and evaluate models
best_model = None
best_accuracy = 0
best_model_name = ""

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{model_name} Accuracy: {accuracy}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = model_name
        best_y_pred = y_pred

# Fit the stacking model on the augmented data
stacking_model.fit(X_train_scaled, y_train)

# Predictions using stacking model
y_pred_stacking = stacking_model.predict(X_test_scaled)

# Evaluate accuracy of the stacking model
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
print(f"\nStacking Model Accuracy: {accuracy_stacking}")

# Check if the stacking model has the best accuracy
if accuracy_stacking > best_accuracy:
    best_accuracy = accuracy_stacking
    best_model = stacking_model
    best_model_name = "Stacking Model"
    best_y_pred = y_pred_stacking

# Save the polynomial transformer, scaler, and best model
joblib.dump(poly, 'poly_transformer.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(best_model, 'best_model.pkl')

# Generate and print the classification report and confusion matrix for the best model
best_classification_report = classification_report(y_test, best_y_pred)
best_confusion_matrix = confusion_matrix(y_test, best_y_pred)

print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy}")
print("\nClassification Report:")
print(best_classification_report)
print("\nConfusion Matrix:")
print(best_confusion_matrix)
