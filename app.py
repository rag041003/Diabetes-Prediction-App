from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model, scaler, and polynomial transformer
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly_transformer.pkl')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        data = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
        # Convert to numpy array and reshape for transformation
        data = np.array(data).reshape(1, -1)
        # Apply polynomial transformation
        data_poly = poly.transform(data)
        # Scale the data
        data_scaled = scaler.transform(data_poly)
        # Make prediction
        prediction = model.predict(data_scaled)
        
        # Map the prediction to 'Diabetic' or 'Non-diabetic'
        if prediction[0] == 1.0:
            result = 'Diabetic'
        else:
            result = 'Non-diabetic'
        
        # Render result
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
