from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store the trained model
model = None

def train_model():
    """Train the linear regression model using the sales dataset"""
    global model
    
    try:
        # Load the dataset
        data = pd.read_csv("sales_dataset.csv")
        
        # Prepare features and target
        X = data[['TV', 'Radio', 'Newspaper']].values
        y = data['Sales'].values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Print model performance
        score = model.score(X_test, y_test)
        print(f"Model trained successfully! R² Score: {score:.4f}")
        
        return True
    except Exception as e:
        print(f"Error training model: {e}")
        return False

@app.route('/')
def serve_frontend():
    """Serve the HTML frontend"""
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract features
        tv = data.get('tv')
        radio = data.get('radio')
        newspaper = data.get('newspaper')
        
        # Validate inputs
        if tv is None or radio is None or newspaper is None:
            return jsonify({'error': 'Missing required fields: tv, radio, newspaper'}), 400
        
        try:
            tv = float(tv)
            radio = float(radio)
            newspaper = float(newspaper)
        except (ValueError, TypeError):
            return jsonify({'error': 'All inputs must be valid numbers'}), 400
        
        # Check if model is trained
        if model is None:
            return jsonify({'error': 'Model not trained'}), 500
        
        # Make prediction
        features = np.array([[tv, radio, newspaper]])
        prediction = model.predict(features)[0]
        
        return jsonify({
            'prediction': round(prediction, 2),
            'inputs': {
                'tv': tv,
                'radio': radio,
                'newspaper': newspaper
            }
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': model is not None
    })

if __name__ == '__main__':
    print("Starting Sales Prediction Server...")
    
    # Train the model on startup
    if train_model():
        print("Starting Flask server on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to train model. Please check your sales_dataset.csv file.")