from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load trained model and scaler
model = joblib.load('final_house_price_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')  # Ensure feature names match training

# Define numerical features for scaling
numerical_features = ["SqFt", "Lot_Area", "Year_Built", "Bedrooms", "Bathrooms", "Garage", "Price_per_SqFt", "House_Age"]

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "House Price Prediction API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.get_json()

        # Convert data into DataFrame
        df = pd.DataFrame([data])

        # 🛠️ Ensure Derived Features Are Present
        df["Price_per_SqFt"] = df["SqFt"] / df["Lot_Area"]
        df["House_Age"] = 2025 - df["Year_Built"]
        df["Has_Garage"] = (df["Garage"] > 0).astype(int)

        # Apply One-Hot Encoding to match training features
        df = pd.get_dummies(df)

        # Align with training feature names
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0  # Add missing columns with value 0

        df = df[feature_names]  # Ensure correct column order

        # Scale numerical features
        df[numerical_features] = scaler.transform(df[numerical_features])

        # Make prediction
        predicted_price = model.predict(df)[0]

        return jsonify({'predicted_price': round(predicted_price, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
