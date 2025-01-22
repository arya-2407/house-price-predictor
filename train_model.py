import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/ca_real_estate.csv")  # Ensure correct path
df = df.drop(columns=['Province'])
# Feature Engineering
df["Price_per_SqFt"] = df["Price"] / df["SqFt"]
df["House_Age"] = 2025 - df["Year_Built"]
df["Has_Garage"] = (df["Garage"] > 0).astype(int)

# Apply One-Hot Encoding (OHE) for categorical features
df = pd.get_dummies(df, columns=['City', 'Type'], drop_first=True).astype(int)

# Define features and target
X = df.drop(columns=['Price'])  # Features
y = df['Price']  # Target variable

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

joblib.dump(X_train.columns.tolist(), 'feature_names.pkl')
print("✅ Feature names saved.")

# Apply Standard Scaling (Only to numerical features)
numerical_features = ["SqFt", "Lot_Area", "Year_Built", "Bedrooms", "Bathrooms", "Garage", "Price_per_SqFt", "House_Age"]
scaler = StandardScaler()

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Save the scaler for later use in deployment
joblib.dump(scaler, 'scaler.pkl')

# Train final Random Forest model with best hyperparameters
final_model = RandomForestRegressor(n_estimators=256, random_state=42)
final_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(final_model, 'final_house_price_model.pkl')
print("✅ Final Random Forest model saved as 'final_house_price_model.pkl'")

# Make predictions on test data
y_pred = final_model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Final Model Evaluation:")
print(f"MAE: ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"R² Score: {r2:.4f}")