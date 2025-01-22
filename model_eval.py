import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

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
y = df['Price']  # Target variable (NOT scaled)

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Standard Scaling (Only to numerical features)
numerical_features = ["SqFt", "Lot_Area", "Year_Built", "Bedrooms", "Bathrooms", "Garage", "Price_per_SqFt", "House_Age"]
scaler = StandardScaler()

X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Save the scaler for deployment
joblib.dump(scaler, 'scaler.pkl')

# Initialize models
models = {
    "Random Forest": RandomForestRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Linear Regression": LinearRegression(),
    "XGBRegressor": XGBRegressor(),
    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor()
}

# Define hyperparameter grids for tuning
params = {
    "Decision Tree": {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
    },
    "Random Forest": {
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "Gradient Boosting": {
        'learning_rate': [0.1, 0.01, 0.05, 0.001],
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "Linear Regression": {},
    "XGBRegressor": {
        'learning_rate': [0.1, 0.01, 0.05, 0.001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "CatBoosting Regressor": {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [30, 50, 100]
    },
    "AdaBoost Regressor": {
        'learning_rate': [0.1, 0.01, 0.5, 0.001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    }
}

# Dictionary to store model results
results = []

# Loop through models
for name, model in models.items():
    print(f"🔹 Training {name}...")
    
    # Perform GridSearchCV if hyperparameters are provided
    if name in params and params[name]:
        grid_search = GridSearchCV(estimator=model, param_grid=params[name], 
                                   scoring='neg_mean_absolute_error', cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"✅ Best Params for {name}: {best_params}")
    else:
        best_model = model.fit(X_train, y_train)

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Store results
    results.append((name, mae, rmse, r2))
    print(f"{name} → MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, R²: {r2:.4f}\n")

# Convert results to DataFrame and sort by MAE (best model first)
df_results = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R² Score"]).sort_values(by="MAE")


