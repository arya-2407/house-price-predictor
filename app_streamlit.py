import streamlit as st
import requests

# Streamlit UI
st.title("🏡 House Price Prediction App")

st.write("Enter the house details below to predict its price.")

# User inputs
sqft = st.number_input("Square Footage", min_value=300, max_value=10000, value=1500)
lot_area = st.number_input("Lot Area", min_value=500, max_value=50000, value=4000)
year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2005)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
garage = st.number_input("Garage Spaces", min_value=0, max_value=5, value=1)

# Categorical inputs (Cities & House Type)
city = st.selectbox("City", ["Toronto", "Vancouver", "Ottawa", "Montreal"])
house_type = st.selectbox("House Type", ["Apartment", "House"])

# One-Hot Encoding for Cities
city_features = {
    "City_Toronto": 1 if city == "Toronto" else 0,
    "City_Vancouver": 1 if city == "Vancouver" else 0,
    "City_Ottawa": 1 if city == "Ottawa" else 0,
    "City_Montreal": 1 if city == "Montreal" else 0
}

# One-Hot Encoding for House Type
type_features = {
    "Type_Apartment": 1 if house_type == "Apartment" else 0,
    "Type_House": 1 if house_type == "House" else 0
}

# Compute derived features
price_per_sqft = sqft / lot_area
house_age = 2025 - year_built

if st.button("Predict Price"):
    #st.rerun()  # Ensure Streamlit updates inputs before making predictions

    data = {
        "SqFt": sqft,
        "Lot_Area": lot_area,
        "Year_Built": year_built,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Garage": garage,
        "Price_per_SqFt": sqft / lot_area,
        "House_Age": 2025 - year_built,
        "City_Toronto": 1 if city == "Toronto" else 0,
        "City_Vancouver": 1 if city == "Vancouver" else 0,
        "City_Ottawa": 1 if city == "Ottawa" else 0,
        "City_Montreal": 1 if city == "Montreal" else 0,
        "Type_Apartment": 1 if house_type == "Apartment" else 0,
        "Type_House": 1 if house_type == "House" else 0
    }

    # Debugging: Print values to check if they update
    st.write("🔍 Sending this data to API:", data)

    # Call Flask API
    response = requests.post("http://127.0.0.1:5000/predict", json=data)

    if response.status_code == 200:
        prediction = response.json()['predicted_price']
        st.success(f"🏡 Estimated House Price: **${prediction:,.2f}**")
    else:
        st.error("Error: Unable to get prediction. Check API.")
