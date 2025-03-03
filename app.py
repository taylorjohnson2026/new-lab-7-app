import streamlit as st

# Streamlit App UI
st.title("Ames Housing Price Predictor")
st.write("Enter details about the house to predict its price.")

# Input fields
overall_qual = st.number_input("Overall Quality (1-10)", min_value=1, max_value=10, value=5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=500, max_value=5000, value=1500)
garage_cars = st.number_input("Garage Spaces", min_value=0, max_value=4, value=2)
total_bsmt_sf = st.number_input("Total Basement Size (sq ft)", min_value=0, max_value=3000, value=800)
year_built = st.number_input("Year Built", min_value=1872, max_value=2023, value=2000)

# Predict button
if st.button("Predict Price"):
    # Prepare input data
    input_data = pd.DataFrame([[overall_qual, gr_liv_area, garage_cars, total_bsmt_sf, year_built]], columns=features)
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)[0]
    
    # Display result
    st.write(f"Estimated House Price: **${prediction:,.2f}**")


# Train a regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model performance
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
