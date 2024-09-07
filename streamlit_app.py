import streamlit as st
from joblib import load
import numpy as np

# Load the model and mappings
model = load('models/best_knn.pkl')
mapping = load('models/mapping.pkl')
columns = load('models/columns.pkl')

# Title of the app
st.title("Milk Quality Prediction")

# Get user inputs
pH = st.number_input('pH (1-14)', min_value=0.0, max_value=14.0, value=0.0, step=0.1)
temperature = st.number_input('Temperature (0-100)', min_value=0, max_value=100, value=0)
taste = st.selectbox('Taste', [0, 1])
odor = st.selectbox('Odor', [0, 1])
fat = st.selectbox('Fat', [0, 1])
turbidity = st.selectbox('Turbidity', [0, 1])
colour = st.number_input('Colour', min_value=0, max_value=255, value=0)

# Predict button
if st.button('Predict'):
    conditions = [pH, temperature, taste, odor, fat, turbidity, colour]
    data = []

    for i in columns:
        data.append(conditions[columns.get_loc(i)])

    # Predict the milk quality
    prediction = model.predict([data]).tolist()

    # Debugging output
    st.write(f"Label Grade: 0 = HIGH, 1 = LOW, 2 = Meduim")
    st.write(f"Prediction: {prediction[0]}")

    # Extract the mapping dictionary
    grade_mapping = mapping.get('Grade', {})

    # Map prediction to a label based on the available mapping
    reverse_mapping = {v: k for k, v in grade_mapping.items()}
    predicted_label = reverse_mapping.get(prediction[0], 'Unknown')

    st.success(f'The predicted milk quality is: {predicted_label}')
