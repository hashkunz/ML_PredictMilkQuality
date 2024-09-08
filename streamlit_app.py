import streamlit as st
from joblib import load
import numpy as np

# Load the model and mappings
model = load('models/best_knn.pkl')
mapping = load('models/mapping.pkl')
columns = load('models/columns.pkl')

# Title of the app
st.title("Milk Quality Prediction")

# Create two columns for input fields
col1, col2 = st.columns(2)

# Get user inputs in the first column
with col1:
    pH = st.number_input('pH (1-14)', min_value=1.0, max_value=14.0, value=1.0, step=0.1)
    taste = st.selectbox('Taste (True: 1, False: 0)', [0, 1])
    odor = st.selectbox('Odor (True: 1, False: 0)', [0, 1])

# Get user inputs in the second column
with col2:
    temperature = st.number_input('Temperature (0-50)', min_value=0, max_value=50, value=0)
    fat = st.selectbox('Fat (True: 1, False: 0)', [0, 1])
    turbidity = st.selectbox('Turbidity (True: 1, False: 0)', [0, 1])

st.image("https://img2.pic.in.th/pic/ML_Milk_1.jpg")
colour = st.number_input('Colour (240-255)', min_value=240, max_value=255, value=255)

# Predict button
if st.button('Predict'):
    conditions = [pH, temperature, taste, odor, fat, turbidity, colour]
    data = [conditions[columns.get_loc(i)] for i in columns]

    # Predict the milk quality
    prediction = model.predict([data]).tolist()

    # Debugging output
    st.write("Label Grade: 0 = HIGH, 1 = LOW, 2 = MEDIUM")
    st.write(f"Prediction: {prediction[0]}")

    # Extract the mapping dictionary
    grade_mapping = mapping.get('Grade', {})

    # Map prediction to a label based on the available mapping
    reverse_mapping = {v: k for k, v in grade_mapping.items()}
    predicted_label = reverse_mapping.get(prediction[0], 'Unknown')

    # Display the result with corresponding color
    if predicted_label == 'high':
        st.success(f'The predicted milk quality is: {predicted_label}')
    elif predicted_label == 'medium':
        st.warning(f'The predicted milk quality is: {predicted_label}')
    elif predicted_label == 'low':
        st.error(f'The predicted milk quality is: {predicted_label}')
    else:
        st.info(f'The predicted milk quality is: {predicted_label}')
