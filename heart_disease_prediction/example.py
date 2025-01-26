# from openai import OpenAI
# client = OpenAI()

# completion = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "Write a haiku about recursion in programming."
#         }
#     ]
# )

# print(completion.choices[0].message)

import streamlit as st
import requests
import json

# Define the Flask API URL
FLASK_API_URL = 'http://127.0.0.1:5000/predict'

# Streamlit UI
st.title('Heart Disease Prediction')

# Form for user input
with st.form(key='prediction_form'):
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    sex = st.selectbox('Sex', [0, 1])
    cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=120)
    chol = st.number_input('Cholesterol', min_value=0, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar', [0, 1])
    restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=300, value=150)
    exang = st.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.number_input('Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3])
    thal = st.selectbox('Thalassemia', [1, 2, 3])

    submit_button = st.form_submit_button(label='Submit')

# Handle form submission
if submit_button:
    # Create the data payload
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    # Send the request to the Flask API
    response = requests.post(FLASK_API_URL, json=data)
    result = response.json()

    # Display results
    st.subheader('Results:')
    st.write(f"**Logistic Regression Prediction:** {result['logistic_prediction']}")
    st.write(f"**Random Forest Prediction:** {result['rf_prediction']}")
    st.write(f"**Neural Network Prediction:** {result['nn_prediction']}")
    st.write(f"**ChatGPT Advice:** {result['chatgpt_advice']}")
