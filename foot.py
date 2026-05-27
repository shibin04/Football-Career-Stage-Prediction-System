import streamlit as st
import pickle
import numpy as np

# Load model & encoders
model = pickle.load(open('model_rf.sav', 'rb'))
encoders = pickle.load(open('encoders.sav', 'rb'))

st.title("⚽ Football Career Stage Predictor")

st.write("Enter Player Details:")

# 🔥 Manual mapping (FIXES numbers issue)
nationality_map = {
    0: "Brazil",
    1: "India",
    2: "Spain",
    3: "Germany",
    4: "France"
}

position_map = {
    0: "Goalkeeper",
    1: "Defender",
    2: "Midfielder",
    3: "Forward"
}

season_map = {
    0: "2019",
    1: "2020",
    2: "2021",
    3: "2022",
    4: "2023"
}

# Dropdowns (show names)
nationality_name = st.selectbox("Nationality", list(nationality_map.values()))
position_name = st.selectbox("Position", list(position_map.values()))
season_name = st.selectbox("Season", list(season_map.values()))

# Convert back to numbers
nat = list(nationality_map.keys())[list(nationality_map.values()).index(nationality_name)]
pos = list(position_map.keys())[list(position_map.values()).index(position_name)]
sea = list(season_map.keys())[list(season_map.values()).index(season_name)]

# Numeric inputs
age = st.number_input("Age", min_value=15, max_value=45)
matches = st.number_input("Matches Played", min_value=0)
goals = st.number_input("Goals", min_value=0)
assists = st.number_input("Assists", min_value=0)

# 🎯 Career stage mapping (FIXES output issue)
stage_map = {
    0: "Early",
    1: "Peak",
    2: "Decline"
}

# Predict
if st.button("Predict"):

    input_data = np.array([[nat, pos, sea, age, matches, goals, assists]])

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    stage = stage_map.get(prediction[0], "Unknown")

    st.success(f"Predicted Career Stage: {stage}")
    st.write(f"Confidence: {prob.max():.2f}")