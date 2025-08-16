import streamlit as st
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
# A dictionary to hold our models for easy access
MODELS = {
    "K-Nearest Neighbors (KNN)": "knn_model.pkl",
    "Random Forest Classifier": "random_forest_model.pkl",
    "Deep Neural Network (DNN)": "dnn_model.pkl" 
}
# Define your 12 distinct input feature names here, matching the UI inputs
INPUT_FEATURE_NAMES = [
    "Age", "Sleep Hours", "University GPA", "Starting Salary",
    "Soft Skills Score", "Networking Score", "Gender", "Country Region",
    "Field Of Study", "Stage Fear", "Personality", "Game Type (Genre)"
]

# The list of all 50 columns that your model expects, derived from your notebook
MODEL_COLUMNS = [
    'age', 'sleep_hours', 'University_GPA', 'Starting_Salary',
    'Communication_skills',
    'Female', 'Male', 'Other',
    'Region_Afghanistan', 'Region_Bangaldesh', 'Region_Bangladesh',
    'Region_China', 'Region_Drnmark', 'Region_Hawaii', 'Region_India',
    'Region_Iran', 'Region_Iraq', 'Region_Italy', 'Region_Japan',
    'Region_Nepal', 'Region_Sudan', 'Region_Thailand', 'Region_Turkey',
    'Region_US', 'Region_Uzbekistan', 'Region_Vietnam', 'Region_uganda',
    'of_Study_Arts', 'of_Study_Business', 'of_Study_Computer Science',
    'of_Study_Engineering', 'of_Study_Law', 'of_Study_Mathematics',
    'of_Study_Medicine',
    'fear_No', 'fear_Yes',
    'Extrovert', 'Introvert',
    'Action', 'Adventure', 'Fighting', 'Misc', 'Platform', 'Puzzle',
    'Racing', 'Role-Playing', 'Shooter', 'Simulation', 'Sports', 'Strategy'
]

# Mapping for categorical features
GENDER_MAP = {'Female': 'Female', 'Male': 'Male', 'Other': 'Other'}
COUNTRY_REGION_MAP = {
    'US': 'Region_US', 'Bangaldesh': 'Region_Bangaldesh', 'Bangladesh': 'Region_Bangladesh',
    'China': 'Region_China', 'Drnmark': 'Region_Drnmark', 'Hawaii': 'Region_Hawaii',
    'India': 'Region_India', 'Iran': 'Region_Iran', 'Iraq': 'Region_Iraq',
    'Italy': 'Region_Italy', 'Japan': 'Region_Japan', 'Nepal': 'Region_Nepal',
    'Sudan': 'Region_Sudan', 'Thailand': 'Region_Thailand', 'Turkey': 'Region_Turkey',
    'Uzbekistan': 'Region_Uzbekistan', 'Vietnam': 'Region_Vietnam', 'uganda': 'Region_uganda',
    'Afghanistan': 'Region_Afghanistan'
}
FIELD_OF_STUDY_MAP = {
    'Arts': 'of_Study_Arts', 'Law': 'of_Study_Law', 'Medicine': 'of_Study_Medicine',
    'Computer Science': 'of_Study_Computer Science', 'Engineering': 'of_Study_Engineering',
    'Mathematics': 'of_Study_Mathematics', 'Business': 'of_Study_Business'
}
STAGE_FEAR_MAP = {'Yes': 'fear_Yes', 'No': 'fear_No'}
PERSONALITY_MAP = {'Extrovert': 'Extrovert', 'Introvert': 'Introvert'}
GENRE_MAP = {
    'Sports': 'Sports', 'Platform': 'Platform', 'Racing': 'Racing',
    'Role-Playing': 'Role-Playing', 'Action': 'Action', 'Adventure': 'Adventure',
    'Fighting': 'Fighting', 'Misc': 'Misc', 'Puzzle': 'Puzzle',
    'Shooter': 'Shooter', 'Simulation': 'Simulation', 'Strategy': 'Strategy'
}

# --- Custom CSS for a soothing color palette ---
def add_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
        body {
            font-family: 'Montserrat', sans-serif;
            background-color: #F7F9FC;
            color: #333;
        }
        .stApp { background-color: #F7F9FC; }
        h1, h2, h3 {
            color: #4A6572;
            font-family: 'Montserrat', sans-serif;
            font-weight: 700;
        }
        .custom-title {
            background-color: #45b3e0;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 2.5em;
            font-weight: 700;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: #87ceeb;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease-in-out;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #5879A9;
            transform: scale(1.05);
        }
        .stSelectbox>div>div {
            border-radius: 5px;
            border: 2px solid #ccc;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            border-radius: 5px;
            border: 2px solid #ccc;
            padding: 8px;
        }
        .stTextInput, .stNumberInput {
            margin-bottom: 10px;
        }
        .stAlert > div > div > div {
            border-left: 5px solid #6C8EBF;
        }
        hr {
            border-top: 2px solid #D1D9E6;
        }
        </style>
    """, unsafe_allow_html=True)

def predict_with_model(model, inputs):
    # Initialize a DataFrame with zeros for all 50 model columns
    df_predict = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)

    # Populate the numerical features
    df_predict['age'] = inputs['age']
    df_predict['sleep_hours'] = inputs['sleep_hours']
    df_predict['University_GPA'] = inputs['University_GPA']
    df_predict['Starting_Salary'] = inputs['Starting_Salary']
    
    # Calculate and populate the combined feature
    df_predict['Communication_skills'] = inputs['Soft_Skills_Score'] + inputs['Networking_Score']
    
    # Populate the one-hot encoded categorical features
    df_predict[GENDER_MAP[inputs['gender']]] = 1
    df_predict[COUNTRY_REGION_MAP[inputs['Country_Region']]] = 1
    df_predict[FIELD_OF_STUDY_MAP[inputs['Field_of_Study']]] = 1
    df_predict[STAGE_FEAR_MAP[inputs['Stage_fear']]] = 1
    df_predict[PERSONALITY_MAP[inputs['Personality']]] = 1
    df_predict[GENRE_MAP[inputs['genre']]] = 1

    # The input to the model should be a numpy array
    prediction = model.predict(df_predict)
    return prediction

def main():
    add_custom_css()
    
    st.markdown('<h1 class="custom-title">Career Satisfaction Prediction App 🔮</h1>', unsafe_allow_html=True)
    st.header("1. Choose Your Model")
    
    model_option = st.selectbox("Select a machine learning model:", list(MODELS.keys()))
    
    try:
        classifier = joblib.load(MODELS[model_option])
        st.success(f"Model '{model_option}' loaded successfully!")
    except FileNotFoundError:
        st.error(f"Error: The model file '{MODELS[model_option]}' was not found. Please ensure it exists.")
        st.info("The application can still be run, but you will not be able to get a prediction.")
        classifier = None
    
    st.markdown("---")
    st.header("2. Enter the Input Features")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(INPUT_FEATURE_NAMES[0], value=22.0, min_value=17.0, max_value=60.0)
        sleep_hours = st.number_input(INPUT_FEATURE_NAMES[1], value=7.0, min_value=0.0, max_value=12.0)
        university_gpa = st.number_input(INPUT_FEATURE_NAMES[2], value=3.0, min_value=0.0, max_value=4.0)
        starting_salary = st.number_input(INPUT_FEATURE_NAMES[3], value=50000.0, min_value=0.0)
        soft_skills_score = st.number_input(INPUT_FEATURE_NAMES[4], value=5.0, min_value=1.0, max_value=10.0)
        networking_score = st.number_input(INPUT_FEATURE_NAMES[5], value=5.0, min_value=1.0, max_value=10.0)
    
    with col2:
        gender = st.selectbox(INPUT_FEATURE_NAMES[6], list(GENDER_MAP.keys()))
        country_region = st.selectbox(INPUT_FEATURE_NAMES[7], list(COUNTRY_REGION_MAP.keys()))
        field_of_study = st.selectbox(INPUT_FEATURE_NAMES[8], list(FIELD_OF_STUDY_MAP.keys()))
        stage_fear = st.selectbox(INPUT_FEATURE_NAMES[9], list(STAGE_FEAR_MAP.keys()))
        personality = st.selectbox(INPUT_FEATURE_NAMES[10], list(PERSONALITY_MAP.keys()))
        game_genre = st.selectbox(INPUT_FEATURE_NAMES[11], list(GENRE_MAP.keys()))

    inputs = {
        'age': age,
        'sleep_hours': sleep_hours,
        'University_GPA': university_gpa,
        'Starting_Salary': starting_salary,
        'Soft_Skills_Score': soft_skills_score,
        'Networking_Score': networking_score,
        'gender': gender,
        'Country_Region': country_region,
        'Field_of_Study': field_of_study,
        'Stage_fear': stage_fear,
        'Personality': personality,
        'genre': game_genre
    }

    st.markdown("---")
    st.header("3. Get Your Prediction")

    if st.button("Predict"):
        if classifier is not None:
            # Here is the corrected line:
            prediction = predict_with_model(classifier, inputs)
            
            satisfaction_levels = {
                0: "Low Career Satisfaction",
                1: "Average Career Satisfaction",
                2: "High Career Satisfaction"
            }
             # It extracts the single element from the array and converts it to an integer
            predicted_value = int(prediction[0])
            
            st.markdown(f'<div style="color: #6C8EBF; font-weight: bold; font-size: 1.2em;">Prediction: {satisfaction_levels[predicted_value]}</div>', unsafe_allow_html=True)
            st.balloons()
        else:
            st.warning("Prediction functionality is disabled. Please add the .pkl model files to enable it.")


if __name__ == '__main__':
    main()






