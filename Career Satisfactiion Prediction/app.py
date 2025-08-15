import streamlit as st
import joblib
import numpy as np

# A dictionary to hold our models for easy access
MODELS = {
    "Random Forest Regressor": "random_forest_model.pkl",
    "K-Nearest Neighbors (KNN)": "knn_model.pkl",
    "Deep Neural Network (DNN)": "dnn_model.pkl"
}

# Define your 12 distinct input feature names here
FEATURE_NAMES = [
    "Age",
    "Gender",
    "Sleep Hours",
    "Country Region",
    "University CGPA",
    "Field Of Study",
    "Starting salary",
    "Stage Fear",
    "Personality",
    "Game Type",
]

# CSS to style the Streamlit app
def add_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');

        /* General Body Styling with a soft background */
        body {
            font-family: 'Montserrat', sans-serif;
            background-color: #F7F9FC; /* A very light blue-gray */
            color: #333;
        }
        
        /* Main Container Styling */
        .stApp {
            background-color: #F7F9FC;
        }
        
        /* Header Styling */
        h1, h2, h3 {
            color: #4A6572; /* A calming, dark gray-blue */
            font-family: 'Montserrat', sans-serif;
            font-weight: 700;
        }

        /* Custom Header for the App Title */
        .custom-title {
            background-color: #45b3e0; /* A soft, muted blue */
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 2.5em;
            font-weight: 700;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Button Styling */
        .stButton>button {
            background-color: #87ceeb; /* A medium blue */
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease-in-out;
            cursor: pointer;
        }

        .stButton>button:hover {
            background-color: #5879A9; /* A darker shade on hover */
            transform: scale(1.05);
        }
        
        /* Selectbox Styling */
        .stSelectbox>div>div {
            border-radius: 5px;
            border: 2px solid #ccc;
        }
        
        /* Input Field Styling */
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            border-radius: 5px;
            border: 2px solid #ccc;
            padding: 8px;
        }
        .stTextInput, .stNumberInput {
            margin-bottom: 10px;
        }
        
        /* Success and Warning Message Styling */
        .stAlert > div > div > div {
            border-left: 5px solid #6C8EBF;
        }
        
        /* Horizontal Rule Styling */
        hr {
            border-top: 2px solid #D1D9E6; /* A very light, subtle gray line */
        }
        
        </style>
    """, unsafe_allow_html=True)

# Load a model based on the selected name
def load_model(model_name):
    model_path = MODELS[model_name]
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: The model file '{model_path}' was not found. Please ensure it exists.")
        return None

# The prediction function
def predict_with_model(model, features):
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    return prediction

# The main Streamlit app function
def main():
    add_custom_css()
    
    st.markdown('<h1 class="custom-title">Career Satisfaction Prediction App 🔮</h1>', unsafe_allow_html=True)

    # --- Section 1: Model Selection ---
    st.header("1. Choose Your Model")
    
    model_option = st.selectbox(
        "Select a machine learning model:",
        list(MODELS.keys())
    )


    classifier = load_model(model_option)
    
    if classifier is None:
        return

    st.success(f"Model '{model_option}' loaded successfully!")
    st.markdown("---")


    # # We will not load the model, just show a message.
    # st.info(f"Frontend for '{model_option}' is being displayed. Prediction functionality is disabled.")
    # st.markdown("---")

    # --- Section 2: Input Features ---
    st.header("2. Enter the 12 Input Features")

    input_features = []
    # Use two columns for better layout of 12 inputs
    col1, col2 = st.columns(2)
    with col1:
        for i in range(5):
            feature_value = st.number_input(FEATURE_NAMES[i], value=0.0)
            input_features.append(feature_value)
    with col2:
        for i in range(5, 10):
            feature_value = st.number_input(FEATURE_NAMES[i], value=0.0)
            input_features.append(feature_value)

    st.markdown("---")

    # --- Section 3: Make a Prediction ---
    st.header("3. Get Your Prediction")

    predict_button_container = st.container()
    with predict_button_container:
        col_empty, col_btn, col_empty2 = st.columns([1, 2, 1])
        with col_btn:
            if st.button("Predict"):
                prediction = predict_with_model(classifier, input_features)
                
                st.info(f"<b>Prediction:</b> {prediction[0]:.4f}")
                st.balloons()

# Run the app
if __name__ == '__main__':
    main()