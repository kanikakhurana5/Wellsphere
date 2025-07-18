import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import base64

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Mental Health Assessment",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==================== BACKGROUND SETUP ====================
def add_bg_gif_local(gif_path):
    try:
        with open(gif_path, "rb") as f:
            gif_data = f.read()
        gif_base64 = base64.b64encode(gif_data).decode()
        
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/gif;base64,{gif_base64}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                padding-top: 0;
                position: relative
            }}

            .stApp::before {{
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background-color: rgba(0, 0, 0, 0.5); /* Adjust the opacity as needed */
                z-index: 2;
            }}

            .stApp * {{
                position: relative;
                z-index: 2; /* Ensures text stays above the overlay */
            }}          
            
            /* Remove default header space */
            header {{
                display: none;
            }}
            
            /* Title container */
            .title-container {{
                padding: 1.5rem 0;
                margin-bottom: 1rem;
                border-bottom: 2px solid #4CAF50;
            }}
            
            /* Question styling - larger and bolder */
            .question-text {{
                font-size: 1.5rem !important;
                font-weight: 700 !important;
                color: white !important;
                margin-bottom: 1rem !important;
                line-height: 1.4 !important;
            }}
            
            /* Question cards */
            .question-card {{
                margin-bottom: 1.8rem;
                padding: 0.8rem;
                background-color: rgba(248, 249, 250, 0.9);
                border-radius: 10px;
            }}
            
            /* Radio buttons */
            .stRadio > div {{
                flex-direction: column;
                align-items: flex-start;
                gap: 0.8rem;
            }}
            
            .stRadio label {{
                font-size: 1.6rem !important;
            }}
            
            /* Button styling */
            .stButton>button {{
                background-color: #4CAF50;
                color: white;
                padding: 1rem 2rem;
                border-radius: 8px;
                border: none;
                font-size: 1.3rem;
                font-weight: 600;
                width: 100%;
                margin: 2rem 0;
                transition: all 0.3s ease;
            }}
            
            .stButton>button:hover {{
                background-color: #3e8e41;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }}
            
            /* Results section */
            .result-card {{
                margin-top: 2rem;
                padding: 2rem;
                background-color: rgba(240, 248, 255, 0.95);
                border-radius: 10px;
            }}
            
            /* Prediction box */
            .prediction-box {{
                background-color: rgba(76, 175, 80, 0.2);
                padding: 1.8rem;
                border-radius: 10px;
                border-left: 5px solid #4CAF50;
                margin: 1.5rem 0;
            }}
            
            /* Resource links */
            .resource-links a {{
                color: #4CAF50 !important;
                text-decoration: underline !important;
                font-size: 1.1rem;
            }}
            
            /* Mobile responsiveness */
            @media (max-width: 768px) {{
                .main-container {{
                    padding: 0 1.5rem 1.5rem;
                    border-radius: 0;
                }}
                
                .question-text {{
                    font-size: 1.3rem !important;
                }}

            }}

            /* Enhanced error message styling */
        .stAlert {{
            background-color:  rgb(255, 0, 0) !important; 
            border-radius: 10px;
            padding: 0.5rem !important;
        }}

        .stException .message {{
            color: rgb(139, 0, 0) !important; /* Dark red for better readability */
            font-size: 1.1rem !important;
            font-weight: 600 !important;
        }}

        .stException .decoration {{
            color:rgb(211, 47, 47) !important; /* Brighter red for emphasis */
            font-size: 1.5rem !important;
        }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #f0f2f6;
                padding-top: 0;
            }
            header {
                display: none;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

# Add background
add_bg_gif_local("background1.gif")

# ==================== DATA & MODEL ====================
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

@st.cache_resource
def train_model(_df):
    # Convert categorical target to numeric
    _df[target_column] = _df[target_column].astype("category")
    category_mapping = dict(enumerate(_df[target_column].cat.categories))
    _df[target_column] = _df[target_column].cat.codes
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        _df[feature_columns], _df[target_column], test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test, category_mapping

# Define features and target
feature_columns = ["Stress_Q1", "Stress_Q2", "Stress_Q3", 
                  "Anxiety_Q1", "Anxiety_Q2", "Anxiety_Q3",
                  "Depression_Q1", "Depression_Q2", "Depression_Q3"]
target_column = "Final_Prediction"

# Load data and train model
df = load_data()
model, X_test, y_test, category_mapping = train_model(df)

# ==================== APP INTERFACE ====================
with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    
    # Title at the very top
    st.markdown("<div class='title-container'>", unsafe_allow_html=True)
    st.markdown("<h1 style='margin: 0;'>🧠 Mental Health Assessment</h1>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Introduction text
    st.markdown("""
    <div style='margin-bottom: 2rem; font-size: 1.1rem; line-height: 1.7;'>
    This confidential assessment will help you understand your current mental health state. 
    Please answer all questions honestly based on how often you've experienced each feeling in the past 2 weeks.
    </div>
    """, unsafe_allow_html=True)
    
    # Rating scale explanation
    with st.expander("ℹ️ How to rate your responses", expanded=False):
        st.markdown("""
        <div style='font-size: 1.1rem;'>
        - 0 (Never): Not at all<br>
        - 1 (Sometimes): Occasionally (1-2 days per week)<br>
        - 2 (Often): Frequently (3-4 days per week)<br>
        - 3 (Always): Nearly every day (5+ days per week)
        </div>
        """, unsafe_allow_html=True)
    
    # Questions with improved formatting
    responses = []
    questions = [
        "I found it difficult to relax or wind down.",
        "I felt overwhelmed and unable to cope with daily challenges.",
        "I had frequent headaches, muscle tension, or trouble sleeping due to stress.",
        "I felt excessively worried and found it hard to control my thoughts.",
        "I experienced sudden fear, shortness of breath, or felt like something bad was going to happen.",
        "I avoided situations, people, or places because they made me feel anxious.",
        "I felt persistently sad, empty, or hopeless.",
        "I lost interest or pleasure in activities I used to enjoy.",
        "I felt like a failure, worthless, or excessively guilty."
    ]
    
    for i, q in enumerate(questions):
        st.markdown(f"<div class='question-card'>", unsafe_allow_html=True)
        st.markdown(f'<div class="question-text">{i+1}. {q}</div>', unsafe_allow_html=True)
        response = st.radio(
            "Select your response:",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["Never (0)", "Sometimes (1)", "Often (2)", "Always (3)"][x],
            key=f"q_{i}",
            index=None,
            label_visibility="collapsed"
        )
        responses.append(response)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Submit button
    if st.button("Get Assessment Results", type="primary"):
        if None in responses:
            st.error("Please answer all questions before submitting.")
        else:
            # Make prediction
            input_data = np.array(responses).reshape(1, -1)
            prediction = model.predict(input_data)[0]
            prediction_label = category_mapping[prediction]
            
            # Calculate scores
            stress_score = sum(responses[:3])
            anxiety_score = sum(responses[3:6])
            depression_score = sum(responses[6:])
            
            # Display results
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.subheader("📊 Your Assessment Results")
            
            # Score columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<div style='font-size: 1.2rem; margin-bottom: 0.5rem;'>Stress Score</div>", unsafe_allow_html=True)
                st.metric("", f"{stress_score}/9")
                st.progress(stress_score / 9)
                st.caption(["Low", "Moderate", "High", "Very High"][min(3, stress_score//3)])
            
            with col2:
                st.markdown("<div style='font-size: 1.2rem; margin-bottom: 0.5rem;'>Anxiety Score</div>", unsafe_allow_html=True)
                st.metric("", f"{anxiety_score}/9")
                st.progress(anxiety_score / 9)
                st.caption(["Low", "Moderate", "High", "Very High"][min(3, anxiety_score//3)])
            
            with col3:
                st.markdown("<div style='font-size: 1.2rem; margin-bottom: 0.5rem;'>Depression Score</div>", unsafe_allow_html=True)
                st.metric("", f"{depression_score}/9")
                st.progress(depression_score / 9)
                st.caption(["Low", "Moderate", "High", "Very High"][min(3, depression_score//3)])
            
            st.divider()
            
            # Overall assessment
            st.markdown(
                f"""
                <div class='prediction-box'>
                    <h3 style='color: #4CAF50; margin-bottom: 0.8rem;'>Overall Assessment</h3>
                    <div style='font-size: 1.5rem; font-weight: 600;'>{prediction_label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Resources with working links
            st.markdown("""
            <div class='resource-links'>
                <strong>Mental Health Resources:</strong> <br> 
                - Crisis Text Line - Text HOME to 741741  
                <a href="https://www.crisistextline.org/" target="_blank" style="color: rgb(76, 175, 80); text-decoration: underline; font-size: 1.1rem; font-weight: 600;">
                https://www.crisistextline.org/</a><br>  
  
            </div>
                """, 
            unsafe_allow_html=True
            )

            st.markdown("""
            <div class='resource-links'>
                - 988 Suicide & Crisis Lifeline - Call or text 988  
                <a href="https://988lifeline.org/" target="_blank" style="color: rgb(76, 175, 80); text-decoration: underline; font-size: 1.1rem; font-weight: 600;">
                https://988lifeline.org/</a><br>  
            </div>
                """, 
            unsafe_allow_html=True
            )

            st.markdown("""
            <div class='resource-links'>
               - Find a Therapist  
                <a href="https://www.psychologytoday.com/us/therapists" target="_blank" style="color: rgb(76, 175, 80); text-decoration: underline; font-size: 1.1rem; font-weight: 600;">
                https://www.psychologytoday.com/us/therapists</a>
            </div>
                """, 
            unsafe_allow_html=True
            )

            
            # Disclaimer
            st.markdown("""
            <div style='font-size: 1.2rem; font-weight: 600;  margin-top: 1.5rem;'>
            Note: This assessment is not a substitute for professional diagnosis. 
            If you're concerned about your mental health, please consult a qualified healthcare provider.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)