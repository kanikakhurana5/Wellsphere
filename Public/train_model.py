import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("dataset.csv")

# Define features and target
feature_columns = ["Stress_Q1", "Stress_Q2", "Stress_Q3", "Anxiety_Q1", "Anxiety_Q2", "Anxiety_Q3", "Depression_Q1", "Depression_Q2", "Depression_Q3"]
target_column = "Final_Prediction"

# Convert categorical target to numeric
df[target_column] = df[target_column].astype("category")
category_mapping = dict(enumerate(df[target_column].cat.categories))  # Store mapping
df[target_column] = df[target_column].cat.codes  # Encode to numeric

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(df[feature_columns], df[target_column], test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model & Category Mapping
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("category_mapping.pkl", "wb") as f:
    pickle.dump(category_mapping, f)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Streamlit UI
st.title("Mental Health Assessment")
st.write("Answer the following questions (0: Never, 3: Always)")

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
    responses.append(st.slider(q, 0, 3, 0))

if st.button("Get Prediction"):
    # Load model and category mapping
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("category_mapping.pkl", "rb") as f:
        category_mapping = pickle.load(f)

    # Predict
    input_data = np.array(responses).reshape(1, -1)
    prediction = model.predict(input_data)[0]

    # Convert numeric prediction back to label
    prediction_label = category_mapping[prediction]

    st.subheader("Your Results")
    st.write(f"Stress Score: {sum(responses[:3])}")
    st.write(f"Anxiety Score: {sum(responses[3:6])}")
    st.write(f"Depression Score: {sum(responses[6:])}")
    st.success(f"Prediction: {prediction_label}")