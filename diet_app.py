
import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("food_dataset_copy.csv")

# Use only selected features for prediction
selected_features = ["Calories", "Proteins", "Carbohydrates", "Veg"]
X = data[selected_features]
y = data["Food_items"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit UI
st.title("Diet Recommendation Model Analysis and Prediction")
st.write("Choose an option below to proceed:")

option = st.radio("Select an Option:", [
    "Preprocess Data",
    "Calculate BMI & Calories"
    "Make a Prediction",
])

if option == "Preprocess Data":
    st.subheader("Data Preprocessing")

    # Handle missing values
    st.write("Handling Missing Values...")
    data.fillna(data.median(numeric_only=True), inplace=True)
    st.write("Missing values filled with median values.")

    # Plot graphs

    st.write("Histogram of Features")
    fig, ax = plt.subplots(figsize=(20, 10))
    data.hist(ax=ax, bins=20, edgecolor='black')
    st.pyplot(fig)

if option == "Make a Prediction":
    st.subheader("Choose Prediction Model")

    calories = st.number_input("Calories", min_value=float(X["Calories"].min()), max_value=float(X["Calories"].max()))
    Proteins = st.number_input("Proteins", min_value=float(X["Proteins"].min()), max_value=float(X["Proteins"].max()))
    Carbohydrates = st.number_input("Carbohydrates", min_value=float(X["Carbohydrates"].min()), max_value=float(X["Carbohydrates"].max()))
    veg = st.radio("Veg (1 for Non-Veg, 0 for Veg)", [0, 1])

    input_features = np.array([[calories, Proteins, Carbohydrates, veg]])

    if st.button("Predict"):
        similarities = cosine_similarity(input_features, X)
        best_match_indices = np.argsort(similarities[0])[-5:][::-1]  # Get top 5 recommendations
        predictions = y.iloc[best_match_indices]

        st.subheader("Recommended Diets:")
        st.write(predictions)

        for idx in range(len(predictions)):
            st.subheader(f"Nutrient Breakdown for {predictions.iloc[idx]}")
            fig, ax = plt.subplots()
            nutrient_values = X.iloc[best_match_indices[idx]]
            nutrient_values.plot(kind='bar', ax=ax, color=['blue', 'green', 'orange', 'red'])
            ax.set_ylabel("Nutrient Amount")
            ax.set_xlabel("Nutrient Type")
            ax.set_title(f"Nutrient Breakdown for {predictions.iloc[idx]}")
            st.pyplot(fig)

if option == "Calculate BMI & Calories":
    st.subheader("Body Mass Index (BMI) Calculator")
    weight = st.number_input("Enter your weight (kg)", min_value=10.0, max_value=300.0)
    height_cm = st.number_input("Enter your height (cm)", min_value=50.0, max_value=250.0)

    if st.button("Calculate BMI"):
        height_m = height_cm / 100
        bmi = weight / (height_m ** 2)
        st.write(f"Your BMI is: {bmi:.2f}")

        if bmi < 18.5:
            st.write("You are underweight.")
        elif 18.5 <= bmi < 24.9:
            st.write("You have a normal weight.")
        elif 25 <= bmi < 29.9:
            st.write("You are overweight.")
        else:
            st.write("You are obese.")

    st.subheader("Daily Caloric Needs Calculator")
    age = st.number_input("Enter your age", min_value=1, max_value=120)
    gender = st.radio("Select your gender", ["Male", "Female"])
    activity_level = st.selectbox("Select your activity level", [
        "Sedentary (little or no exercise)",
        "Lightly active (light exercise/sports 1-3 days/week)",
        "Moderately active (moderate exercise/sports 3-5 days/week)",
        "Very active (hard exercise/sports 6-7 days a week)",
        "Super active (very hard exercise, physical job, or training)"
    ])

    if st.button("Calculate Calories"):
        if gender == "Male":
            bmr = 10 * weight + 6.25 * height_cm - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height_cm - 5 * age - 161

        activity_multiplier = {
            "Sedentary (little or no exercise)": 1.2,
            "Lightly active (light exercise/sports 1-3 days/week)": 1.375,
            "Moderately active (moderate exercise/sports 3-5 days/week)": 1.55,
            "Very active (hard exercise/sports 6-7 days a week)": 1.725,
            "Super active (very hard exercise, physical job, or training)": 1.9
        }

        daily_calories = bmr * activity_multiplier[activity_level]
        st.write(f"Maintain weight: {daily_calories:.0f} Calories/day")
        st.write(f"Mild weight loss: {daily_calories - 250:.0f} Calories/day (-0.25 kg/week)")
        st.write(f"Weight loss: {daily_calories - 500:.0f} Calories/day (-0.5 kg/week)")
        st.write(f"Extreme weight loss: {daily_calories - 1000:.0f} Calories/day (-1 kg/week)")



