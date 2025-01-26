import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Custom CSS
st.markdown(
    """
    <style>
        /* Background color for the app */
        .main {
            background-color: #E3F2FD;
        }

        /* Sidebar styles */
        .css-1d391kg {
            background-color: #FAFAFA;
        }

        /* Header styles */
        h1, h2, h3, h4, h5, h6 {
            color: #00796B;
        }

        /* Button styles */
        .stButton > button {
            background-color: #FF7043;
            color: white;
            border: None;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stButton > button:hover {
            background-color: #F4511E;
        }

        /* Text color */
        body, .markdown-text-container {
            color: #212121;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sample Data Generation
def generate_sample_data():
    np.random.seed(42)
    bmi = np.random.normal(25, 5, 200)  # BMI values around 25 with std deviation of 5
    cholesterol = np.random.normal(200, 50, 200)  # Cholesterol values around 200 with std deviation of 50
    age = np.random.randint(5, 80, size=200)  # Age between 5 and 80
    hba1c = np.random.normal(6, 1, 200)  # HBA1c levels
    rbs = np.random.normal(100, 20, 200)  # Random Blood Sugar levels
    hypertension = np.random.choice([0, 1], size=200)  # 0 = no hypertension, 1 = hypertension
    diabetes = np.random.choice([0, 1], size=200)  # 0 = no diabetes, 1 = diabetes
    return pd.DataFrame({'BMI': bmi, 'Cholesterol': cholesterol, 'Age': age, 'HBA1c': hba1c, 'Random Blood Sugar': rbs, 'Hypertension': hypertension, 'Diabetes': diabetes})

# Age classification for health recommendations
def classify_age(age):
    if age < 12:
        return 'Child'
    elif 12 <= age <= 19:
        return 'Teenager'
    elif 20 <= age <= 35:
        return 'Young Adult'
    elif 36 <= age <= 60:
        return 'Middle-Aged'
    else:
        return 'Older Adult'

# Health risk recommendation based on age, BMI, Cholesterol, Hypertension, HBA1c, and Random Blood Sugar
def get_recommendation(age, bmi, cholesterol, hba1c, rbs, hypertension, diabetes):
    age_group = classify_age(age)
    recommendations = []

    # General Advice
    recommendations.append(f"Age Group: {age_group}")
    
    # Age-specific advice
    if age_group == 'Child':
        recommendations.append("Children should maintain a healthy diet and stay active. Ensure balanced meals with sufficient fruits and vegetables. Avoid sugary drinks.")
    elif age_group == 'Teenager':
        recommendations.append("Teens should avoid processed foods, focus on maintaining a balanced diet, and incorporate exercise into their routine. Avoid excessive weight gain.")
    elif age_group == 'Young Adult':
        recommendations.append("For young adults, maintaining a balanced diet, regular exercise, and weight management is crucial to prevent future health issues.")
    elif age_group == 'Middle-Aged':
        recommendations.append("Middle-aged individuals should prioritize regular health check-ups, especially for blood pressure, cholesterol, and blood sugar. Maintain a balanced diet and exercise regimen.")
    else:  # Older Adult
        recommendations.append("Older adults should focus on heart health, managing cholesterol levels, maintaining healthy blood sugar levels, and staying physically active. Regular health screenings are essential.")
    
    # Risk Factors
    if bmi > 30:
        recommendations.append("BMI is high. Consider reducing weight by following a calorie-controlled diet and increasing physical activity. Consult with a healthcare professional for a tailored plan.")
    if cholesterol > 240:
        recommendations.append("High cholesterol detected. Limit saturated fats, reduce red meat consumption, and increase intake of fruits, vegetables, and whole grains. Regular exercise can help lower cholesterol.")
    if hba1c >= 6.5:
        recommendations.append("HBA1c level suggests diabetes. Consult with a doctor for further tests. Follow a diabetic-friendly diet: low-carb, high-fiber, and focus on whole grains.")
    if rbs > 140:
        recommendations.append("Random blood sugar is high. Consider lifestyle changes such as a low-glycemic index diet and regular physical activity.")
    if hypertension == 1:
        recommendations.append("Hypertension detected. Limit salt intake, avoid processed foods, and aim for regular physical activity. Medication may be required, consult a healthcare professional.")
    if diabetes == 1:
        recommendations.append("Diabetes risk detected. Follow a diabetic-friendly diet (low-carb, high-fiber), exercise regularly, and monitor blood sugar levels.")
        recommendations.append("For Type 1 Diabetes: Insulin injections may be required, along with regular monitoring of blood sugar levels. Consult your healthcare provider for a tailored insulin regimen.")
        recommendations.append("For Type 2 Diabetes: Consider oral medications such as Metformin to manage blood sugar levels. Lifestyle changes including weight loss, exercise, and diet are crucial. Consult your doctor for a tailored treatment plan.")
    
    return recommendations

# Main Streamlit App
def main():
    st.title("Health Risk Recommendation System")
    st.markdown(
        """
        <p style="font-size: 18px;">This app predicts a patient's health risk based on BMI, Cholesterol, Age, HBA1c, Random Blood Sugar, and Hypertension. It also provides recommendations on diet, lifestyle, and medications.</p>
        """, unsafe_allow_html=True
    )

    # Sidebar enhancements with icons
    st.sidebar.title("Health Info")
    st.sidebar.markdown(
        """
        <div style="font-size: 18px; color: #00796B;">
            <i class="fas fa-heartbeat"></i> Health Risk Management
        </div>
        """, unsafe_allow_html=True
    )

    # Generate sample data
    df = generate_sample_data()

    # Preprocess Data
    df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
    df['Cholesterol'] = pd.to_numeric(df['Cholesterol'], errors='coerce')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['HBA1c'] = pd.to_numeric(df['HBA1c'], errors='coerce')
    df['Random Blood Sugar'] = pd.to_numeric(df['Random Blood Sugar'], errors='coerce')
    imputer = SimpleImputer(strategy='mean')
    df[['BMI', 'Cholesterol', 'HBA1c', 'Random Blood Sugar']] = imputer.fit_transform(df[['BMI', 'Cholesterol', 'HBA1c', 'Random Blood Sugar']])

    # Perform KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['BMI', 'Cholesterol', 'Age', 'HBA1c', 'Random Blood Sugar']])

    # Cluster Mapping
    cluster_names = {
        0: 'Low Risk (Healthy)',
        1: 'Moderate Risk (Overweight/High Cholesterol)',
        2: 'High Risk (Diabetes/Hypertension)'
    }
    df['Cluster_Name'] = df['Cluster'].map(cluster_names)

    # Display Dataset
    st.subheader("Clustered Dataset")
    st.write(df.head())

    # Input fields for patient details with icons
    st.subheader("Enter Patient Details")
    age = st.number_input("Age", min_value=0, max_value=120, value=30, help="Enter your age to get health recommendations. <i class='fas fa-birthday-cake'></i>", format="%d")
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1, help="Enter your BMI for health risk assessment. <i class='fas fa-weight'></i>")
    cholesterol = st.number_input("Cholesterol Level", min_value=0.0, max_value=400.0, value=200.0, step=1.0, help="Enter cholesterol level. <i class='fas fa-heart'></i>")
    hba1c = st.number_input("HBA1c Level", min_value=0.0, max_value=20.0, value=6.0, step=0.1, help="Enter your HBA1c level. <i class='fas fa-vial'></i>")
    rbs = st.number_input("Random Blood Sugar Level", min_value=0.0, max_value=400.0, value=100.0, step=1.0, help="Enter your Random Blood Sugar level. <i class='fas fa-tint'></i>")
    hypertension = st.selectbox("Do you have Hypertension?", options=["Yes", "No"], index=1, help="Select Yes or No for Hypertension.")
    hypertension = 1 if hypertension == "Yes" else 0
    diabetes = st.selectbox("Do you have Diabetes?", options=["Yes", "No"], index=1, help="Select Yes or No for Diabetes.")
    diabetes = 1 if diabetes == "Yes" else 0

    if st.button("Get Health Risk and Recommendations"):
        # Get the recommendation based on the entered data
        recommendations = get_recommendation(age, bmi, cholesterol, hba1c, rbs, hypertension, diabetes)

        # Display the recommendations
        st.subheader("Health Recommendations")
        for rec in recommendations:
            st.write(f"- {rec}")

    # Visualize Clusters
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster]
        ax.scatter(cluster_data['BMI'], cluster_data['Cholesterol'], label=cluster_names[cluster])

    ax.set_title("Clusters based on BMI, Cholesterol, Age, HBA1c, and Blood Sugar")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Cholesterol")
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
