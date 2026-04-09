import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
@st.cache_resource
def load_model():
  model_path = hf_hub_download(
      repo_id = 'vineeth32/tourism-model',
      filename = 'best_tourism_model.joblib'
      )
  # Load the model
  model = joblib.load(model_path)
  return model

model = load_model()

# Streamlit UI whether customer purchases a package or not
st.title('Visit with Us: Customer Purchase Predictor')
st.subheader('An internal application predicts whether the customer is likely to purchase a package or not based on customer details and interaction data.')
st.write('**Enter the customer details here.**')

# Collect user input data
# Customer details
Age = st.number_input("Age (Customer's age in years)", min_value=18, max_value=100, value=45)
TypeofContact = st.selectbox('The method by which the customer was contacted', ['Self Enquiry', 'Company Invited '])
CityTier = st.selectbox('The city category customer belongs based on development, population, and living standards', [1, 2, 3])
Occupation = st.selectbox("Customer's occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
Gender = st.selectbox('Gender of the customer', ['Male', 'Female'])
NumberOfPersonVisiting = st.number_input('Total number of people accompanying the customer on the trip', min_value=0, max_value=10, value=2)
PreferredPropertyStar = st.selectbox('Preferred hotel rating by the customer', [3, 4, 5])
MaritalStatus = st.selectbox('Marital status of the customer', ['Married', 'Divorced', 'Unmarried', 'Single'])
NumberOfTrips = st.number_input('Average number of trips the customer takes annually', min_value=0, max_value=50, value=2)
Passport = st.selectbox('Whether the customer holds a valid passport', ['Yes', 'No'])
OwnCar = st.selectbox('Whether the customer owns a car', ['Yes', 'No'])
NumberOfChildrenVisiting = st.number_input('Number of children below age 5 accompanying the customer', min_value=0, max_value=3, value=2)
Designation = st.selectbox("Customer's designation in their current organization", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
MonthlyIncome = st.number_input('Gross monthly income of the customer', min_value=1000, max_value=100000, value=50000)

# Customer interaction data
st.write('**Sales Interaction Summary**')
PitchSatisfactionScore = st.selectbox("Score indicating the customer's satisfaction with the sales pitch", [1, 2, 3, 4, 5])
ProductPitched = st.selectbox('The type of product pitched to the customer', ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
NumberOfFollowups = st.slider('Total number of follow-ups by the salesperson after the sales pitch', min_value=1, max_value=10, value=3)
DurationOfPitch = st.number_input('Duration of the sales pitch delivered to the customer', min_value=3, max_value=200, value=20)

# Assemble input data into a dataframe
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == 'Yes' else 0,
    'OwnCar': 1 if OwnCar == 'Yes' else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button('Predict'):
  prediction_proba = model.predict_proba(input_data)[0, 1]
  prediction = (prediction_proba >= classification_threshold).astype(int)
  result = 'likely' if prediction == 1 else 'unlikely'
  st.write(f"Based on the information provided the customer is '{result}' to purchase the package")
