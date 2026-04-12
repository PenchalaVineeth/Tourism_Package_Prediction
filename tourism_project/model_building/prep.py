# Library for data manipulation and reading
import pandas as pd
import numpy as np
# Library to perform train_test_split
from sklearn.model_selection import train_test_split
# Library for imputing any missing values
from sklearn.impute import SimpleImputer
# Hugging face library for authentication and creating repo
from huggingface_hub import HfApi
# Library to interact with operating system
import os

# Initialize API client
api = HfApi(token = os.getenv('HF_TOKEN'))

# Load dataset from hugging face
dataset_path = 'hf://datasets/vineeth32/tourism-data/tourism.csv'
# Read data
data = pd.read_csv(dataset_path)
print('Dataset successfully loaded.')

# Remove any unnamed or empty columns exists
if 'Unnamed: 0' in data.columns or data.columns[0] == '':
  data = data.iloc[:, 1:]

# Dropping customerID column, which is unique and will not play significant role for analysis
if 'CustomerID' in data.columns:
  data.drop('CustomerID', axis=1, inplace=True)
print('Removed un-necessary columns, which are not required for the analysis.')

# Correcting Female records in Gender attribute
if 'Gender' in data.columns:
  data['Gender'] = data['Gender'].str.strip().replace({'Fe Male': 'Female'})
  print("Necessary correction in gender column for few 'Female' records is done")

# Handling missing values, if new dataset is added in future
print('Checking for any missing values in all columns and imputing the missing values using median and mode strategy for numerical and categorical columns respectively\n')
# numeric column missing value treatment
print('Examining if any missing values present in each numerical column')
# Get only numeric columns
numeric_cols = data.select_dtypes(include=np.number)
# Lopping over each column and checking for null value and impute the null values
for col in numeric_cols:
  if data[col].isnull().sum() > 0:
    num_imputer = SimpleImputer(strategy='median')
    data[col] = num_imputer.fit_transform(data[[col]])
    print(f'Missing values for {col} attribute were imputed using median strategy')
  else:
    print(f'There are no Missing values in {col} attribute.')

print('\n')

# List of numerical columns
numeric_features = [
    'Age',                        # Age of the customer
    'NumberOfPersonVisiting',     # Total number of people accompanying the customer on the trip
    'PreferredPropertyStar',      # Preferred hotel rating by the customer
    'NumberOfTrips',              # Average number of trips the customer takes annually
    'NumberOfChildrenVisiting',   # Number of children below age 5 accompanying the customer
    'MonthlyIncome',              # Gross monthly income of the customer
    'PitchSatisfactionScore',     # Score indicating the customer's satisfaction with the sales pitch
    'NumberOfFollowups',          # Total number of follow-ups by the salesperson after the sales pitch
    'DurationOfPitch'             # Duration of the sales pitch delivered to the customer
]

# categorical column missing value treatment
# Get only categorical columns
categorical_cols = data.select_dtypes(exclude=np.number)
print('Examining if any missing values present in each categorical column')
# Lopping over each column and checking for null value and impute the null values
for col in categorical_cols:
  if data[col].isnull().sum() > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    data[col] = cat_imputer.fit_transform(data[[col]])
    print(f'Missing values for {col} attribute were imputed using mode strategy.')
  else:
    print(f'There are no Missing values in {col} attribute.')

# List of categorical features
categorical_features = [
    'CityTier',        # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3)
    'TypeofContact',   # The method by which the customer was contacted (Company Invited or Self Inquiry)
    'Occupation',      # Customer's occupation (e.g., Salaried, Freelancer)
    'Gender',          # Gender of the customer (Male, Female)
    'MaritalStatus',   # Marital status of the customer (Single, Married, Divorced)
    'Passport',        # Whether the customer holds a valid passport (0: No, 1: Yes)
    'OwnCar',          # Whether the customer owns a car (0: No, 1: Yes)
    'Designation',     # Customer's designation in their current organization
    'ProductPitched'   # The type of product pitched to the customer
]

# Define target variable
target = 'ProdTaken'

# Defining explanatory variables
X = data[numeric_features + categorical_features]

# Defining response variable
y = data[target]

# Split the dataset into train and test set
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,                # Predictor variables
    y,                # target variable
    test_size = 0.2,  # 20% of data into test set
    random_state = 42, # Ensure reproducibility by setting a fixed random seed
)

# Saving each train_test split in csv format
Xtrain.to_csv('Xtrain.csv', index=False)
Xtest.to_csv('Xtest.csv', index=False)
ytrain.to_csv('ytrain.csv', index=False)
ytest.to_csv('ytest.csv', index=False)

files = ['Xtrain.csv', 'Xtest.csv', 'ytrain.csv', 'ytest.csv']

# Upload each csv file to the data space created
for file_path in files:
  api.upload_file(
      path_or_fileobj = file_path,
      path_in_repo = file_path.split('/')[-1],    # Only file name
      repo_id = 'vineeth32/tourism-data',
      repo_type = 'dataset',
  )
