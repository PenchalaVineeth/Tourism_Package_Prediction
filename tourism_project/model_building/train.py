# Library for manipulating data
import pandas as pd
# Libraries for data preprocessing and creating model pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# Libraries for training, tuning and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, recall_score
# Library to serialize model
import joblib
# Library for creating folder
import os
# Library for authenticating hugging face and to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
# Library for tracking results and parameters
import mlflow

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('mlops-tourism-tracking')

# Initialize API client
api = HfApi()

Xtrain_path = 'hf://datasets/vineeth32/tourism-data/Xtrain.csv'
Xtest_path = 'hf://datasets/vineeth32/tourism-data/Xtest.csv'
ytrain_path = 'hf://datasets/vineeth32/tourism-data/ytrain.csv'
ytest_path = 'hf://datasets/vineeth32/tourism-data/ytest.csv'

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Define target variable
target = 'ProdTaken'

# List of numerical features
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

# Set class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define xgb model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],    # number of tree to build
    'xgbclassifier__max_depth': [2, 3, 4],    # maximum depth of each tree
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each tree
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each level of a tree
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],    # learning rate
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],    # L2 regularization factor
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Scoring type to used to compare parameter combinations
scorer = metrics.make_scorer(metrics.precision_score)

with mlflow.start_run():
  # Hyperparameter tuning
  grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring=scorer, n_jobs=-1)
  grid_search.fit(Xtrain, ytrain)

  # Log all parameter combinations with their mean test scores
  results = grid_search.cv_results_
  for i in range(len(results['params'])):
    param_set = results['params'][i]
    mean_score = results['mean_test_score'][i]
    std_score = results['std_test_score'][i]

    # Log each combination as seperate mlflow run
    with mlflow.start_run(nested=True):
      mlflow.log_params(param_set)
      mlflow.log_metric('mean_test_score', mean_score)
      mlflow.log_metric('std_test_score', std_score)

  # Log best parameters seperately in main run
  mlflow.log_params(grid_search.best_params_)

  # Store and evaulate best model
  best_model = grid_search.best_estimator_

  # Predict on train and test data
  y_pred_train = best_model.predict(Xtrain)
  y_pred_test = best_model.predict(Xtest)

  # Probability predictions for precision
  y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
  y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]

  train_report = classification_report(ytrain, y_pred_train, output_dict=True)
  test_report = classification_report(ytest, y_pred_test, output_dict=True)

  mlflow.log_metrics({
      'train_accuracy': train_report['accuracy'],
      'train_precision': train_report['1']['precision'],
      'train_recall': train_report['1']['recall'],
      'train_f1-score': train_report['1']['f1-score'],
      'test_accuracy': test_report['accuracy'],
      'test_precision': test_report['1']['precision'],
      'test_recall': test_report['1']['recall'],
      'test_f1-score': test_report['1']['f1-score']
  })

  # Save the best model locally
  model_path = 'best_tourism_model.joblib'
  joblib.dump(best_model, model_path)

  # Log the model artifact
  mlflow.log_artifact(model_path, artifact_path='model')
  print(f'Model saved as artifact at: {model_path}')

  # Upload to Hugging face
  repo_id = 'vineeth32/tourism-model'
  repo_type = 'model'

  # Check if the space exists
  try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
  except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space.")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

  # Upload best model to the model space created
  api.upload_file(
      path_or_fileobj='best_tourism_model.joblib',
      path_in_repo='best_tourism_model.joblib',
      repo_id=repo_id,
      repo_type=repo_type
  )
