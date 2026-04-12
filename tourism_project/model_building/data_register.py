# Exception handling in case repository not found
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
# Hugging face library for authentication and creating repo
from huggingface_hub import HfApi, create_repo
# Library to interact with operating system
import os

repo_id = 'vineeth32/tourism-data'
repo_type = 'dataset'

# Initialize API client
api = HfApi(token = os.getenv('HF_TOKEN'))

# Check if dataset space exists, if not create one
try:
  api.repo_info(repo_id=repo_id, repo_type=repo_type)
  print(f'Space {repo_id} already exists. Using it.')
except RepositoryNotFoundError:
  print(f'Space {repo_id} not found. Creating new space.')
  api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
  print(f'Space {repo_id} created.')

# Upload data folder to the space created earlier
api.upload_folder(
    folder_path = 'tourism_project/data',
    repo_id = repo_id,
    repo_type = repo_type
)
