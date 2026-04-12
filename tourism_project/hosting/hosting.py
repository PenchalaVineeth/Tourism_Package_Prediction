from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv('HF_TOKEN'))
# Upload the deployment folder to hugging face space\
api.upload_folder(
    folder_path = 'tourism_project/deployment',       # Local folder contaniing all the files
    repo_id = 'vineeth32/Tourism-Package-Prediction', # Target repo
    repo_type = 'space',                              # Target repo type
    path_in_repo = ''
)
