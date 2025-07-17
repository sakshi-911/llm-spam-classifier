from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="spam_classifier.pth",                  # local file
    path_in_repo="spam_classifier.pth",                     # name it should have in the repo
    repo_id="sakshi-911/spam-classifier",                   # your repo
    repo_type="model"                                       # important
)