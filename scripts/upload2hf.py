from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_in_repo="output/t2i/janus.zip", 
    path_or_fileobj="janus.zip",  
    repo_id="fesvhtr/TRIG", 
    repo_type="dataset" 
