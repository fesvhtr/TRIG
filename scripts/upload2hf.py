from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_in_repo="output/t2i/omnigen.zip", 
    path_or_fileobj="/home/muzammal/Projects/TRIG/data/output/t2i/omnigen.zip",  
    repo_id="TRIG-bench/TRIG", 
    repo_type="dataset" 
)