from huggingface_download.hf_download import download_repo, download_file_multi_part
from huggingface_download.config import config

access_token = config.access_token
headers = {
    'Authorization': f'Bearer {access_token}'
}


repo = "litagin/reazon-speech-v2-denoised"
root_path = "W:\\datasets\\audio\\reazon-speech-v2-denoised"
datasets = True

repo = "01-ai/Yi-1.5-9B"  # modelscope/Yi-1.5-9B-Chat-AWQ 01-ai/Yi-1.5-34B 
root_path = "Y:\\ChatAI\\models\\Yi-1.5-9B"
datasets = False
if __name__ == "__main__":
    download_repo(repo, root_path, headers, datasets=datasets)
