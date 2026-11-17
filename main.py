from huggingface_download.hf_download import download_repo, download_file_multi_part
from huggingface_download.config import config

access_token = config.access_token
headers = {
    'Authorization': f'Bearer {access_token}'
}


# ericflo/unnaturalhermes-reflections-100k
# 1_instruction\\__Instruction
# nvidia/ChatQA2-Long-SFT-data
# nvidia/ChatQA-Training-Data
# 1_instruction\____RolePlay
# 4_rlhf&dpo&cpo
# 
# 
# 

repo_list = [
    "AaronZ345/GTSinger",
    # "espnet/ace-opencpop-segments",
    # "espnet/ace-kising-segments",
    # "projectlosangeles/Annotated-MIDI-Songs"
]

repo = "magic-mirror/minoricloud-galgame-vocal-mp3-collection-1989-2023"
root_path = "E:\\datasets\\audio\\magic-mirror\\minoricloud-galgame-vocal-mp3-collection-1989-2023"
datasets = True
folder_name = None  # "ZH"  # FR
use_mirror = True
workers = 2  # 并发下载线程数，1 为逐个下载；镜像站报 429 时调小
use_system_proxy = False  # False 时跳过系统代理，直连下载

# Qwen/Qwen2-7B


# repo = "ACE-Step/ACE-Step-v1-3.5B"
# root_path = "Y:\\music\\ACE-Step-v1-3.5B"
# datasets = False
# folder_name = None
# ArliAI/Mistral-Nemo-12B-ArliAI-RPMax-v1.1
# ArliAI/Llama-3.1-8B-ArliAI-RPMax-v1.1
# mistralai/Mistral-Nemo-Instruct-2407
# repo = "ArliAI/Mistral-Nemo-12B-ArliAI-RPMax-v1.1"  # modelscope/Yi-1.5-9B-Chat-AWQ 01-ai/Yi-1.5-34B 
# root_path = "Y:\\ChatAI\\models\\Mistral-Nemo-12B-ArliAI-RPMax-v1.1"
# datasets = False
if __name__ == "__main__":
    # download_repo(repo, root_path, headers, datasets=datasets, folder_name=folder_name)
    for repo in repo_list:
        root_path = f"E:\\datasets\\svs\\{repo.split('/')[-1]}"
        download_repo(repo, root_path, headers, datasets=datasets, folder_name=folder_name, use_mirror=use_mirror, workers=workers,
                      use_system_proxy=use_system_proxy)
