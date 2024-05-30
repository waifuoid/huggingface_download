from concurrent.futures import ThreadPoolExecutor
import json
import requests
import os
from tqdm import tqdm

session = requests.Session()


def get_all_file(repo_name: str, headers, datasets=True):
    if os.path.exists(f"repos/{repo_name}.json"):
        with open(f"repos/{repo_name}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        if datasets:
            url = f"https://huggingface.co/api/datasets/{repo_name}"
        else:
            url = f"https://huggingface.co/api/models/{repo_name}"
        response = session.get(url, headers=headers)
        data = response.json()
        os.makedirs(os.path.dirname(f"repos/{repo_name}.json"), exist_ok=True)
        with open(f"repos/{repo_name}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=4, ensure_ascii=False))
    all_files = []
    if os.path.exists(f"repos/{repo_name}.txt"):
        with open(f"repos/{repo_name}.txt", "r", encoding="utf-8") as f:
            for line in f:
                filename, url = line.strip().split("\t")
                all_files.append({"filename": filename, "url": url})
    else:
        if "siblings" in data:
            with open(f"repos/{repo_name}.txt", "w", encoding="utf-8") as f:
                for file in data["siblings"]:
                    if datasets:
                        url = f"https://huggingface.co/datasets/{repo_name}/resolve/main/{file['rfilename']}"
                    else:
                        url = f"https://huggingface.co/{repo_name}/resolve/main/{file['rfilename']}"
                    f.write(f"{file['rfilename']}\t{url}\n")
                    all_files.append({"filename": file['rfilename'], "url": url})
    return all_files


# 添加多线程下载与进度条
def download_file(url, file_name, save_path, headers):
    if os.path.exists(f"{save_path}/{file_name}"):
        print(f"{file_name} already exists")
        return
    os.makedirs(os.path.dirname(f"{save_path}/{file_name}"), exist_ok=True)
    with session.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        total_length = int(r.headers.get('content-length'))
        with open(f"{save_path}/{file_name}", "wb") as f:
            with tqdm(total=total_length, unit='B', unit_scale=True, unit_divisor=1024, desc=f"Downloading {file_name}") as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    print(f"Downloaded {file_name}")
    return


def download_repo(repo: str, save_path: str, headers, datasets=True):
    all_files = get_all_file(f"{repo}", headers, datasets)

    for file_data in all_files:
        file_name = file_data["filename"]
        url = file_data["url"]
        file_path = os.path.join(save_path, file_name)
        if os.path.exists(file_path):
            if file_path.endswith(".tar") and os.path.getsize(file_path) < 1024 * 1024:
                os.remove(file_path)
            else:
                print(f"{file_name} already exists")
                continue
        print(f"Start Download {file_name}")
        download_file(url, file_name, save_path, headers)
        print(f"Downloaded {file_name}")
    print("Download completed")


def download_part(url, start, end, save_path, file_name, headers, part_idx):
    part_path = os.path.join(save_path, f"{file_name}.part{part_idx}")

    # 检查已下载的进度
    resume_byte = 0
    if os.path.exists(part_path):
        resume_byte = os.path.getsize(part_path)

    if resume_byte > 0:
        start += resume_byte  # 调整起始点

    # 分段请求数据
    local_headers = headers.copy()
    local_headers['Range'] = f"bytes={start}-{end}"
    try:
        response = session.get(url, headers=local_headers, stream=True)
        response.raise_for_status()  # 检查响应状态码
        with open(part_path, 'ab' if resume_byte else 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except requests.RequestException as e:
        print(f"Error downloading part {part_idx}: {e}")

    return part_path


def combine_parts(file_parts, destination):
    # 合并下载的各个部分
    with open(destination, 'wb') as destination_file:
        for part in file_parts:
            with open(part, 'rb') as part_file:
                destination_file.write(part_file.read())
            os.remove(part)  # 删除临时文件


def download_file_multi_part(url, file_name, save_path, headers, min_size=1024*1024*10, num_threads=4):
    file_path = os.path.join(save_path, file_name)
    if os.path.exists(file_path):
        if file_path.endswith(".tar") and os.path.getsize(file_path) < 1024 * 1024:
            os.remove(file_path)
        else:
            print(f"{file_name} already exists")
            return

    response = session.head(url, headers=headers)
    file_size = int(response.headers.get('content-length', 0))
    if file_size == 0:
        raise Exception("Cannot get file size from server")
    print(f"Start Download {file_name} ({file_size} bytes)")
    if file_size <= min_size or num_threads == 1:
        # 文件小于最小分段大小或单线程下载
        file_parts = [download_part(url, 0, file_size - 1, save_path, file_name, headers, 0)]
    else:
        part_size = file_size // num_threads
        file_parts = []

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(download_part, url, i * part_size, (i + 1) * part_size - 1 if i < num_threads - 1 else file_size - 1, save_path, file_name, headers, i) for i in range(num_threads)]
            file_parts = [future.result() for future in futures]

    combine_parts(file_parts, file_path)
    print(f"Downloaded {file_name}")


def download_repo_multi_part(repo: str, save_path: str, headers):
    all_files = get_all_file(f"{repo}", headers)
    for file_data in all_files:
        file_name = file_data["filename"]
        url = file_data["url"]
        download_file_multi_part(url, file_name, save_path, headers, num_threads=8)
    print("Download completed")