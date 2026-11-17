from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import json
import random
import time
import requests
from requests.adapters import HTTPAdapter
import os
from tqdm import tqdm

session = requests.Session()


def get_all_file(repo_name: str, headers, datasets=True, folder_name=None, use_mirror=False):
    if os.path.exists(f"repos/{repo_name}.json"):
        with open(f"repos/{repo_name}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        if datasets:
            if use_mirror:
                url = f"https://hf-mirror.com/api/datasets/{repo_name}"
            else:
                url = f"https://huggingface.co/api/datasets/{repo_name}"
        else:
            if use_mirror:
                url = f"https://hf-mirror.com/api/models/{repo_name}"
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
                if folder_name and folder_name not in filename:
                    continue
                all_files.append({"filename": filename, "url": url})
    else:
        if "siblings" in data:
            with open(f"repos/{repo_name}.txt", "w", encoding="utf-8") as f:
                for file in data["siblings"]:
                    if datasets:
                        if use_mirror:
                            url = f"https://hf-mirror.com/datasets/{repo_name}/resolve/main/{file['rfilename']}"
                        else:
                            url = f"https://huggingface.co/datasets/{repo_name}/resolve/main/{file['rfilename']}"
                    else:
                        if use_mirror:
                            url = f"https://hf-mirror.com/models/{repo_name}/resolve/main/{file['rfilename']}"
                        else:
                            url = f"https://huggingface.co/{repo_name}/resolve/main/{file['rfilename']}"
                    if folder_name and folder_name not in file['rfilename']:
                        continue
                    f.write(f"{file['rfilename']}\t{url}\n")
                    all_files.append({"filename": file['rfilename'], "url": url})
    return all_files


def _noop(*args, **kwargs):
    pass


def _parse_total_from_response(r, resume_byte):
    """从 GET 响应推断文件总大小。响应被压缩 (Content-Encoding) 时无法推断，返回 None。"""
    encoding = r.headers.get('content-encoding', '').lower()
    if encoding and encoding != 'identity':
        return None
    if r.status_code == 206:
        content_range = r.headers.get('content-range', '')  # bytes 100-199/200
        if '/' in content_range:
            total = content_range.rsplit('/', 1)[-1].strip()
            if total.isdigit():
                return int(total)
    content_length = r.headers.get('content-length')
    if content_length and content_length.isdigit():
        return resume_byte + int(content_length) if r.status_code == 206 else int(content_length)
    return None


def download_file(url, file_name, save_path, headers, known_total_length, quiet=False):
    """
    Downloads a single file using GET, supports resume, uses known size.
    Raises FileSizeMismatchError if local > known before download.
    Raises DownloadVerificationError if download completes with wrong size.
    Returns the file's total size (known, inferred from the GET response, or the final local size).
    quiet=True 时不打印日志、不显示单文件进度条（多线程模式使用）。
    """
    log = _noop if quiet else print
    file_path = os.path.join(save_path, file_name)
    resume_byte = 0

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        resume_byte = os.path.getsize(file_path)
        log(f"找到已存在文件 {file_name}，大小: {resume_byte} 字节")

        if known_total_length is not None:
            if resume_byte == known_total_length:
                log(f"{file_name} 已完全下载 (大小匹配记录: {known_total_length})。")
                return known_total_length
            elif resume_byte > known_total_length:
                raise FileSizeMismatchError(file_path, resume_byte, known_total_length)

    log(f"准备下载/续传 {file_name} (从字节 {resume_byte})")

    local_headers = headers.copy()
    if resume_byte > 0:
        local_headers['Range'] = f"bytes={resume_byte}-"

    try:
        with session.get(url, stream=True, headers=local_headers, timeout=(15, 300)) as r: # connect_timeout, read_timeout
            if r.status_code == 416:
                log(f"服务器返回 416 (Range Not Satisfiable) for {file_name}。")
                current_size = os.path.getsize(file_path) # Re-check size
                if known_total_length is not None and current_size != known_total_length:
                     log(f"警告: 416 状态，但本地大小 {current_size} 与记录大小 {known_total_length} 不符。文件可能损坏或记录过时。")
                     raise DownloadVerificationError(file_path, current_size, known_total_length)
                elif known_total_length is None:
                     log(f"警告: 416 状态，但无本地记录大小比较。当前大小 {current_size}。")
                log(f"假定文件 {file_name} 已完成。")
                return current_size

            r.raise_for_status()

            # 服务器忽略了 Range 头，返回了完整文件：从头重写，避免追加导致文件损坏
            if resume_byte > 0 and r.status_code == 200:
                log(f"服务器不支持续传 {file_name}，将从头下载。")
                resume_byte = 0
            mode = "ab" if resume_byte > 0 else "wb"

            inferred_total = _parse_total_from_response(r, resume_byte)
            expected_total = known_total_length if known_total_length is not None else inferred_total
            if known_total_length is None and inferred_total is not None:
                log(f"从 GET 响应头获得总大小: {inferred_total}")

            with open(file_path, mode) as f:
                with tqdm(total=expected_total, initial=resume_byte, unit='B',
                          unit_scale=True, unit_divisor=1024, disable=quiet,
                          desc=f"下载 {os.path.basename(file_name)}", miniters=1, ncols=100) as pbar:
                    for chunk in r.iter_content(chunk_size=8192 * 4):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            final_size = os.path.getsize(file_path)
            log(f"{file_name} 下载流结束，最终本地大小: {final_size} 字节。")
            if expected_total is not None and final_size != expected_total:
                raise DownloadVerificationError(file_path, final_size, expected_total)
            return final_size

    except requests.exceptions.RequestException as e:
        log(f"网络或请求错误下载 {file_name}: {e}")
        raise
    except IOError as e:
        log(f"文件读写错误处理 {file_name}: {e}")
        raise


class FileSizeMismatchError(Exception):
    """Local file size > known size, requires verification."""
    def __init__(self, filepath, local_size, known_size):
        self.filepath = filepath
        self.local_size = local_size
        self.known_size = known_size
        super().__init__(f"Local file '{filepath}' size ({local_size}) > known size ({known_size}). Needs verification.")

class DownloadVerificationError(Exception):
    """Download completed, but final size != expected size."""
    def __init__(self, filepath, final_size, expected_size):
        self.filepath = filepath
        self.final_size = final_size
        self.expected_size = expected_size
        super().__init__(f"Verification failed for '{filepath}': Final size ({final_size}) != Expected size ({expected_size}).")


class _SizeInfoStore:
    """线程安全的 sizeinfo.json 读写，按时间间隔批量落盘，避免每个小文件都重写一次 JSON。"""
    def __init__(self, json_path, save_interval=5.0):
        self.json_path = json_path
        self.data = load_size_info(json_path)
        self.lock = Lock()
        self.save_interval = save_interval
        self.dirty = False
        self.last_save = time.monotonic()

    def get(self, name):
        with self.lock:
            return self.data.get(name)

    def set(self, name, size):
        with self.lock:
            if self.data.get(name) == size:
                return
            self.data[name] = size
            self.dirty = True
            if time.monotonic() - self.last_save >= self.save_interval:
                self._save_locked()

    def flush(self):
        with self.lock:
            if self.dirty:
                self._save_locked()

    def _save_locked(self):
        save_size_info(self.json_path, self.data)
        self.dirty = False
        self.last_save = time.monotonic()


def _process_file(file_data, save_path, headers, store, retry, head_check, quiet):
    """下载单个文件（含重试与大小校验）。返回 True 表示成功/已存在，False 表示失败。"""
    # 多线程模式下只输出错误信息，并通过 tqdm.write 避免打乱总进度条
    log = _noop if quiet else print
    warn = tqdm.write if quiet else print

    file_name = file_data.get("filename")
    url = file_data.get("url")
    if not file_name or not url:
        warn(f"错误: 文件信息不完整，跳过: {file_data}")
        return False

    known_size = store.get(file_name)

    # 可选：下载前用 HEAD 获取大小。默认关闭，因为 GET 响应头就能拿到大小，省一次请求
    if known_size is None and head_check:
        log(f"文件 {file_name} 大小未知，尝试 HEAD 请求...")
        server_size = get_server_size(url, headers, verbose=not quiet)
        if server_size is not None:
            known_size = server_size
            store.set(file_name, known_size)

    retry_count_download = 0
    while retry_count_download < retry:
        try:
            log(f"\n--- 开始处理: {file_name} (预期大小: {known_size}) ---")
            total = download_file(url, file_name, save_path, headers, known_size, quiet=quiet)
            if total is not None:
                store.set(file_name, total)
            log(f"--- 成功处理: {file_name} ---")
            return True

        # Handle verification when local > known
        except FileSizeMismatchError as e:
            warn(f"{file_name}: 本地文件 ({e.local_size} bytes) > 记录大小 ({e.known_size} bytes)，验证服务器大小...")
            server_size_verify = get_server_size(url, headers, verbose=not quiet)

            if server_size_verify is not None and server_size_verify == e.local_size:
                # Local file matches NEW server size. Old record was stale.
                log("本地大小与当前服务器大小匹配。更新记录，文件视为已完成。")
                store.set(file_name, server_size_verify)
                return True

            if server_size_verify is None:
                log("无法通过 HEAD 验证服务器大小。为安全起见，删除本地过大文件并重试下载。")
            elif server_size_verify == e.known_size:
                log("服务器大小确认旧记录。本地文件损坏。将删除并重试下载。")
            else:
                log("服务器大小与本地和记录均不符。文件已更改。将删除本地，更新记录，重试下载。")
                store.set(file_name, server_size_verify)
                known_size = server_size_verify

            try:
                os.remove(e.filepath)
            except OSError as del_e:
                warn(f"错误: 删除文件 {e.filepath} 失败: {del_e}。放弃。")
                return False

            retry_count_download += 1
            if retry_count_download >= retry:
                warn(f"放弃 {file_name}")
                return False
            time.sleep(random.uniform(1, 3))

        # Handle post-download verification failure
        except DownloadVerificationError as e:
            retry_count_download += 1
            warn(f"{file_name}: 大小校验失败 ({e.final_size} != {e.expected_size})，删除并重试 ({retry_count_download}/{retry})")
            try: os.remove(e.filepath)
            except OSError as del_e: warn(f"警告: 删除校验失败文件 {e.filepath} 出错: {del_e}")
            if retry_count_download >= retry:
                warn(f"放弃 {file_name}")
                return False
            time.sleep(random.uniform(1, 3) * retry_count_download)

        # Handle other general errors
        except Exception as e:
            retry_count_download += 1
            warn(f"下载 {file_name} 时发生错误 (尝试 {retry_count_download}/{retry}): {type(e).__name__} - {e}")
            if retry_count_download >= retry:
                warn(f"放弃 {file_name}")
                return False
            time.sleep(random.uniform(1, 5) * retry_count_download)

    return False


def download_repo(repo: str, save_path: str, headers, datasets=True, folder_name=None, retry=10, use_mirror=False,
                  workers=8, head_check=False, use_system_proxy=True):
    """
    Downloads repository files, using sizeinfo.json for size checks and resume.
    workers: 并发下载线程数。workers=1 时为原来的逐个下载模式（带单文件进度条和详细日志）。
    head_check: 下载前是否先发 HEAD 请求获取大小（默认关闭，GET 响应头即可获得大小）。
    use_system_proxy: False 时忽略系统代理（HTTP(S)_PROXY 环境变量和 Windows 系统代理设置），直连下载。
    """
    # trust_env=False 会让 requests 不读取环境变量/系统代理
    session.trust_env = use_system_proxy
    print(f"系统代理: {'使用' if use_system_proxy else '不使用（直连）'}")
    sizeinfo_json_path = os.path.join(save_path, "sizeinfo.json")
    store = _SizeInfoStore(sizeinfo_json_path)
    print(f"从 {sizeinfo_json_path} 加载了 {len(store.data)} 条文件大小记录。")

    retry_count_list = 0
    all_files = []
    while retry_count_list < retry:
        try:
            all_files = get_all_file(f"{repo}", headers, datasets, folder_name, use_mirror=use_mirror)
            break
        except Exception as e:
            retry_count_list += 1; print(f"获取列表失败 (尝试 {retry_count_list}/{retry}): {e}")
            if retry_count_list == retry: print("无法获取列表。退出。"); return
            time.sleep(random.uniform(1, 3) * retry_count_list); print("重试...")
    if not all_files: print("文件列表为空。"); return
    print(f"成功获取到 {len(all_files)} 个文件。")

    # 本地已存在且大小与记录一致的文件直接跳过，不发任何请求
    todo = []
    already_done = 0
    for file_data in all_files:
        name = file_data.get("filename")
        known = store.data.get(name) if name else None
        path = os.path.join(save_path, name) if name else None
        if known is not None and os.path.isfile(path) and os.path.getsize(path) == known:
            already_done += 1
        else:
            todo.append(file_data)
    print(f"已完成 {already_done} 个，待下载 {len(todo)} 个，线程数 {workers}。")
    print("-" * 20)

    files_processed = already_done
    files_failed = []
    try:
        if workers <= 1:
            for file_data in todo:
                if _process_file(file_data, save_path, headers, store, retry, head_check, quiet=False):
                    files_processed += 1
                else:
                    files_failed.append(file_data.get("filename"))
        else:
            _ensure_pool_size(workers)
            with ThreadPoolExecutor(max_workers=workers) as executor, \
                    tqdm(total=len(todo), unit="file", desc=repo.split('/')[-1], ncols=100) as pbar:
                futures = {executor.submit(_process_file, fd, save_path, headers, store, retry, head_check, True): fd
                           for fd in todo}
                for future in as_completed(futures):
                    name = futures[future].get("filename")
                    try:
                        ok = future.result()
                    except Exception as e:
                        tqdm.write(f"处理 {name} 时出现未预期错误: {type(e).__name__} - {e}")
                        ok = False
                    if ok:
                        files_processed += 1
                    else:
                        files_failed.append(name)
                    pbar.update(1)
                    pbar.set_postfix(failed=len(files_failed))
    finally:
        store.flush()

    # --- Final Summary ---
    print("\n" + "=" * 20)
    print("仓库下载处理完成。")
    print(f"总文件数: {len(all_files)}")
    print(f"成功处理/已存在文件数: {files_processed}")
    print(f"失败文件数: {len(files_failed)}")
    if files_failed: print(f"失败的文件列表: {files_failed}")
    print("=" * 20)


def _ensure_pool_size(workers):
    """让 requests 连接池容量不小于线程数，否则多余线程会反复新建连接。"""
    adapter = HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    session.mount("https://", adapter)
    session.mount("http://", adapter)


def get_server_size(url, headers, timeout=15, verbose=True):
    """
    Uses HEAD request to get file size from server. Returns size (int) or None.
    """
    log = print if verbose else _noop
    try:
        log(f"发送 HEAD 请求到: {url}")
        with session.head(url, headers=headers, timeout=timeout, allow_redirects=True) as r:
            r.raise_for_status()
            content_length = r.headers.get('content-length')
            if content_length:
                try:
                    size = int(content_length)
                    log(f"HEAD 请求成功，获取到 Content-Length: {size}")
                    return size
                except ValueError:
                    log(f"警告: 无法将 Content-Length '{content_length}' 解析为整数。")
                    return None
            else:
                # Some servers might not support HEAD or not return Content-Length for it
                log(f"警告: HEAD 请求成功，但未返回 Content-Length 头。")
                return None
    except requests.exceptions.Timeout:
         log(f"警告: HEAD 请求超时 ({timeout}s)。")
         return None
    except requests.exceptions.RequestException as e:
        log(f"警告: HEAD 请求获取 {url} 大小失败: {e}")
        return None


def load_size_info(json_path):
    """从 JSON 文件加载文件大小信息。"""
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict): # 确保加载的是字典
                    return data
                else:
                    print(f"警告: {json_path} 格式无效（不是JSON对象/字典）。将忽略。")
                    return {}
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告: 无法加载或解析 {json_path}: {e}。将视为空白信息。")
        except Exception as e:
            print(f"警告: 加载 {json_path} 时发生未知错误: {e}。")
    return {} # 文件不存在或加载失败，返回空字典


def save_size_info(json_path, data):
    """将文件大小信息保存到 JSON 文件。"""
    try:
        # 确保目标目录存在
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        # 使用 'w' 模式写入，覆盖旧文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4) # indent 参数使JSON文件更易读
        # print(f"文件大小信息已保存到 {json_path}") # 可以取消注释以获得详细日志
    except IOError as e:
        print(f"警告: 无法保存大小信息到 {json_path}: {e}")
    except Exception as e:
        print(f"警告: 保存大小信息到 {json_path} 时发生未知错误: {e}")


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