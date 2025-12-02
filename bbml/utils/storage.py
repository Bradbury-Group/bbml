import hashlib
from pathlib import Path
import traceback

import requests


def get_file_md5(file_path: str|Path) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):  # 1MB
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_md5(data: bytes) -> str:
    hash_md5 = hashlib.md5()
    hash_md5.update(data)
    return hash_md5.hexdigest()


def get_str_md5(string: str, encoding='utf-8') -> str:
    hash_md5 = hashlib.md5()
    hash_md5.update(string.encode(encoding))
    return hash_md5.hexdigest()
    

def download_url(
    url,
    to_path:str|None=None,
    reraise:bool=False,
    timeout:int=60,
) -> str|None:
    try:
        if to_path is None:
            to_path = url.split("/")[-1]
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        with open(to_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB
                f.write(chunk)
        print(f"Downloaded to: {to_path}")
        return to_path
    except Exception as e:
        traceback.print_exc()
        print(f"Download error: {e}")
        if reraise:
            raise RuntimeError("Download Error") from e
        return None

