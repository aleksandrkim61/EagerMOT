import os
from typing import Iterable, IO, Optional


def makedirs_if_new(path: str) -> bool:
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False


def close_files(files: Iterable[Optional[IO]]) -> None:
    for f in files:
        if f is not None:
            f.close()

def create_writable_file_if_new(folder: str, name: str):
    makedirs_if_new(folder)
    results_file = os.path.join(folder, name + '.txt')
    if os.path.isfile(results_file):
        return None
    return open(results_file, 'w')
