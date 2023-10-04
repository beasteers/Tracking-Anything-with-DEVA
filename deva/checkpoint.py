import os

FILE_IDS = {
    'DEVA-propagation': 'https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth',
}
DRIVE_URL = 'https://docs.google.com/uc?export=download&confirm=t&id={}'

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.getenv("MODEL_DIR") or os.path.join(ROOT_DIR, 'saves')

def ensure_checkpoint(
        key='DEVA-propagation', path=None, file_id=None,
        model_dir=MODEL_DIR, download_url_pattern=DRIVE_URL,
):
    if not file_id:
        file_id = FILE_IDS.get(key)
        if not file_id:
            path = path or key  # assume key is the path
            key = os.path.splitext(os.path.basename(key))[0]  # and the key is the filename
            file_id = FILE_IDS[key]

    path = path or os.path.join(model_dir, f'{key}.pth')
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        print("No checkpoint found. Downloading...")
        def show_progress(i, size, total):
            print(f'downloading checkpoint to {path}: {i * size / total:.2%}', end="\r")
        
        import urllib.request
        url = download_url_pattern.format(file_id) if '/' not in file_id else file_id
        urllib.request.urlretrieve(url, path, show_progress)
    return path

