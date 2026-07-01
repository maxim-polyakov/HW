from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

data_dir = Path(__file__).resolve().parent / "data"
data_dir.mkdir(exist_ok=True)

for split in ["train", "test", "val"]:
    file = f"{split}.csv"
    cache_path = hf_hub_download(repo_id="mks-logic/gender_prediction", repo_type="dataset", filename=file)
    shutil.copy(cache_path, data_dir / file)