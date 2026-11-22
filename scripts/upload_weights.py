"""Upload the trained weights in artifacts/ to the Hugging Face Hub (R7: no weights in git).

    huggingface-cli login            # paste a WRITE token from https://huggingface.co/settings/tokens
    python scripts/upload_weights.py # creates/updates  huggingface.co/<WEIGHTS_REPO>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from huggingface_hub import HfApi  # noqa: E402

from skb import config  # noqa: E402

api = HfApi()
api.create_repo(config.WEIGHTS_REPO, repo_type="model", exist_ok=True, private=False)
api.upload_folder(folder_path=str(config.ART_DIR), repo_id=config.WEIGHTS_REPO, repo_type="model",
                  commit_message="Upload trained Smart Keyboard weights")
print(f"Uploaded -> https://huggingface.co/{config.WEIGHTS_REPO}")
