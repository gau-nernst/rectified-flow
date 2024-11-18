from huggingface_hub import hf_hub_download
from safetensors import safe_open


def load_hf_state_dict(repo_id: str, filename: str):
    local_path = hf_hub_download(repo_id, filename)
    with safe_open(local_path, framework="pt") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    return state_dict
