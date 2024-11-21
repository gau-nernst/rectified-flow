import safetensors.torch
from huggingface_hub import hf_hub_download


def load_hf_state_dict(repo_id: str, filename: str):
    local_path = hf_hub_download(repo_id, filename)
    state_dict = safetensors.torch.load_file(local_path)
    return state_dict
