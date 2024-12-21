import safetensors.torch
from huggingface_hub import hf_hub_download


def load_hf_state_dict(repo_id: str, filename: str, prefix: str | None = None):
    local_path = hf_hub_download(repo_id, filename)
    state_dict = safetensors.torch.load_file(local_path)
    if prefix is not None:
        state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}
    return state_dict
