import json

import safetensors.torch
import torch
from huggingface_hub import hf_hub_download


def load_hf_state_dict(repo_id: str, filename: str, prefix: str | None = None):
    local_path = hf_hub_download(repo_id, filename)

    if local_path.endswith(".safetensors"):
        state_dict = safetensors.torch.load_file(local_path)

    elif local_path.endswith(".pth"):
        state_dict = torch.load(local_path, map_location="cpu", mmap=True)

    elif local_path.endswith(".safetensors.index.json"):
        index = json.load(open(local_path))
        names = sorted(set(index["weight_map"].values()))

        base_path = (filename.rsplit("/", 1)[0] + "/") if "/" in filename else ""
        state_dict = dict()
        for name in names:
            state_dict.update(load_hf_state_dict(repo_id, base_path + name))

    else:
        raise RuntimeError(f"Unsupported file type. path={local_path}")

    if prefix is not None:
        state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}
    return state_dict
