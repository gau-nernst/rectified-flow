# https://github.com/black-forest-labs/flux/blob/805da8571a0b49b6d4043950bd266a65328c243b/src/flux/modules/image_embedders.py#L66

import timm
import torch
from torch import nn

from ..utils import load_hf_state_dict


def load_flux_redux(dtype: torch.dtype = torch.bfloat16):
    siglip = timm.create_model(
        "vit_so400m_patch14_siglip_378.webli",
        pretrained=True,
        dynamic_img_size=True,
    )  # 428M params
    siglip.forward = siglip.forward_features  # with 378x378 input, output is (B, 729, 1152)

    in_dim = 1152  # siglip
    out_dim = 4096  # t5
    with torch.device("meta"):
        redux = nn.Sequential()  # 64.5M params
        redux.redux_up = nn.Linear(in_dim, out_dim * 3)
        redux.silu = nn.SiLU()
        redux.redux_down = nn.Linear(out_dim * 3, out_dim)

    state_dict = load_hf_state_dict("black-forest-labs/FLUX.1-Redux-dev", "flux1-redux-dev.safetensors")
    redux.load_state_dict(state_dict, assign=True)
    return nn.Sequential(siglip, redux).to(dtype=dtype)
