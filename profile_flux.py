import time

import torch
import torch.nn.functional as F

from modelling import load_flux1
from modelling.mx import MXLinear
from modelling.nvfp4 import NVFP4Linear

seen_shapes = dict()
old_linear = F.linear
old_sdpa = F.scaled_dot_product_attention


def patched_linear(x, y, b=None):
    M = tuple(x.shape[:-1])
    N, K = y.shape
    key = (M, N, K)
    seen_shapes[key] = seen_shapes.get(key, 0) + 1
    return old_linear(x, y, b)


def patched_sdpa(q, k, v, *args, **kwargs):
    key = (q.shape, k.shape, v.shape)
    seen_shapes[key] = seen_shapes.get(key, 0) + 1
    return old_sdpa(q, k, v, *args, **kwargs)


# F.linear = patched_linear
# F.scaled_dot_product_attention = patched_sdpa

flux = load_flux1().cuda()

allowed_keys = [
    "img_attn",
    "img_mlp",
    "txt_attn",
    "txt_mlp",
    "linear1",
    "linear2",
]

qtype = "bf16"  # 0.41
# qtype = "mxfp8"  # 0.23
# qtype = "mxfp4"  # 0.20
# qtype = "mxfp4_nv"  # 0.20
# qtype = "nvfp4"

attn_impl = "pt"

for name, module in flux.named_modules():
    if hasattr(module, "attn_impl"):
        module.attn_impl = attn_impl

    if isinstance(module, torch.nn.Linear) and any(x in name for x in allowed_keys):
        if qtype == "mxfp8":
            MXLinear.convert(module, torch.float8_e4m3fn, compute_scale_method="nv")
        elif qtype == "mxfp4":
            MXLinear.convert(module, torch.float4_e2m1fn_x2, compute_scale_method="nv")
        elif qtype == "nvfp4":
            NVFP4Linear.install_calibration_hook(module)

inputs = torch.randn(1, 16, 128, 128, device="cuda", dtype=torch.bfloat16)
ts = torch.randn([1], device="cuda", dtype=torch.float)
txt = torch.randn(1, 512, 4096, device="cuda", dtype=torch.bfloat16)
y = torch.randn(1, 768, device="cuda", dtype=torch.bfloat16)

# NVFP4 requires calibration
if qtype == "nvfp4":
    with torch.no_grad():
        flux(inputs, ts, txt, y)
    NVFP4Linear.convert(flux)

flux.compile()
with torch.no_grad():
    flux(inputs, ts, txt, y)

    # warmup
    for _ in range(5):
        flux(inputs, ts, txt, y)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        flux(inputs, ts, txt, y)
    latency = (time.perf_counter() - t0) / 100
    print(f"{latency:.2f} s")

    # with torch.profiler.profile(with_stack=True) as prof:
    #     flux(inputs, ts, txt, y)
    # prof.export_chrome_trace(f"flux_{qtype}_{attn_impl}.json.gz")

# print("|".join(["M", "N", "K", "count"]))
# print("|".join(["---"] * 4))
# for key, count in sorted(seen_shapes.items()):
#     print("|".join(str(x) for x in key), "|", count, sep="")
