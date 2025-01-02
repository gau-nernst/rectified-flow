import argparse
import re
from pathlib import Path

from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
from lmdeploy.vl import load_image
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=Path, required=True)
parser.add_argument("--save_path", type=Path, required=True)
parser.add_argument("--model", default="OpenGVLab/InternVL2_5-8B-MPO-AWQ")
parser.add_argument("--prompt", nargs="+", default=["Describe this image"])
args = parser.parse_args()

pipe = pipeline(args.model, backend_config=TurbomindEngineConfig(session_len=8192))
gen_config = GenerationConfig(do_sample=True)

prompt_list = []
for prompt in args.prompt:
    if Path(prompt).exists():
        prompt = open(prompt, encoding="utf-8").read()
    prompt_list.append(prompt)
prompt = "\n".join(prompt_list)
print("USING THE FOLLOWING PROMPT")
print(prompt)

files = [str(x.relative_to(args.img_dir)) for x in args.img_dir.glob("**/*.jpg")]
files.sort()
args.save_path.parent.mkdir(exist_ok=True, parents=True)

with open(args.save_path, "w") as f:
    f.write("filename\tprompt\n")
    for filename in tqdm(files, dynamic_ncols=True):
        img_pil = Image.open(args.img_dir / filename)
        image = load_image(img_pil)
        response = pipe((prompt, image), gen_config=gen_config)
        output = re.sub(r"\s+", " ", response.text)
        f.write(f"{filename}\t{output}\n")
