from pathlib import Path

import argparse
import torch
from tqdm import tqdm
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("type", choices=("asr", "mt"))
parser.add_argument("--out", default=None)
args = parser.parse_args()

out_path = args.out or Path(args.checkpoint).with_suffix(f".{args.type}.pt").as_posix()
print(f"out_path: {out_path}")

chkpt = torch.load(args.checkpoint)
new_model_params = {}
new_model_cfg = {
    "_name": getattr(chkpt['cfg']['model'], f"{args.type}_arch"),
    "arch": getattr(chkpt['cfg']['model'], f"{args.type}_arch"),
}
model_yaml = getattr(chkpt['cfg']['model'], f"{args.type}_model_conf", None)
if model_yaml is not None:
    with open(model_yaml) as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg_dict = {key.replace("-", "_"): val for key, val in cfg_dict.items()}
        new_model_cfg.update(cfg_dict)

new_task_cfg = chkpt['cfg']['task'][f'{args.type}_task_cfg']


if args.type == "asr":
    for key, tensor in tqdm(chkpt["model"].items()):
        if key.startswith("encoder.asr_model."):
            new_model_params[key.removeprefix("encoder.asr_model.")] = tensor
elif args.type == "mt":
    for key, tensor in tqdm(chkpt["model"].items()):
        if key.startswith("encoder.mt_encoder."):
            new_model_params["encoder." + key.removeprefix("encoder.mt_encoder.")] = tensor
        elif key.startswith("decoder."):
            new_model_params[key] = tensor

new_chkpt = {"model": new_model_params, "cfg": {"model": argparse.Namespace(**new_model_cfg), "task": new_task_cfg}}
for k, v in chkpt.items():
    if k in ["model", "cfg"]:
        continue
    new_chkpt[k] = v
torch.save(new_chkpt, out_path)
