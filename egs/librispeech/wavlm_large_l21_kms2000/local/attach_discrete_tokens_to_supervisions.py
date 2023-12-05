import os

import jsonlines
from tqdm import tqdm

os.system("cp /mnt/lustre/sjtu/home/yfy62/librispeech_data/data/manifests/*.jsonl.gz .")
os.system("chmod -R 644 *.jsonl.gz")
os.system("gunzip *.gz")

dataset_parts = (
    # "dev-clean",
    # "dev-other",
    # "test-clean",
    # "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
)

with open(
    "/mnt/lustre/sjtu/home/yfy62/discrete_token_data/LibriSpeech/wavlm_large_l21_kms2000/out_quantized_sp0.9"
) as f:
    discrete_tokens = f.read().splitlines()

discrete_tokens_info = {}
for discrete_token in discrete_tokens:
    discrete_token = discrete_token.split(" ", 1)
    discrete_tokens_info[discrete_token[0]] = discrete_token[1]

for part in dataset_parts:
    with jsonlines.open(f"librispeech_supervisions_{part}.jsonl") as reader:
        with jsonlines.open(
            f"librispeech_supervisions_{part}_new.jsonl", mode="w"
        ) as writer:
            for obj in tqdm(reader):
                obj["custom"] = {"discrete_tokens": discrete_tokens_info[obj["id"]]}
                writer.write(obj)

os.system('for file in *_new.jsonl; do mv "$file" "${file%_new.jsonl}.jsonl"; done')
