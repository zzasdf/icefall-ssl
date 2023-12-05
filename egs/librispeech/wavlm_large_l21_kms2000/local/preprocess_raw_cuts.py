import os

import jsonlines
from tqdm import tqdm

dataset_parts = (
    # "dev-clean",
    # "dev-other",
    # "test-clean",
    # "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
)

for part in dataset_parts:
    with jsonlines.open(f"librispeech_cuts_{part}_raw.jsonl") as reader:
        with jsonlines.open(f"librispeech_cuts_{part}-sp1_1.jsonl", mode="w") as writer:
            for obj in tqdm(reader):
                obj["custom"] = {
                    "discrete_tokens": obj["supervisions"][0]["custom"][
                        "discrete_tokens"
                    ]
                }
                del obj["supervisions"][0]["custom"]

                # Speed perturb
                obj["duration"] /= 1.1
                obj["supervisions"][0]["duration"] /= 1.1
                obj["id"] += "_sp1.1"
                obj["supervisions"][0]["id"] += "_sp1.1"

                writer.write(obj)

os.system("rm *_raw.jsonl")
os.system("gzip *.jsonl")
