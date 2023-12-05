import jsonlines
from tqdm import tqdm


with jsonlines.open('data/fbank/gigaspeech_cuts_M-sp1_1.jsonl') as reader:
    with jsonlines.open('output.jsonl', mode='w') as writer:
        for obj in tqdm(reader):
            obj["duration"] /= 1.1
            obj["supervisions"][0]["duration"] /= 1.1
            obj["id"] += "_sp1.1"
            obj["supervisions"][0]["id"] += "_sp1.1"
            writer.write(obj)
