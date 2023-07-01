# Copyright    2023  Xiaomi Corp.        (authors: Yifan Yang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path

import crnn
import dataset
import jsonlines
import torch
import utils
from PIL import Image


def test():
    model_path = "./exp/crnn.pt"
    root_dir = Path("/star-data/yifan/IIIT5K")
    test_data = Path("/star-data/yifan/IIIT5K/testdata.jsonl")
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"

    model = crnn.CRNN(32, 1, 37, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    logging.info("loading pretrained model from %s" % model_path)
    model.load_state_dict(torch.load(model_path))

    converter = utils.strLabelConverter(alphabet)

    transformer = dataset.resizeNormalize((100, 32))

    true_cnt = 0
    tot_cnt = 0
    with open(test_data) as f:
        for test_data in jsonlines.Reader(f):
            label = test_data["GroundTruth"]
            image = Image.open(root_dir / test_data["ImgName"]).convert("L")
            image = transformer(image)
            if torch.cuda.is_available():
                image = image.cuda()
            image = image.view(1, *image.size())

            model.eval()
            preds = model(image)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)

            preds_size = torch.IntTensor([preds.size(0)])
            raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
            sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
            logging.info("%-20s => %-20s | %s" % (raw_pred, sim_pred, label.lower()))
            if sim_pred == label.lower():
                true_cnt += 1
            tot_cnt += 1

    logging.info(f"Accuracy: {true_cnt / tot_cnt * 100:.2f}%")


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    test()


if __name__ == "__main__":
    main()
