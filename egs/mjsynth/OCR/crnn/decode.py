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

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import crnn
import dataset
import jsonlines
import pybktree
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from utils import LevenshteinDistance, resizeNormalize, strLabelConverter

from icefall.utils import AttributeDict, str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="crnn/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="/star-data/yifan/IIIT5K",
        help="The experiment dir",
    )

    parser.add_argument("--lexicon", type=str, default="small", help="small or medium")

    parser.add_argument(
        "--threshold",
        type=int,
        default=20,
        help="Limit search to the nearest-neighbor candidates",
    )

    return parser


def decode_dataset(
    args: argparse.ArgumentParser,
    model: nn.Module,
    converter: strLabelConverter,
    transformer: resizeNormalize,
    device: torch.device,
) -> None:
    no_lexicon_cnt = 0
    small_lexicon_cnt = 0
    medium_lexicon_cnt = 0
    tot_cnt = 0
    with open(args.data_dir / "testdata.jsonl") as f:
        for test_data in jsonlines.Reader(f):
            label = test_data["GroundTruth"]
            image = Image.open(args.data_dir / test_data["ImgName"]).convert("L")
            image = transformer(image).to(device)
            image = image.view(1, *image.size())

            model.eval()
            nn_output = model(image)

            _, preds = nn_output.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)

            preds_size = torch.IntTensor([preds.size(0)])
            raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
            sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
            if sim_pred == label.lower():
                no_lexicon_cnt += 1

            if args.lexicon == "small":
                smallLexi = set(test_data["smallLexi"])
                small_tree = pybktree.BKTree(LevenshteinDistance)
                for word in smallLexi:
                    small_tree.add(word.lower())
                small_candidates = small_tree.find(sim_pred, args.threshold)
                small_candidates = [i[1] for i in small_candidates]
                cost = float("inf")
                for candidate in small_candidates:
                    text, text_length = converter.encode(candidate)
                    loss = F.ctc_loss(nn_output, text, preds_size, text_length)

                    if loss < cost:
                        small_pred = candidate
                        cost = loss

                if small_pred == label.lower():
                    small_lexicon_cnt += 1

                logging.info(
                    "%-20s => %-20s | %-20s | %-20s"
                    % (raw_pred, sim_pred, small_pred, label.lower())
                )
            elif args.lexicon == "medium":
                mediumLexi = set(test_data["mediumLexi"])
                medium_tree = pybktree.BKTree(LevenshteinDistance)
                for word in mediumLexi:
                    medium_tree.add(word.lower())
                medium_candidates = medium_tree.find(sim_pred, args.threshold)
                medium_candidates = [i[1] for i in medium_candidates]
                cost = float("inf")
                for candidate in medium_candidates:
                    text, text_length = converter.encode(candidate)
                    loss = F.ctc_loss(nn_output, text, preds_size, text_length)

                    if loss < cost:
                        medium_pred = candidate
                        cost = loss

                if medium_pred == label.lower():
                    medium_lexicon_cnt += 1

                logging.info(
                    "%-20s => %-20s | %-20s | %-20s"
                    % (raw_pred, sim_pred, medium_pred, label.lower())
                )

            tot_cnt += 1

            if tot_cnt % 500 == 0:
                print(
                    f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}: {tot_cnt:-5} | {3000:-5}"
                )

    if args.lexicon == "small":
        logging.info(
            f"Accuracy with small lexicon: {small_lexicon_cnt / tot_cnt * 100:.2f}%"
        )
    elif args.lexicon == "medium":
        logging.info(
            f"Accuracy with medium lexicon: {medium_lexicon_cnt / tot_cnt * 100:.2f}%"
        )
    else:
        logging.info(f"Accuracy without lexicon: {no_lexicon_cnt / tot_cnt * 100:.2f}%")


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.data_dir = Path(args.data_dir)
    args.alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    if not os.path.exists(args.exp_dir / "decode"):
        os.mkdir(args.exp_dir / "decode")

    if args.lexicon is not None:
        assert args.lexicon in ("small", "medium"), args.lexicon
        log_filename = (
            args.exp_dir
            / "decode"
            / f"log-decode-lexicon-{args.lexicon}-threshold-{args.threshold}-{date_time}"
        )
    else:
        log_filename = (
            args.exp_dir / "decode" / f"log-decode-without-lexicon-{date_time}"
        )

    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=logging.INFO,
    )

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    model = crnn.CRNN(32, 1, 37, 256)
    model_path = args.exp_dir / "crnn.pt"
    logging.info("loading pretrained model from %s" % str(model_path))
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    converter = strLabelConverter(args.alphabet)
    transformer = resizeNormalize((100, 32))

    decode_dataset(
        args=args,
        model=model,
        converter=converter,
        transformer=transformer,
        device=device,
    )


if __name__ == "__main__":
    main()
