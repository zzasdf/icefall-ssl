#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

"""
Usage:
  ./rnn_lm/compute_perplexity.py \
    --epoch 4 \
    --avg 2 \
    --use-averaged-model 1 \
    --lm-data ./data/lm_training_bpe_500/sorted_lm_data-test.pt
"""

import argparse
import logging
import math
from pathlib import Path

import torch
from dataset import get_dataloader
from model import RnnLmModel

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import AttributeDict, setup_logger, str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=49,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=20,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
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
        "--use-averaged-model",
        type=str2bool,
        default=False,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="rnn_lm/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--lm-data",
        type=str,
        help="Path to the LM test data for computing perplexity",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=500,
        help="Vocabulary size of the model",
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=2048,
        help="Embedding dim of the model",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=2048,
        help="Hidden dim of the model",
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of RNN layers the model",
    )

    parser.add_argument(
        "--tie-weights",
        type=str2bool,
        default=True,
        help="""True to share the weights between the input embedding layer and the
        last output linear layer
        """,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=150,
        help="Number of RNN layers the model",
    )

    parser.add_argument(
        "--max-sent-len",
        type=int,
        default=200,
        help="Number of RNN layers the model",
    )

    parser.add_argument(
        "--sos-id",
        type=int,
        default=1,
        help="SOS ID",
    )

    parser.add_argument(
        "--eos-id",
        type=int,
        default=1,
        help="EOS ID",
    )

    parser.add_argument(
        "--blank-id",
        type=int,
        default=0,
        help="Blank ID",
    )

    return parser


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lm_data = Path(args.lm_data)

    params = AttributeDict(vars(args))

    if params.use_averaged_model:
        params.suffix = "-use-averaged-model"
    else:
        params.suffix = ""

    if params.iter > 0:
        setup_logger(
            f"{params.exp_dir}/log-ppl/log-ppl-iter-{params.iter}-avg-{params.avg}{params.suffix}"
        )
    else:
        setup_logger(
            f"{params.exp_dir}/log-ppl/log-ppl-epoch-{params.epoch}-avg-{params.avg}{params.suffix}"
        )
    logging.info("Computing perplexity started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    logging.info("About to create model")
    model = RnnLmModel(
        vocab_size=params.vocab_size,
        embedding_dim=params.embedding_dim,
        hidden_dim=params.hidden_dim,
        num_layers=params.num_layers,
        tie_weights=params.tie_weights,
    )

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to(device)
    model.eval()
    num_param = sum([p.numel() for p in model.parameters()])
    num_param_requires_grad = sum(
        [p.numel() for p in model.parameters() if p.requires_grad]
    )

    logging.info(f"Number of model parameters: {num_param}")
    logging.info(
        f"Number of model parameters (requires_grad): "
        f"{num_param_requires_grad} "
        f"({num_param_requires_grad/num_param_requires_grad*100}%)"
    )

    logging.info(f"Loading LM test data from {params.lm_data}")
    test_dl = get_dataloader(
        filename=params.lm_data,
        is_distributed=False,
        params=params,
    )

    tot_loss = 0.0
    num_tokens = 0
    num_sentences = 0
    for batch_idx, batch in enumerate(test_dl):
        x, y, sentence_lengths = batch
        x = x.to(device)
        y = y.to(device)
        sentence_lengths = sentence_lengths.to(device)

        nll = model(x, y, sentence_lengths)
        loss = nll.sum().cpu().item()

        tot_loss += loss
        num_tokens += sentence_lengths.sum().cpu().item()
        num_sentences += x.size(0)

    ppl = math.exp(tot_loss / num_tokens)
    logging.info(
        f"total nll: {tot_loss}, num tokens: {num_tokens}, "
        f"num sentences: {num_sentences}, ppl: {ppl:.3f}"
    )


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


if __name__ == "__main__":
    main()
