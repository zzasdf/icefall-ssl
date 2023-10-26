#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#              2022  The University of Electro-Communications (author: Teo Wen Shen)  # noqa
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
from pathlib import Path

from lhotse import CutSet, load_manifest

ARGPARSE_DESCRIPTION = """
This file displays duration statistics of utterances in a manifest.
You can use the displayed value to choose minimum/maximum duration
to remove short and long utterances during the training.

See the function `remove_short_and_long_utt()` in
pruned_transducer_stateless5/train.py for usage.
"""


def get_parser():
    parser = argparse.ArgumentParser(
        description=ARGPARSE_DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--manifest-dir", type=Path, help="Path to cutset manifests")

    return parser.parse_args()


def main():
    args = get_parser()

    for part in ["eval1", "eval2", "eval3", "valid", "excluded", "train"]:
        path = args.manifest_dir / f"csj_cuts_{part}.jsonl.gz"
        cuts: CutSet = load_manifest(path)

        print("\n---------------------------------\n")
        print(path.name + ":")
        cuts.describe()


if __name__ == "__main__":
    main()

"""
csj_cuts_eval1.jsonl.gz:
Cut statistics:
╒═══════════════════════════╤══════════╕
│ Cuts count:               │ 1023     │
├───────────────────────────┼──────────┤
│ Total duration (hh:mm:ss) │ 01:55:40 │
├───────────────────────────┼──────────┤
│ mean                      │ 6.8      │
├───────────────────────────┼──────────┤
│ std                       │ 2.7      │
├───────────────────────────┼──────────┤
│ min                       │ 0.2      │
├───────────────────────────┼──────────┤
│ 25%                       │ 4.9      │
├───────────────────────────┼──────────┤
│ 50%                       │ 7.7      │
├───────────────────────────┼──────────┤
│ 75%                       │ 9.0      │
├───────────────────────────┼──────────┤
│ 99%                       │ 10.0     │
├───────────────────────────┼──────────┤
│ 99.5%                     │ 10.0     │
├───────────────────────────┼──────────┤
│ 99.9%                     │ 10.0     │
├───────────────────────────┼──────────┤
│ max                       │ 10.0     │
├───────────────────────────┼──────────┤
│ Recordings available:     │ 1023     │
├───────────────────────────┼──────────┤
│ Features available:       │ 0        │
├───────────────────────────┼──────────┤
│ Supervisions available:   │ 1023     │
╘═══════════════════════════╧══════════╛
SUPERVISION custom fields:
Speech duration statistics:
╒══════════════════════════════╤══════════╤══════════════════════╕
│ Total speech duration        │ 01:55:40 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total speaking time duration │ 01:55:40 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total silence duration       │ 00:00:00 │ 0.00% of recording   │
╘══════════════════════════════╧══════════╧══════════════════════╛

---------------------------------

csj_cuts_eval2.jsonl.gz:
Cut statistics:
╒═══════════════════════════╤══════════╕
│ Cuts count:               │ 1025     │
├───────────────────────────┼──────────┤
│ Total duration (hh:mm:ss) │ 02:02:07 │
├───────────────────────────┼──────────┤
│ mean                      │ 7.1      │
├───────────────────────────┼──────────┤
│ std                       │ 2.5      │
├───────────────────────────┼──────────┤
│ min                       │ 0.1      │
├───────────────────────────┼──────────┤
│ 25%                       │ 5.9      │
├───────────────────────────┼──────────┤
│ 50%                       │ 7.9      │
├───────────────────────────┼──────────┤
│ 75%                       │ 9.1      │
├───────────────────────────┼──────────┤
│ 99%                       │ 10.0     │
├───────────────────────────┼──────────┤
│ 99.5%                     │ 10.0     │
├───────────────────────────┼──────────┤
│ 99.9%                     │ 10.0     │
├───────────────────────────┼──────────┤
│ max                       │ 10.0     │
├───────────────────────────┼──────────┤
│ Recordings available:     │ 1025     │
├───────────────────────────┼──────────┤
│ Features available:       │ 0        │
├───────────────────────────┼──────────┤
│ Supervisions available:   │ 1025     │
╘═══════════════════════════╧══════════╛
SUPERVISION custom fields:
Speech duration statistics:
╒══════════════════════════════╤══════════╤══════════════════════╕
│ Total speech duration        │ 02:02:07 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total speaking time duration │ 02:02:07 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total silence duration       │ 00:00:00 │ 0.00% of recording   │
╘══════════════════════════════╧══════════╧══════════════════════╛

---------------------------------

csj_cuts_eval3.jsonl.gz:
Cut statistics:
╒═══════════════════════════╤══════════╕
│ Cuts count:               │ 865      │
├───────────────────────────┼──────────┤
│ Total duration (hh:mm:ss) │ 01:26:44 │
├───────────────────────────┼──────────┤
│ mean                      │ 6.0      │
├───────────────────────────┼──────────┤
│ std                       │ 3.0      │
├───────────────────────────┼──────────┤
│ min                       │ 0.3      │
├───────────────────────────┼──────────┤
│ 25%                       │ 3.3      │
├───────────────────────────┼──────────┤
│ 50%                       │ 6.8      │
├───────────────────────────┼──────────┤
│ 75%                       │ 8.7      │
├───────────────────────────┼──────────┤
│ 99%                       │ 10.0     │
├───────────────────────────┼──────────┤
│ 99.5%                     │ 10.0     │
├───────────────────────────┼──────────┤
│ 99.9%                     │ 10.0     │
├───────────────────────────┼──────────┤
│ max                       │ 10.0     │
├───────────────────────────┼──────────┤
│ Recordings available:     │ 865      │
├───────────────────────────┼──────────┤
│ Features available:       │ 0        │
├───────────────────────────┼──────────┤
│ Supervisions available:   │ 865      │
╘═══════════════════════════╧══════════╛
SUPERVISION custom fields:
Speech duration statistics:
╒══════════════════════════════╤══════════╤══════════════════════╕
│ Total speech duration        │ 01:26:44 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total speaking time duration │ 01:26:44 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total silence duration       │ 00:00:00 │ 0.00% of recording   │
╘══════════════════════════════╧══════════╧══════════════════════╛

---------------------------------

csj_cuts_valid.jsonl.gz:
Cut statistics:
╒═══════════════════════════╤══════════╕
│ Cuts count:               │ 3743     │
├───────────────────────────┼──────────┤
│ Total duration (hh:mm:ss) │ 06:40:15 │
├───────────────────────────┼──────────┤
│ mean                      │ 6.4      │
├───────────────────────────┼──────────┤
│ std                       │ 3.0      │
├───────────────────────────┼──────────┤
│ min                       │ 0.1      │
├───────────────────────────┼──────────┤
│ 25%                       │ 3.9      │
├───────────────────────────┼──────────┤
│ 50%                       │ 7.4      │
├───────────────────────────┼──────────┤
│ 75%                       │ 9.0      │
├───────────────────────────┼──────────┤
│ 99%                       │ 10.0     │
├───────────────────────────┼──────────┤
│ 99.5%                     │ 10.0     │
├───────────────────────────┼──────────┤
│ 99.9%                     │ 10.1     │
├───────────────────────────┼──────────┤
│ max                       │ 11.8     │
├───────────────────────────┼──────────┤
│ Recordings available:     │ 3743     │
├───────────────────────────┼──────────┤
│ Features available:       │ 0        │
├───────────────────────────┼──────────┤
│ Supervisions available:   │ 3743     │
╘═══════════════════════════╧══════════╛
SUPERVISION custom fields:
Speech duration statistics:
╒══════════════════════════════╤══════════╤══════════════════════╕
│ Total speech duration        │ 06:40:15 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total speaking time duration │ 06:40:15 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total silence duration       │ 00:00:00 │ 0.00% of recording   │
╘══════════════════════════════╧══════════╧══════════════════════╛

---------------------------------

csj_cuts_excluded.jsonl.gz:
Cut statistics:
╒═══════════════════════════╤══════════╕
│ Cuts count:               │ 980      │
├───────────────────────────┼──────────┤
│ Total duration (hh:mm:ss) │ 00:56:06 │
├───────────────────────────┼──────────┤
│ mean                      │ 3.4      │
├───────────────────────────┼──────────┤
│ std                       │ 3.1      │
├───────────────────────────┼──────────┤
│ min                       │ 0.1      │
├───────────────────────────┼──────────┤
│ 25%                       │ 0.8      │
├───────────────────────────┼──────────┤
│ 50%                       │ 2.2      │
├───────────────────────────┼──────────┤
│ 75%                       │ 5.8      │
├───────────────────────────┼──────────┤
│ 99%                       │ 9.9      │
├───────────────────────────┼──────────┤
│ 99.5%                     │ 9.9      │
├───────────────────────────┼──────────┤
│ 99.9%                     │ 10.0     │
├───────────────────────────┼──────────┤
│ max                       │ 10.0     │
├───────────────────────────┼──────────┤
│ Recordings available:     │ 980      │
├───────────────────────────┼──────────┤
│ Features available:       │ 0        │
├───────────────────────────┼──────────┤
│ Supervisions available:   │ 980      │
╘═══════════════════════════╧══════════╛
SUPERVISION custom fields:
Speech duration statistics:
╒══════════════════════════════╤══════════╤══════════════════════╕
│ Total speech duration        │ 00:56:06 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total speaking time duration │ 00:56:06 │ 100.00% of recording │
├──────────────────────────────┼──────────┼──────────────────────┤
│ Total silence duration       │ 00:00:00 │ 0.00% of recording   │
╘══════════════════════════════╧══════════╧══════════════════════╛

---------------------------------

csj_cuts_train.jsonl.gz:
Cut statistics:
╒═══════════════════════════╤════════════╕
│ Cuts count:               │ 914151     │
├───────────────────────────┼────────────┤
│ Total duration (hh:mm:ss) │ 1695:29:43 │
├───────────────────────────┼────────────┤
│ mean                      │ 6.7        │
├───────────────────────────┼────────────┤
│ std                       │ 2.9        │
├───────────────────────────┼────────────┤
│ min                       │ 0.1        │
├───────────────────────────┼────────────┤
│ 25%                       │ 4.6        │
├───────────────────────────┼────────────┤
│ 50%                       │ 7.5        │
├───────────────────────────┼────────────┤
│ 75%                       │ 8.9        │
├───────────────────────────┼────────────┤
│ 99%                       │ 11.0       │
├───────────────────────────┼────────────┤
│ 99.5%                     │ 11.0       │
├───────────────────────────┼────────────┤
│ 99.9%                     │ 11.1       │
├───────────────────────────┼────────────┤
│ max                       │ 18.0       │
├───────────────────────────┼────────────┤
│ Recordings available:     │ 914151     │
├───────────────────────────┼────────────┤
│ Features available:       │ 0          │
├───────────────────────────┼────────────┤
│ Supervisions available:   │ 914151     │
╘═══════════════════════════╧════════════╛
SUPERVISION custom fields:
Speech duration statistics:
╒══════════════════════════════╤════════════╤══════════════════════╕
│ Total speech duration        │ 1695:29:43 │ 100.00% of recording │
├──────────────────────────────┼────────────┼──────────────────────┤
│ Total speaking time duration │ 1695:29:43 │ 100.00% of recording │
├──────────────────────────────┼────────────┼──────────────────────┤
│ Total silence duration       │ 00:00:00   │ 0.00% of recording   │
╘══════════════════════════════╧════════════╧══════════════════════╛
"""