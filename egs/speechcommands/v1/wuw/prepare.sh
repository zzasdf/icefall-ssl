#!/usr/bin/env bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

nj=15
stage=-1
stop_stage=100

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/SpeechCommands1/speech_commands_v0.01
#	Speech Commands v0.01 dataset
#       From http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
#
#  - $dl_dir/SpeechCommands1/speech_commands_test_set_v0.01
#       Speech Commands test v0.01 dataset
#	From http://download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz

dl_dir=$PWD/download
enable_speed_perturb=False

. shared/parse_options.sh || exit 1

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  # If you have pre-downloaded them to
  #   /path/to/speech_commands_v0.01
  #   /path/to/speech_commands_test_set_v0.01
  # you can make a directory and create a symlink
  #   mkdir $dl_dir/SpeechCommands1
  #   cd $dl_dir/SpeechCommands1
  #   ln -sfv /path/to/speech_commands_v0.01 .
  #   ln -sfv /path/to/speech_commands_test_set_v0.01 .

  if [ ! -d $dl_dir/SpeechCommands1/speech_commands_v0.01 ] || [ ! -d $dl_dir/SpeechCommands1/speech_commands_test_set_v0.01 ]; then
    lhotse download speechcommands "1" $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare Speech Commands v0.01 manifest"
  # We assume that you have downloaded the speech_commands_v0.01 and speech_commands_test_set_v0.01 corpus
  # to $dl_dir/SpeechCommands1
  mkdir -p data/manifests
  if [ ! -e data/manifests/.speechcommands1.done ]; then
    lhotse prepare speechcommands "1" $dl_dir/SpeechCommands1 data/manifests
    touch data/manifests/.speechcommands1.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Compute fbank for Speech Commands v0.01"
  mkdir -p data/fbank
  if [ ! -e data/fbank/.speechcommands1.done ]; then
    ./local/compute_fbank_speechcommands1.py \
      --enable-speed-perturb=${enable_speed_perturb}
    touch data/fbank/.speechcommands1.done
  fi
fi

