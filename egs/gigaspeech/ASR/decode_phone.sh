for ((avg=1; avg<=39; avg++)); do
  ./zipformer/decode_phone.py \
    --epoch 40 \
    --avg $avg \
    --exp-dir ./zipformer/exp_M_phone \
    --max-duration 1000 \
    --decoding-method greedy_search \
    --metrics PER
done
