for ((avg=1; avg<=29; avg++)); do
  ./zipformer/decode_phone.py \
    --epoch 30 \
    --avg $avg \
    --exp-dir ./zipformer/exp_XL_phone \
    --max-duration 1000 \
    --decoding-method greedy_search \
    --metrics PER \
    --subset XL
done
