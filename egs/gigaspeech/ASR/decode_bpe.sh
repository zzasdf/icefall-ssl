for ((avg=1; avg<=29; avg++)); do
  ./zipformer/decode_bpe.py \
    --epoch 30 \
    --avg $avg \
    --exp-dir ./zipformer/exp_XL_bpe \
    --max-duration 1000 \
    --decoding-method greedy_search \
    --metrics PER \
    --subset XL
done
