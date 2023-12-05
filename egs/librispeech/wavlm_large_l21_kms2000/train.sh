./zipformer/train.py \
  --world-size 4 \
  --num-epochs 40 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --full-libri 1 \
  --max-duration 1000 \
  --lr-epochs 3.5
