./zipformer/train_debug.py \
  --world-size 1 \
  --num-epochs 40 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --full-libri 0 \
  --max-duration 1000 \
  --lr-epochs 10.5
