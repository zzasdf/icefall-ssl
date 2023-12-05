./zipformer/train.py \
  --world-size 4 \
  --num-epochs 20 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --subset XL \
  --max-duration 1000 \
  --lr-epochs 0.3333
