./zipformer/train_phone.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 23 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_XL_phone \
  --lr-epochs 1 \
  --subset XL \
  --max-duration 750 \
  --use-transducer 1 \
  --use-ctc 0
