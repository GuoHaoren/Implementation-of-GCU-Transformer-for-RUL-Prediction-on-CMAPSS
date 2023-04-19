GPU_ID=0
DATASET='FD001'

MODES='Train'

MODEL_PATH='saved_model/model/test1'
SAVE_PATH='saved_model/log/test1'


SMOOTH_PARAM=0.8
EPOCH=200
DIM_EN=64
HEADS=4
NUM_ENC_LAYERS=1
NUM_FEATURES=14
DROP_OUT=0.1
BATCH_SIZE=20
TRAIN_SEQ_LEN=30
TEST_SEQ_LEN=30
MASK_PERCENTAGE=0.25
SPLIT_SLICES=5
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001
DECAY_STEP=100
DECAY_RATIO=0.5
PATCH=3


args=(--dataset "$DATASET" --modes "$MODES" --path "$MODEL_PATH"
      --epoch $EPOCH --dim_en $DIM_EN --head $HEADS --smooth_param $SMOOTH_PARAM
      --num_enc_layers $NUM_ENC_LAYERS --num_features $NUM_FEATURES 
      --drop_out $DROP_OUT --batch_size $BATCH_SIZE --train_seq_len $TRAIN_SEQ_LEN --patch_size $PATCH --test_seq_len $TEST_SEQ_LEN
      --slices $SPLIT_SLICES --LR $LEARNING_RATE --save_path $SAVE_PATH --weight_decay $WEIGHT_DECAY
      --decay_step $DECAY_STEP --decay_ratio $DECAY_RATIO)

CUDA_VISIBLE_DEVICES=$GPU_ID python3 main.py "${args[@]}"
