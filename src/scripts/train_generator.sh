WORKING_DIR=/home/wenjiaxin/AutoCAD

MODEL_CONFIG=t5-large

TASK=nli


TRAIN_PATH=${WORKING_DIR}/data/snli/rationale_mask/train.json
DEV_PATH=${WORKING_DIR}/data/snli/rationale_mask/dev.json

# save_dir
SAVE_DIR=${WORKING_DIR}/results/t5-large

LR=1e-5
ALPHA=1
WARMUP=0.0


echo "TASK: $TASK"

TOKENIZERS_PARALLELISM=false python generator.py \
    --task $TASK \
    --train_set $TRAIN_PATH \
    --dev_set $DEV_PATH \
    --save_dir $SAVE_DIR \
    --train \
    --lr ${LR} \
    --max_epochs 10 \
    --gpus 1 \
    --model_config $MODEL_CONFIG \
    --warm_up ${WARMUP} \
    --weight_decay 0.1 \
    --batch_size 8 \
    --alpha $ALPHA