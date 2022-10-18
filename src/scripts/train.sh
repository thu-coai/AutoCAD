WORKING_DIR=/home/wenjiaxin/AutoCAD



MODEL_CONFIG=roberta-large

TASK=nli

DATASET=snli

TRAIN_PATH=${WORKING_DIR}/data/snli/ori_data/train.json
DEV_PATH=${WORKING_DIR}/data/snli/ori_data/dev.json

# save_dir
SAVE_DIR=${WORKING_DIR}/results/roberta-large

echo "Using GPUs: [$1]"
echo "TASK: $TASK"

TOKENIZERS_PARALLELISM=false python nlu.py \
    --task $TASK \
    --train_set $TRAIN_PATH \
    --dev_set $DEV_PATH \
    --save_dir $SAVE_DIR \
    --lr 1e-5 \
    --max_epochs 20 \
    --gpus $1 \
    --model_config $MODEL_CONFIG \
    --warm_up 0 \
    --weight_decay 0.1 \
    --batch_size 32