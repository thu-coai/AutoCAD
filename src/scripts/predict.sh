WORKING_DIR=/xxx


# model
MODEL_CONFIG=roberta-large

# data
TASK=nli

for SPLIT in train dev
do
    TEST_FILE=${WORKING_DIR}/data/${DATASET}/${SPLIT}.json
    PREDICT_FILE=${WORKING_DIR}/data/${DATASET}/${SPLIT}_predict.json

    # path to your fine-tuned classifier checkpint
    CKPT_PATH=/xxxx/yyy/*.ckpt

    python nlu.py \
        --task $TASK \
        --predict \
        --predict_file $PREDICT_FILE \
        --test_set $TEST_FILE \
        --load_dir $CKPT_PATH \
        --gpus $1 \
        --model_config $MODEL_CONFIG \
        --batch_size 1024
done