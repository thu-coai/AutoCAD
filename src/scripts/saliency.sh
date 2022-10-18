WORKING_DIR=/xxx

MODEL_CONFIG=roberta-large

TASK=nli

for SPLIT in train dev
do
    TEST_PATH=${WORKING_DIR}/data/snli/ori_data/${SPLIT}.json

    # path to your fine-tuned classifier checkpint
    CKPT_PATH=xxx.ckpt

    OUTPUT_PATH=${WORKING_DIR}/data/snli/ori_data/${SPLIT}_saliency.json

    python nlu.py \
        --task $TASK \
        --saliency \
        --output_path $OUTPUT_PATH \
        --test_set $TEST_PATH \
        --load_dir $CKPT_PATH \
        --gpus $1 \
        --model_config $MODEL_CONFIG \
        --batch_size 64 \
        --saliency_mode gradient
done