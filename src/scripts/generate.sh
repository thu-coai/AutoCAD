WORKING_DIR=/home/wenjiaxin/AutoCAD

MODEL_CONFIG=t5-large

TASK=nli

for SPLIT in train dev
do
    TEST_PATH=${WORKING_DIR}/data/snli/rationale_mask/${SPLIT}_cad.json

    # path to your fine-tuned classifier checkpint
    CKPT_PATH=/xxxx/yyy/*.ckpt

    SAVE_FILE=${SPLIT}_generate.json

    python generator.py \
        --task $TASK \
        --generate \
        --test_set $TEST_PATH \
        --load_dir $CKPT_PATH \
        --save_file $SAVE_FILE \
        --gpus 0 \
        --model_config $MODEL_CONFIG \
        --batch_size 512
done