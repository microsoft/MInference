#!/usr/bin/bash
i=$(hostname | awk -F'-' '{print $2}')
NODE_RANK=$i
export NUM_NODES=1
export REUSE_TYPE="match"

export HF_TRUST_REMOTE_CODE=true
export HF_DATASETS_TRUST_REMOTE_CODE=true

export MASTER_ADDR="node-0"
export MASTER_PORT="12345"

export NNSCALER_HOME="/home/aiscuser/.conda/envs/mtrain/lib/python3.10/site-packages/nnscaler/autodist/"
export PYTHONPATH="${NNSCALER_HOME}:${PYTHONPATH}"

# -----------------------------------------------
# TODO: Basic Environment Settings
SEQUENCE_LENGTH=524288
export GPU_NAME=A100
export GPU_PER_NODE=4
export WORLD_SIZE=4
export GPU_SET="${GPU_NAME}_${WORLD_SIZE}"
export EXPR_HOME="/scratch/MInference/mtraining"
export NNSCALER_STORE="${EXPR_HOME}/experiments/expr_data_store/${GPU_SET}"
mkdir -p $NNSCALER_STORE
cd $EXPR_HOME

# ------------------------------------------
# /blob/nnscaler_store/MI300_8/minfer_qwen/qwen_fp090_512K_mini
export EXPR_DIR="minfer_qwen" # Name for the experiment set
export EXPR_NAME="qwen_fp090_512K_mini" # Name for the single experiment run
export MODEL_ID="Qwen/Qwen2.5-3B"
export DATASET_PATH="/scratch/nnscaler_store/prolong_fixed_filter_qwen_524288"
export MODEL_CONFIG_PATH="${EXPR_HOME}/model_configs/qwen2/lc_config_mini"
echo "Using model config path: $MODEL_CONFIG_PATH"
TRANSFER_CONFIG_DIR="none"
export TRAIN_ATTN_CONFIG_PATH="/scratch/MInference/mtraining/experiments/train_attn_configs/qwen_flex_090.yaml"
export ATTN_TYPE="minfer"

# ------------------------------------------
# Training Path settings
export TF_LOG_PATH="$NNSCALER_STORE/$EXPR_DIR/tf_logs"
export CKPT_PATH="$NNSCALER_STORE/$EXPR_DIR/$EXPR_NAME/checkpoints"
export COMPILE_PATH="$NNSCALER_STORE/compile_config/rank_${NODE_RANK}"
mkdir -p $TF_LOG_PATH
mkdir -p $CKPT_PATH
mkdir -p $COMPILE_PATH

# -------------------------------------------
# Training Settings
export GLOBAL_BATCH_SIZE=4 # TODO
export MICRO_BATCH_SIZE=1
export MEM_CONSTRAINT=72

export NUM_ITER=10
export NUM_EPOCH=0

export CKPT_SAVE_STEP=5
export CKPT_SAVE_EPOCH=0

export CHECK_RESUME=1
if [ "$CHECK_RESUME" -eq 1 ]; then
    CHECK_RESUME="--check_resume"
else
    CHECK_RESUME=""
fi

# -------------------------------------------
# Logging Path
export LOG_PATH="${NNSCALER_STORE}/${EXPR_DIR}/${EXPR_NAME}/rank_${NODE_RANK}"
mkdir -p $LOG_PATH
echo "Logging directed to $LOG_PATH/train.log"

export TRACE_STRATEGY="reuse_cache"
torchrun --nproc_per_node=$GPU_PER_NODE \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train.py  --plan_ngpus $WORLD_SIZE \
                    --runtime_ngpus $WORLD_SIZE \
                    --name $EXPR_NAME \
                    --seq_len $SEQUENCE_LENGTH \
                    --attn_type $ATTN_TYPE \
                    --train_attn_config_path $TRAIN_ATTN_CONFIG_PATH \
                    --reuse_type $REUSE_TYPE \
                    --model_id $MODEL_ID \
                    --n_iter $NUM_ITER \
                    --n_epochs $NUM_EPOCH \
                    --global_batch_size $GLOBAL_BATCH_SIZE \
                    --micro_batch_size $MICRO_BATCH_SIZE \
                    --dataset_path $DATASET_PATH \
                    --compile_save_path $COMPILE_PATH \
                    --tf_log_dir $TF_LOG_PATH \
                    --model_config_path $MODEL_CONFIG_PATH \
                    --ckpt_save_dir $CKPT_PATH \
                    --ckpt_n_step $CKPT_SAVE_STEP \
                    --ckpt_n_epoch $CKPT_SAVE_EPOCH \
                    --trace_strategy $TRACE_STRATEGY \
                    --transfer_config_dir $TRANSFER_CONFIG_DIR \
                    --mem_constraint $MEM_CONSTRAINT \
                    $CHECK_RESUME > $LOG_PATH/train.log 2>&1