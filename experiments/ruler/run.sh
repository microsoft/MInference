#!/bin/bash
# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

export TOKENIZERS_PARALLELISM=false
RULER_PATH=$(dirname $0)
python -c "import nltk; nltk.download('punkt')"

SEQ_LENGTHS=(
    4096
    # 8192
    # 16384
    # 32768
    # 65536
    # 131072
)

TASKS=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)

# Experiment Setup
NUM_SAMPLES=25
TEMPERATURE="0.0"
TOP_P="1.0"
TOP_K="32"

# The model
MODEL_NAME=$1
BENCHMARK="synthetic"
MODEL_TEMPLATE_TYPE="base"
MODEL_FRAMEWORK=$2

# MInference
STARTING_LAYER=-1
KV_CACHE_CPU="false"
USE_SNAPKV="false"
TRUST_REMOTE_CODE="true"

if [ "${MODEL_FRAMEWORK}" == "minference" ]; then
    MINFERENCE_PARAMS="--starting_layer ${STARTING_LAYER}"

    if [ -n "${CONFIG_PATH}" ]; then
        MINFERENCE_PARAMS="${MINFERENCE_PARAMS} --config_path ${CONFIG_PATH}"
    fi

    if [ "${USE_SNAPKV}" == "true" ]; then
        MINFERENCE_PARAMS="${MINFERENCE_PARAMS} --use_snapkv"
    fi

    echo "MInference enabled with params: ${MINFERENCE_PARAMS}"
fi

if [ "${TRUST_REMOTE_CODE}" == "true" ]; then
    EXTRA_PARAMS="${EXTRA_PARAMS} --trust_remote_code"
fi

if [ "${KV_CACHE_CPU}" == "true" ]; then
    EXTRA_PARAMS="${EXTRA_PARAMS} --kv_cache_cpu --kv_cache_cpu_device cpu"
fi

# Gpu and output path
GPUS="1" # GPU size for tensor_parallel.
ROOT_DIR=$3 # the path that stores generated task samples and model predictions.

for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do

    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}_${MODEL_FRAMEWORK}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}

    for TASK in "${TASKS[@]}"; do
        python ${RULER_PATH}/data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${MODEL_NAME} \
            --tokenizer_type "hf" \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB}

        python ${RULER_PATH}/pred/call_api.py \
            --data_dir ${DATA_DIR} \
            --save_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --server_type ${MODEL_FRAMEWORK} \
            --model_name_or_path ${MODEL_NAME} \
            --temperature ${TEMPERATURE} \
            --top_k ${TOP_K} \
            --top_p ${TOP_P} \
            ${MINFERENCE_PARAMS} \
            ${EXTRA_PARAMS} \
            ${STOP_WORDS}
    done

    python ${RULER_PATH}/eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK}
done
