#! /bin/bash
START_TIME=$SECONDS

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-250328

input_data_dir=$1
tokenizer=$2
json_keys=$3
output_data_dir=$4
load_dir=$5
dataset_name=$6

INPUT="${input_data_dir}"

if [ $tokenizer = "Qwen2Tokenizer" ]; then
  python preprocess_data_megatron.py \
  --input ${INPUT} \
  --output-prefix ${output_data_dir}/mmap_qwen2_datasets \
  --patch-tokenizer-type Qwen2Tokenizer \
  --json-keys ${json_keys} \
  --load ${load_dir} \
  --workers 2 \
  --partitions 2 \
  --keep-sequential-samples \
  --append-eod

elif [ $tokenizer = "Qwen3Tokenizer" ]; then
  python preprocess_data_megatron_qwen3.py \
  --input ${INPUT} \
  --output-prefix ${output_data_dir}/mmap_qwen3_datasets_test_arrow \
  --patch-tokenizer-type Qwen3Tokenizer \
  --json-keys ${json_keys} \
  --load ${load_dir} \
  --workers 64 \
  --partitions 1 \
  --keep-sequential-samples \
  --load_from_arrow \
  --append-eod

fi

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
