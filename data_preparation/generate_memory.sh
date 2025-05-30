#! /bin/bash

# Get arguments with defaults
do_preprocessing=${1:-"true"}
data_list=${2:-"/mnt/bn/videonasi18n/longlin.kylin/mmagent/data/annotations/small_test.jsonl"}
log_dir=${3:-"data/logs"}

if [ "$do_preprocessing" = "true" ]; then
    preprocessing="voice,face"
else
    preprocessing=""
fi

python3 data_preparation/generate_memory.py --data_list ${data_list} --preprocessing ${preprocessing} --log_dir ${log_dir}

# an example to run the script
# bash data_preparation/generate_memory.sh true /mnt/bn/videonasi18n/longlin.kylin/mmagent/data/annotations/small_test.jsonl data/logs
