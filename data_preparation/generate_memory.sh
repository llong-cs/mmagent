#! /bin/bash

# Get arguments with defaults
do_preprocessing=${1:-"true"}
data_list=${2:-"MLVU/1_plotQA,MLVU/2_needle,MLVU/3_ego,MLVU/4_count,MLVU/5_order,MLVU/6_anomaly_reco,MLVU/7_topic_reasoning,MLVU/8_sub_scene,MLVU/9_summary"}
# data_list=${2:-"Video-MME"}
# data_list=${2:-"CZ_1,CZ_2,CZ_3,ZZ_1,ZZ_2,ZZ_3,ZZ_4,ZZ_5"}
log_dir=${3:-"logs"}

if [ "$do_preprocessing" = "true" ]; then
    preprocessing="voice,face"
else
    preprocessing=""
fi

python3 data_preparation/generate_memory.py --data_list ${data_list} --preprocessing ${preprocessing} > ${log_dir}/gm_${data_list//\//_}.log