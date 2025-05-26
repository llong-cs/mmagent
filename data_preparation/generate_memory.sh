#! /bin/bash

do_preprocessing="true"
if [ "$do_preprocessing" = "true" ]; then
    preprocessing="voice,face"
else
    preprocessing=""
fi

data_list="MLVU/1_plotQA,MLVU/2_needle,MLVU/3_ego,MLVU/4_count,MLVU/5_order,MLVU/6_anomaly_reco,MLVU/7_topic_reasoning,MLVU/8_sub_scene,MLVU/9_summary"
python3 data_preparation/generate_memory.py --data_list ${data_list} --preprocessing ${preprocessing} > logs/gm_mlvu.log

data_list="Video-MME"
python3 data_preparation/generate_memory.py --data_list ${data_list} --preprocessing ${preprocessing} > logs/gm_mme.log

# data_list="CZ_1,CZ_2,CZ_3,ZZ_1,ZZ_2,ZZ_3,ZZ_4,ZZ_5"
# python3 data_preparation/generate_memory.py --data_list ${data_list} --preprocessing ${preprocessing} > logs/gm_ytb.log