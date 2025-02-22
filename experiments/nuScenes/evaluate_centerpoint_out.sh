#!/bin/bash


for data_dir in /media/cds-k/data/nuScenes/traj++_processed_data/processed_centerpoint_out_no_minus_50_dur_180s/*
do
    if [[ $data_dir == *".pkl" ]]; then
      model=./models/int_ee_ph_10_maxhl_4
      model_name=$(basename $model)
      track=$(basename $data_dir)
      python evaluate_centerpoint_out.py \
        --model=$model \
        --checkpoint=20 \
        --data=$data_dir \
        --output_path=/media/cds-k/data/nuScenes/traj++_results_centerpoint_out_veh/$model_name/$track \
        --output_tag=$model_name
    fi
done

