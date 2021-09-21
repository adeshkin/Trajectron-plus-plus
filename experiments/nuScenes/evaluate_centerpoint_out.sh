#!/bin/bash


for data_dir in /media/cds-k/data/nuScenes/traj++_processed_data/processed_centerpoint_out_new_no_minus_50_dur_180s/*
do
    if [[ $data_dir == *".pkl" ]]; then
      model=./models/my_int_ee_new
      model_name=$(basename $model)
      python evaluate_centerpoint_out.py \
        --model=$model \
        --checkpoint=12 \
        --data=$data_dir \
        --output_path=/media/cds-k/data/nuScenes/traj++_results_centerpoint_out_ped/$model_name \
        --output_tag=$model_name
    fi
done

