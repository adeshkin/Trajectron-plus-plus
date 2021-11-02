#!/bin/bash


for data_dir in /home/cds-k/Desktop/motion_prediction/motion_prediction_validation/processed/*
do
    if [[ $data_dir == *".pkl" ]]; then
      model=./models/int_ee_new_no_kalman_ph_10_maxhl_4
      model_name=$(basename $model)
      track=$(basename $data_dir)
      python evaluate_centerpoint_out.py \
        --model=$model \
        --checkpoint='model_registrar-20' \
        --data=$data_dir \
        --output_path=/home/cds-k/Desktop/motion_prediction/motion_prediction_validation/processed/results/$track \
        --output_tag=$model_name
    fi
done

