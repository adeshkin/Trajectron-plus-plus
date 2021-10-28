#!/bin/bash


data_dir="/home/cds-k/Desktop/centerpoint_out/motion_prediction_validation_2Hz/gt_validation.pkl"
model=./models/int_ee_ph_10_maxhl_4
model_name=$(basename $model)
track=$(basename $data_dir)
python evaluate_centerpoint_out.py \
  --model=$model \
  --checkpoint='model_registrar-20' \
  --data=$data_dir \
  --output_path=/home/cds-k/Desktop/centerpoint_out/motion_prediction_validation_2.5Hz/results \
  --output_tag=$model_name


