#!/bin/bash

mode='train'
data_dir=/home/cds-k/Desktop/centerpoint_out/centerpoint_out
python process_centerpoint_out.py \
  --data_dir=$data_dir \
  --mode=$mode \
  --output_path=/media/cds-k/data/nuScenes/traj++_processed_data/processed_centerpoint_out_augment
