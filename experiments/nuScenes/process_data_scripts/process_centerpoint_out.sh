#!/bin/bash

data_dir=/home/cds-k/Desktop/centerpoint_out/val
for path in $data_dir/*
do
  if [[ $path == *".yaml" ]]; then
    filename=$(basename $path)
    echo $filename
    python process_centerpoint_out.py \
      --data_dir=$data_dir \
      --filename=$filename \
      --output_path=/media/cds-k/data/nuScenes/traj++_processed_data/processed_centerpoint_out_no_minus_50_dur_180s
  fi
done


#python process_centerpoint_out.py \
#	--data_dir=/home/cds-k/Desktop/centerpoint_out/val \
#	--filename="mipt_cifra_arktika_lol_2021-06-10-22-01-27_0" \
#	--output_path=/media/cds-k/data/nuScenes/traj++_processed_data/processed_centerpoint
