#!/bin/bash

python process_data.py \
  --data=/media/cds-k/data/nuScenes/v1.0-trainval \
  --version="v1.0-trainval" \
  --output_path=/media/cds-k/data/nuScenes/traj++_processed_data/processed_boris_new_no_kalman
