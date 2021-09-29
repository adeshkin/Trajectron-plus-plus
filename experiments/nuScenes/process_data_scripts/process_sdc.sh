#!/bin/bash

for mode in 'train' 'validation'
do
  python process_sdc.py \
    --data=/media/cds-k/Data_2/canonical-trn-dev-data/data \
    --version=$mode \
    --output_path=/media/cds-k/data/nuScenes/traj++_processed_data/processed_sdc_20000
done