#!/bin/bash

for mode in 'train'
do
  python process_sdc.py \
    --data=/media/cds-k/Data_2/canonical-trn-dev-data/data \
    --version=$mode \
    --output_path=/media/cds-k/data/nuScenes/traj++_processed_data/processed_sdc_train_all
done