#!/bin/bash

python evaluate.py \
--model models/models_08_Sep_2021_12_18_51_int_ee \
--checkpoint=13 \
--data ../processed/nuScenes_test_full.pkl \
--output_path results \
--output_tag int_ee_13 \
--node_type VEHICLE \
--prediction_horizon 6