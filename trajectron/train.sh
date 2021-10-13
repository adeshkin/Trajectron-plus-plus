#!/bin/bash

python train_sdc.py \
--conf ../experiments/nuScenes/train_configs/config_ph_25_maxhl_24_minhl_24_map.json \
--preprocess_workers 6 \
--batch_size 32 \
--log_dir ../experiments/nuScenes/models \
--train_epochs 5 \
--log_tag _int_ee_sdc_ph_25_maxhl_24_min_hl_24_map_bs_32
