#!/bin/bash

python train.py \
--conf ../experiments/nuScenes/train_configs/config_ph_25_maxhl_24_minhl_24.json \
--preprocess_workers 10 \
--batch_size 64 \
--log_dir ../experiments/nuScenes/models \
--train_epochs 50 \
--log_tag _int_ee_sdc_ph_25_maxhl_24_min_hl_24_bs_64
