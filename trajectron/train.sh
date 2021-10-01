#!/bin/bash

python train.py \
--eval_every 5 \
--conf ../experiments/nuScenes/train_configs/config_ph_25_maxhl_24_minhl_24.json \
--data_dir /media/cds-k/data/nuScenes/traj++_processed_data/processed_sdc \
--train_data_dict sdc_train.pkl \
--eval_data_dict sdc_validation.pkl \
--offline_scene_graph no \
--preprocess_workers 10 \
--batch_size 256 \
--log_dir ../experiments/nuScenes/models \
--train_epochs 10 \
--node_freq_mult_train \
--log_tag _int_ee_sdc_ph_25_maxhl_24_min_hl_24
