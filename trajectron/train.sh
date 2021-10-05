#!/bin/bash

python train.py \
--conf ../experiments/nuScenes/train_configs/config_ph_25_maxhl_24_minhl_24.json \
--data_dir ../experiments/processed_data/processed_sdc \
--train_data_dict sdc_train.pkl \
--eval_data_dict sdc_validation.pkl \
--offline_scene_graph no \
--preprocess_workers 10 \
--batch_size 16 \
--log_dir ../experiments/nuScenes/models \
--train_epochs 50 \
--node_freq_mult_train \
--log_tag _int_ee_sdc_ph_25_maxhl_24_min_hl_24_bs_512_new_dataset
